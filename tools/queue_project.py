"""Run numbered ComfyUI API exports one at a time and keep the resulting media."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


def request(server, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    return urlopen(Request(server + path, data=data, headers={"Content-Type": "application/json"}),
                   timeout=25)


def api(server, path, payload=None):
    with request(server, path, payload) as response:
        return json.load(response)


def write_receipt(path, receipt):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


@contextmanager
def lock_project(folder):
    with (folder / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("此项目已有一个队列进程") from exc
        yield


def load_shots(folder):
    shots = []
    numbers = set()
    for path in sorted((folder / "shots").glob("*.json")):
        match = re.match(r"([0-9]+)[-_]", path.stem)
        if not match or int(match[1]) in numbers:
            raise ValueError(f"{path.name} 需要独立的数字序号，例如 001-opening.json")
        number = int(match[1])
        numbers.add(number)
        graph = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(graph, dict) or not graph or not all(
            isinstance(node, dict) and isinstance(node.get("class_type"), str)
            and isinstance(node.get("inputs"), dict) for node in graph.values()
        ):
            raise ValueError(f"{path.name} 需要 ComfyUI 的「导出（API）」JSON")
        canonical = json.dumps(graph, sort_keys=True, ensure_ascii=False).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        receipt_path = folder / "receipts" / path.name
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
        if receipt and receipt["graph_sha256"] != digest:
            raise ValueError(f"{path.name} 导出内容已变化，请用新的镜头编号提交修订版")
        shots.append((number, path, graph, digest, receipt_path, receipt))
    if not shots:
        raise ValueError(f"{folder / 'shots'} 中没有镜头")
    return sorted(shots, key=lambda shot: shot[0])


def collect_media(server, folder, shot, history):
    media = []
    for output in history["outputs"].values():
        for items in output.values():
            if isinstance(items, list):
                media.extend(item for item in items if isinstance(item, dict)
                             and item.get("type") == "output" and item.get("filename"))
    if not media:
        raise ValueError(f"{shot} 没有保存到输出目录的素材，请使用保存图片或视频的输出节点")
    paths = []
    target = folder / "renders" / shot
    target.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(media, start=1):
        filename = Path(item["filename"]).name
        if item["filename"] != filename or filename in {".", ".."}:
            raise ValueError(f"ComfyUI 返回了无效文件名 {item['filename']}")
        dest = target / f"{index:02d}-{filename}"
        pending = dest.with_suffix(".partial")
        query = urlencode({"filename": filename, "type": "output", "subfolder": item.get("subfolder", "")})
        with request(server, "/view?" + query) as response, pending.open("wb") as binary:
            shutil.copyfileobj(response, binary)
        pending.replace(dest)
        paths.append(str(dest.relative_to(folder)))
    return paths


def submit(server, graph, digest, path):
    prompt_id = str(uuid4())
    receipt = {"graph_sha256": digest, "prompt_id": prompt_id, "state": "submitting"}
    write_receipt(path, receipt)
    try:
        reply = api(server, "/prompt", {"prompt": graph, "prompt_id": prompt_id, "client_id": prompt_id})
    except HTTPError as exc:
        if 400 <= exc.code < 500:
            receipt.update(state="failed", error=exc.read().decode(errors="replace"))
            write_receipt(path, receipt)
        raise
    receipt.update(prompt_id=reply["prompt_id"], state="queued")
    write_receipt(path, receipt)
    return receipt


def find_job(server, prompt_id):
    for _ in range(3):
        history = api(server, "/history/" + prompt_id).get(prompt_id)
        if history is not None:
            return history, True
        queue = api(server, "/queue")
        if any(job[1] == prompt_id for section in ("queue_running", "queue_pending")
               for job in queue[section]):
            return None, True
        time.sleep(0.5)
    return None, False


def run_project(args, shots):
    remaining = args.limit
    for _, path, graph, digest, receipt_path, receipt in shots:
        state = receipt.get("state")
        if state == "complete":
            if any(not (args.project / file).is_file() for file in receipt["files"]):
                raise ValueError(f"{path.stem} 回执已完成，素材文件缺失，请从归档恢复")
            print(f"已有结果  {path.stem}", flush=True)
            continue
        if remaining == 0:
            return
        if state == "failed":
            if not args.retry_failed:
                raise ValueError(f"{path.stem} 上次失败，请检查回执后使用 --retry-failed")
            receipt = {}
        if receipt:
            _, known = find_job(args.server, receipt["prompt_id"])
            if not known:
                if args.retry_missing != path.stem:
                    raise ValueError(f"{path.stem} 的任务不在当前服务器队列或历史中。"
                                     "确认旧服务器已停止且没有出片后，再使用 --retry-missing 镜头名")
                receipt = {}
        if not receipt:
            receipt = submit(args.server, graph, digest, receipt_path)
        prompt_id = receipt["prompt_id"]
        print(f"等待结果  {path.stem}  {prompt_id}", flush=True)
        while True:
            history, known = find_job(args.server, prompt_id)
            if history is not None:
                break
            if not known:
                raise ValueError(f"{path.stem} 的任务从队列消失，已保留回执")
            time.sleep(args.poll_seconds)
        status = history["status"]
        if status["status_str"] != "success":
            receipt.update(state="failed", status=status)
            write_receipt(receipt_path, receipt)
            raise ValueError(f"{path.stem} 执行失败，请查看 {receipt_path}")
        files = collect_media(args.server, args.project, path.stem, history)
        receipt.update(state="complete", status=status, files=files, outputs=history["outputs"])
        write_receipt(receipt_path, receipt)
        print(f"已保存    {path.stem}  {', '.join(files)}", flush=True)
        if remaining is not None:
            remaining -= 1


def main():
    parser = argparse.ArgumentParser(description="按镜头编号排队，暂停后可从已有回执继续。")
    parser.add_argument("project", type=Path, help="项目目录，镜头放在 shots/ 中")
    parser.add_argument("--server", default="http://127.0.0.1:6006", help="自托管 ComfyUI 地址")
    parser.add_argument("--check", action="store_true", help="检查并列出顺序，无需连接服务器")
    parser.add_argument("--limit", type=int, help="这次最多完成的镜头数")
    parser.add_argument("--retry-failed", action="store_true", help="重新提交上次运行失败的镜头")
    parser.add_argument("--retry-missing", metavar="镜头名", help="重新提交服务器已找不到的指定任务")
    parser.add_argument("--poll-seconds", type=float, default=5, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.poll_seconds <= 0:
        parser.error("镜头数和轮询间隔须为正数")
    args.project = args.project.resolve()
    args.server = args.server.rstrip("/")
    try:
        if args.check:
            for _, path, _, _, _, receipt in load_shots(args.project):
                print(f"{receipt.get('state', 'pending'):>10}  {path.stem}")
            return
        (args.project / "receipts").mkdir(parents=True, exist_ok=True)
        with lock_project(args.project):
            run_project(args, load_shots(args.project))
    except KeyboardInterrupt:
        sys.exit("已停止提交新镜头。服务器上的当前任务可在下次续跑时接回。")
    except (OSError, ValueError) as exc:
        sys.exit(f"队列暂停：{exc}")


if __name__ == "__main__":
    main()
