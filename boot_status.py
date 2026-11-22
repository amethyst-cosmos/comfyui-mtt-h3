"""Serve startup diagnostics independently of the ComfyUI process."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.request import urlopen

import image_runtime
import runtime


PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>H3 短片启动诊断</title>
<style>
:root{color-scheme:dark;background:#111719;color:#e7f1ee;font:16px system-ui,sans-serif}
body{margin:0 auto;padding:clamp(20px,6vw,72px) 22px;max-width:720px}
h1{font-size:clamp(25px,4vw,38px)} a,button{color:#aff2da}
li{background:#1d282a;border-radius:8px;margin:9px 0;padding:12px;overflow-wrap:anywhere}
ul{list-style:none;padding:0} .ok{border-left:4px solid #5ed6ab} .pending{border-left:4px solid #e4ad63}
button{font:inherit;background:#1d282a;border:1px solid #627670;border-radius:6px;padding:7px 12px}
</style>
<h1>H3 短片启动诊断</h1>
<p id="message" aria-live="polite">正在读取实例状态…</p>
<ul id="checks" aria-label="启动检查"></ul>
<button id="refresh" type="button">立即检查</button>
<p id="workspace"></p>
<script>
const checks = document.getElementById('checks');
const message = document.getElementById('message');
const workspace = document.getElementById('workspace');
const refresh = document.getElementById('refresh');
let busy = false;
async function read() {
  if (busy) return;
  busy = true;
  refresh.disabled = true;
  try {
    const res = await fetch('status', {cache:'no-store'});
    if (!res.ok) throw Error();
    const data = await res.json();
    message.textContent = data.summary;
    checks.replaceChildren(...data.checks.map(check => {
      const row = document.createElement('li');
      row.className = check.ok ? 'ok' : 'pending';
      row.textContent = (check.ok ? '通过 · ' : '待处理 · ') + check.name + ' · ' + check.detail;
      return row;
    }));
    workspace.replaceChildren();
    if (data.open_url) {
      const link = document.createElement('a');
      link.href = data.open_url;
      link.textContent = '打开 ComfyUI';
      workspace.append(link);
    } else {
      workspace.textContent = '在实例面板点击 WebUI-' + data.comfy_port + ' 打开工作台。';
    }
  } catch {
    message.textContent = '暂时读不到诊断服务，可重新检查。';
  } finally {
    busy = false;
    refresh.disabled = false;
  }
}
refresh.addEventListener('click', read);
read();
setInterval(read, 5000);
</script></html>""".encode()


def http_json(port, endpoint):
    with urlopen(f"http://127.0.0.1:{port}/api/{endpoint}", timeout=2) as response:
        return json.load(response)


def check(name, operation, success):
    try:
        operation()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        return {"name": name, "ok": False, "detail": str(error)}
    return {"name": name, "ok": True, "detail": success}


def probe(args, process):
    def image():
        engine, _ = image_runtime.image_paths(check_device=False)
        image_runtime.check_image_cli(engine)

    output = Path(os.environ.get("COMFYUI_OUTPUT", "/root/autodl-tmp/mtt-h3-output"))

    def writable():
        if not output.is_dir() or not os.access(output, os.W_OK | os.X_OK):
            raise RuntimeError(f"输出目录未就绪 · {output}")

    def registered():
        node = http_json(args.comfy_port, "object_info/MttS4000H3Image")
        if "MttS4000H3Image" not in node:
            raise RuntimeError("ComfyUI 已响应，S4000 生图节点尚未注册。")

    def opened():
        if not http_json(args.comfy_port, "h3-workbench/frontend-ready")["opened"]:
            raise RuntimeError("还未收到浏览器成功打开工作台的信号。")

    if os.getenv("H3_WORKBENCH_BACKEND") == "910c":
        try:
            catalog = http_json(args.comfy_port, "h3-workbench/profiles")
            profiles = catalog["profiles"]
            available = [key for key in ("cann-bf16", "cann-int8")
                         if all(item["ok"] for item in profiles[key]["checks"]["generate"])]
            selected = profiles[available[0] if available else "cann-bf16"]
            checks = [{"name": item["label"], "ok": item["ok"], "detail": item["detail"]}
                      for item in selected["checks"]["generate"]]
        except (OSError, ValueError, KeyError):
            checks = [{"name": "910C 工作台", "ok": False, "detail": "等待 ComfyUI 加载昇腾节点。"}]
    else:
        checks = [
            check("S4000 设备", runtime.check_s4000, "已识别单张 S4000"),
            check("共享盘与图像权重", image, "图像主模型、编码器、图像 VAE 可读，生图入口存在"),
            check("高清视频运行文件", lambda: runtime.runtime_paths(tuple(runtime.PROFILES)[1], device=False),
                  "首尾帧程序、Flash 桥、模型、LoRA、FFmpeg 和 FFprobe 已找到"),
            check("输出目录", writable, "可写"),
            check("ComfyUI 后端", registered, "S4000 生图节点已注册"),
        ]
    checks.append(check("浏览器工作台", opened, "已有浏览器完成工作台扩展初始化"))
    code = process.poll()
    summary = (f"ComfyUI 已退出（{code}），请检查启动日志 · {args.log}" if code is not None
               else "启动检查通过。生成能力以实卡样片验收为准。" if all(item["ok"] for item in checks)
               else "请查看待处理项。硬件与权重检查不会提交生成任务。")
    return {"summary": summary, "checks": checks, "comfy_port": args.comfy_port,
            "open_url": os.environ.get("H3_COMFY_PUBLIC_URL", "")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--comfy-port", type=int, default=6006)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing ComfyUI command")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    data = {"summary": "正在启动 ComfyUI…", "checks": [], "comfy_port": args.comfy_port}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, kind = PAGE, "text/html; charset=utf-8"
            elif self.path == "/status":
                body = json.dumps(data, ensure_ascii=False).encode()
                kind = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer((args.listen, args.port), Handler) as server, args.log.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            while not stop.is_set():
                try:
                    data = probe(args, process)
                except Exception as error:
                    data = {"summary": f"诊断失败 · {error}", "checks": [],
                            "comfy_port": args.comfy_port}
                stop.wait(5)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
