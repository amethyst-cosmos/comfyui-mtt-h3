import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const backends = { s4000: "摩尔线程 S4000", "910c": "华为昇腾 910C" };
const sourceNames = { existing: "使用已有关键帧", generate: "H3 Image 生成关键帧" };
const draft = {
  motion: "", dialogue: "", first_prompt: "", last_prompt: "", rate: "",
};
let currentDialog;

function field(form, label, input) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = "display:grid;gap:4px;margin:10px 0";
  wrapper.append(label, input);
  form.append(wrapper);
  return wrapper;
}

function choose(select, options, value) {
  select.replaceChildren(...options.map(([key, title]) => new Option(title, key)));
  if (options.some(([key]) => key === value)) select.value = value;
}

async function request(path, options) {
  const response = await api.fetchApi("/h3-workbench/" + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "无法读取工作流");
  return data;
}

async function openSetup() {
  if (currentDialog) return;
  const dialog = document.createElement("dialog");
  currentDialog = dialog;
  dialog.setAttribute("aria-labelledby", "h3-setup-title");
  dialog.style.cssText = [
    "width:min(560px,calc(100vw - 32px));max-height:calc(100dvh - 32px);box-sizing:border-box;overflow:auto",
    "padding:20px;border:1px solid var(--border-color,#777);border-radius:12px",
    "color:var(--input-text,#ddd);background:var(--comfy-menu-bg,#222);z-index:9999",
  ].join(";");
  const heading = document.createElement("h2");
  heading.id = "h3-setup-title";
  heading.textContent = "新建 H3 镜头";
  heading.style.marginTop = "0";
  const form = document.createElement("form");
  const device = document.createElement("select");
  const quality = document.createElement("select");
  const source = document.createElement("select");
  field(form, "运行设备", device);
  field(form, "画质档", quality);
  field(form, "关键帧来源", source);
  const diagnostics = document.createElement("details");
  const status = document.createElement("summary");
  status.setAttribute("aria-live", "polite");
  const list = document.createElement("ul");
  list.style.overflowWrap = "anywhere";
  diagnostics.append(status, list);
  form.append(diagnostics);
  const attentionNote = document.createElement("p");
  attentionNote.textContent = "910C 会自动分开运行生图和视频，首尾帧直接交接。";
  attentionNote.hidden = true;
  form.append(attentionNote);
  const inputs = {};
  for (const [key, label, rows] of [
    ["first_prompt", "首帧画面描述", 2], ["last_prompt", "尾帧变化描述", 2],
    ["motion", "动作、机位与声音描述", 3], ["dialogue", "对白（可选）", 1],
  ]) {
    const input = document.createElement("textarea");
    input.rows = rows;
    input.value = draft[key];
    input.addEventListener("input", () => { draft[key] = input.value; });
    inputs[key] = { input, area: field(form, label, input) };
  }
  inputs.motion.input.required = true;
  const rate = document.createElement("input");
  rate.type = "number";
  rate.min = "0";
  rate.step = "any";
  rate.value = draft.rate;
  field(form, "算力小时价（元，可选）", rate);
  const estimate = document.createElement("p");
  estimate.setAttribute("aria-live", "polite");
  const errorBox = document.createElement("p");
  errorBox.setAttribute("role", "alert");
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end";
  const refresh = document.createElement("button");
  refresh.type = "button";
  refresh.textContent = "重新检测";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "关闭";
  cancel.addEventListener("click", () => dialog.close());
  const open = document.createElement("button");
  open.type = "submit";
  open.textContent = "打开工作流";
  open.disabled = true;
  actions.append(refresh, cancel, open);
  form.append(estimate, errorBox, actions);
  dialog.append(heading, form);
  document.body.append(dialog);
  dialog.addEventListener("close", () => {
    currentDialog = undefined;
    dialog.remove();
  });
  dialog.showModal();

  let catalog;

  function updateEstimate(profile) {
    if (!profile) return;
    const cost = rate.value !== "" && rate.validity.valid
      ? " · 约 ¥" + (profile.seconds / 3600 * Number(rate.value)).toFixed(2)
      : "";
    estimate.textContent = `参考耗时约 ${Math.round(profile.seconds)} 秒${cost}（${profile.evidence}，仅视频推理）`;
  }

  function updateSource() {
    const profile = catalog[quality.value];
    const generate = source.value === "generate";
    attentionNote.hidden = profile.backend !== "910c";
    for (const key of ["first_prompt", "last_prompt"]) {
      const shouldShow = generate && (key === "first_prompt" || profile.requires_last);
      inputs[key].area.hidden = !shouldShow;
      inputs[key].area.style.display = shouldShow ? "grid" : "none";
      inputs[key].input.required = shouldShow;
    }
    const checks = profile.checks[source.value];
    const ready = checks.every((item) => item.ok);
    status.textContent = `${ready ? "预检通过" : "需要配置"} · ${profile.model} · ${profile.required_frames}`;
    diagnostics.open = !ready;
    open.textContent = ready ? "打开工作流" : "打开工作流（待配置）";
    list.replaceChildren(...checks.map(({ label, ok, detail }) => {
      const row = document.createElement("li");
      row.textContent = `${ok ? "通过" : "待处理"} · ${label} · ${detail}`;
      row.style.whiteSpace = "pre-line";
      return row;
    }));
    updateEstimate(profile);
  }

  function updateProfile() {
    const profile = catalog[quality.value];
    const firstReadySource = profile.sources.find((name) => profile.checks[name].every((item) => item.ok));
    choose(source, profile.sources.map((name) => [name, sourceNames[name]]),
      profile.sources.includes(source.value) ? source.value : firstReadySource);
    updateSource();
  }

  function updateDevice(preferred) {
    const options = Object.entries(catalog).filter(([, profile]) => profile.backend === device.value);
    const ready = options.find(([, profile]) => profile.ready);
    choose(quality, options.map(([key, profile]) => [key, profile.title]),
      preferred || ready?.[0]);
    updateProfile();
  }

  async function detect() {
    refresh.disabled = true;
    open.disabled = true;
    errorBox.textContent = "";
    status.textContent = "正在检测当前 GPU 实例…";
    try {
      const data = await request("profiles");
      const selected = quality.value || data.suggested;
      catalog = data.profiles;
      choose(device, Object.entries(backends), catalog[selected].backend);
      updateDevice(selected);
      open.disabled = false;
    } catch (error) {
      errorBox.textContent = error.message;
      status.textContent = "设备检测失败";
    } finally {
      refresh.disabled = false;
    }
  }

  device.addEventListener("change", () => updateDevice());
  quality.addEventListener("change", updateProfile);
  source.addEventListener("change", updateSource);
  rate.addEventListener("input", () => {
    draft.rate = rate.value;
    updateEstimate(catalog?.[quality.value]);
  });
  refresh.addEventListener("click", detect);
  async function loadWorkflow(graph, name) {
    const loaded = await app.handleFile(new File([JSON.stringify(graph)], name, { type: "application/json" }));
    if (loaded === false) throw new Error("ComfyUI 未能打开新工作流。");
    dialog.close();
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    open.disabled = true;
    errorBox.textContent = "";
    try {
      const { motion, dialogue, first_prompt, last_prompt } = draft;
      const profileKey = quality.value;
      const prepared = await request("workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile_key: profileKey, source: source.value, motion, dialogue, first_prompt, last_prompt,
        }),
      });
      const name = `H3 镜头 · ${catalog[profileKey].title} · ${Date.now()}.json`;
      await loadWorkflow(prepared, name);
    } catch (error) {
      errorBox.textContent = error.message;
      open.disabled = false;
    }
  });
  await detect();
}

app.registerExtension({
  name: "H3WorkBench.ShotSetup",
  async setup() {
    await request("frontend-ready", { method: "POST" }).catch(() => {});
  },
  commands: [{ id: "H3WorkBench.NewShot", label: "新建 H3 镜头", function: openSetup }],
  menuCommands: [{ path: ["H3 短片"], commands: ["H3WorkBench.NewShot"] }],
});
