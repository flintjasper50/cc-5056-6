const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const viewName = document.getElementById("view-name");
const coordEl = document.getElementById("coord");
const metricsEl = document.getElementById("metrics");

let currentView = "composite";
let params = {};
let drawing = false;
let lastPoint = null;
let regionPoints = [];
let regionMode = false;
const viewLabels = {
  original: "原件（永不覆盖）", paper_field: "纸色场 P", confidence: "可信度 c",
  ink_layer_gray: "墨层（中灰底，显浅笔）", alpha: "有效覆盖率 b=a·t",
  coverage: "几何覆盖率 a", strength: "笔触强度 t",
  composite_light: "浅底重铺", composite_dark: "深底重铺（查白边）",
  composite: "最终着色（区域色 + 墨层）", region_layer: "区域颜色层",
  recomp_orig: "原底重建（与原件核对）", residual_heat: "残差热图",
  missing_heat: "失墨热图",
};

async function api(path, body) {
  const opt = body ? {method: "POST", headers: {"Content-Type": "application/json"},
                      body: JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch (e) { data = {}; }
  if (!r.ok || (data && data.error)) {
    showError(data.error || `请求失败 (${r.status})`);
    throw new Error(data.error || r.statusText);
  }
  clearError();
  return data;
}

const errorBanner = document.getElementById("error-banner");
const faultBanner = document.getElementById("fault-banner");
function showError(msg) {
  errorBanner.textContent = `运算异常：${msg}`;
  errorBanner.hidden = false;
}
function clearError() { errorBanner.hidden = true; }
function showFault(fault, extra) {
  if (!fault || fault === "none") { faultBanner.hidden = true; return; }
  const label = fault === "paper_error" ? "已注入错误底色估计" : "已颠倒混合次序";
  faultBanner.textContent = extra ? `${label}：${extra}` : label;
  faultBanner.hidden = false;
}

function refreshView() {
  const url = `/view?name=${currentView}&t=${Date.now()}`;
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    if (regionMode && regionPoints.length) drawRegionPreview();
  };
  img.onerror = async () => {
    // 视图运算返回 500 JSON：拉取 /state 取回后端记录的异常信息
    try { const s = await fetch("/state").then(x => x.json());
      if (s.last_error) showError(s.last_error); } catch (e) {}
  };
  img.src = url;
  viewName.textContent = viewLabels[currentView] || currentView;
}

async function refreshState() {
  const s = await api("/state");
  if (!s.ready) return;
  params = s.params;
  for (const [k, v] of Object.entries(params)) {
    const el = document.querySelector(`input[data-k="${k}"]`);
    const lab = document.getElementById(`v-${k}`);
    if (el) { el.value = v; if (lab) lab.textContent = v; }
  }
  metricsEl.textContent = s.metrics
    ? JSON.stringify(s.metrics, null, 2)
    : "指标不可用（运算异常，见顶部提示）";
  if (s.last_error) showError(s.last_error); else clearError();
  showFault(s.fault);
}

function tool() {
  return document.querySelector('input[name="tool"]:checked').value;
}
function eventPos(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * canvas.width / r.width,
          (e.clientY - r.top) * canvas.height / r.height];
}

canvas.addEventListener("mousedown", (e) => {
  const p = eventPos(e);
  if (regionMode) { regionPoints.push(p); drawRegionPreview(); return; }
  if (tool() === "nav") return;
  drawing = true; lastPoint = p;
  void sendStroke([p]);
});
canvas.addEventListener("dblclick", async () => {
  if (!regionMode || regionPoints.length < 3) return;
  const color = hexToRgb(document.getElementById("region-color").value);
  await api("/region", {name: `region-${Date.now()}`, color,
                        points: regionPoints, feather: 1.2});
  regionPoints = []; regionMode = false;
  overlay.style.display = "none";
  await refreshState(); refreshView();
});
canvas.addEventListener("mousemove", async (e) => {
  const p = eventPos(e);
  coordEl.textContent = `x=${p[0].toFixed(0)} y=${p[1].toFixed(0)}`;
  if (!drawing || tool() === "nav") return;
  await sendStroke([lastPoint, p]);
  lastPoint = p;
});
window.addEventListener("mouseup", () => { drawing = false; lastPoint = null; });

async function sendStroke(points) {
  const radius = Number(document.getElementById("brush").value);
  await api("/mark", {kind: tool(), points, radius, add: true});
  refreshView();
  refreshState();
}

function drawRegionPreview() {
  ctx.save();
  ctx.strokeStyle = "#ff5d5d"; ctx.fillStyle = "rgba(255,93,93,.25)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  regionPoints.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
  regionPoints.forEach((p) => { ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 7); ctx.fill(); });
  ctx.restore();
}

document.getElementById("view-buttons").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-view]");
  if (!btn) return;
  currentView = btn.dataset.view;
  document.querySelectorAll("#view-buttons button").forEach(
    b => b.classList.toggle("active", b === btn));
  await api("/view", {view: currentView});
  refreshView();
});

document.querySelectorAll("input[data-k]").forEach((el) => {
  el.addEventListener("change", async () => {
    params[el.dataset.k] = Number(el.value);
    document.getElementById(`v-${el.dataset.k}`).textContent = el.value;
    const res = await api("/params", params);
    metricsEl.textContent = JSON.stringify(res.metrics, null, 2);
    refreshView();
  });
});

document.getElementById("light-color").addEventListener("change", async (e) => {
  await api("/base", {base_color: hexToRgb(e.target.value)});
  refreshView();
});
document.getElementById("dark-color").addEventListener("change", async (e) => {
  await api("/base", {dark_color: hexToRgb(e.target.value)});
  refreshView();
});

document.getElementById("btn-demo").addEventListener("click", async () => {
  await api("/load/demo", {seed: 7});
  faultBanner.hidden = true;
  await refreshState(); refreshView();
});
document.getElementById("btn-triple").addEventListener("click", async () => {
  const res = await api("/load/triple", {seed: 23, fault: "none"});
  showFault("none");
  if (res.coverage_agreement) {
    metricsEl.textContent =
      `两路参照一致性：\n${JSON.stringify(res.coverage_agreement, null, 2)}`;
  }
  refreshView();
});
document.getElementById("fault-buttons").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-fault]");
  if (!btn) return;
  const fault = btn.dataset.fault;
  document.querySelectorAll("#fault-buttons button").forEach(
    b => b.classList.toggle("active", b === btn));
  try {
    const res = await api("/fault", {fault});
    showFault(fault, res.order_error_residual
      ? `重合区残差=${res.order_error_residual.triple_overlap?.toFixed(4)
         ?? "n/a"}，纯纸区=${res.order_error_residual.clean_paper?.toFixed(4) ?? "n/a"}`
      : "");
    await refreshState(); refreshView();
  } catch (err) { /* 横幅已提示 */ }
});
document.getElementById("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const buf = new Uint8Array(await file.arrayBuffer());
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < buf.length; i += chunk) {
    bin += String.fromCharCode.apply(null, buf.subarray(i, i + chunk));
  }
  await api("/load/upload", {png_b64: btoa(bin)});
  await refreshState(); refreshView();
});
document.getElementById("btn-clear-marks").addEventListener("click", async () => {
  await api("/marks/clear", {});
  refreshView(); refreshState();
});
document.getElementById("btn-save").addEventListener("click", async () => {
  const path = prompt("保存目录", "output/saved_project");
  if (path) { const r = await api("/save", {path}); alert(r.ok ? "已保存" : r.error); }
});
document.getElementById("btn-region").addEventListener("click", () => {
  regionMode = !regionMode;
  regionPoints = [];
  overlay.textContent = regionMode ? "单击放置顶点，双击闭合区域" : "";
  overlay.style.display = regionMode ? "block" : "none";
  if (!regionMode) refreshView();
});

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

document.querySelector('button[data-view="composite"]').classList.add("active");
refreshState().then(refreshView);
