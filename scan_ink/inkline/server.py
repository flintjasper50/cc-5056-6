"""本地 HTTP 界面：浏览器中查看纸色场/墨层/合成，画笔标记可靠纸区与
保护细节，滑块改参数后从原件重新求值。原件只在内存中读取，从不覆盖。

仅依赖标准库 http.server + numpy/PIL，无第三方 web 框架。
"""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .model import Params
from .pipeline import Project
from .regions import RegionLayer
from .synthetic import SyntheticSpec, generate
from .triple_layer import (
    TripleSpec, build_triple_scene, inject_paper_error, invert_blend_order,
    coverage_agreement, zone_residuals,
)
from .util import to_float, to_u8

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_LOCK = threading.Lock()
STATE: dict = {"project": None, "view": "composite", "base_color": [0.97, 0.96, 0.93],
               "dark_color": [0.25, 0.27, 0.32], "triple": None,
               "fault": "none", "last_error": None}


def _png_bytes(arr: np.ndarray) -> bytes:
    im = Image.fromarray(to_u8(arr))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _current_views():
    proj: Project = STATE["project"]
    h, w = proj.image.shape[:2]
    dark = np.tile(np.array(STATE["dark_color"], np.float32), (h, w, 1))
    light = np.tile(np.array(STATE["base_color"], np.float32), (h, w, 1))
    return proj.views(base_dark=dark, base_light=light)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html")
            return
        if route == "/app.js":
            self._send(200, (WEB_DIR / "app.js").read_bytes(), "application/javascript")
            return
        if route == "/style.css":
            self._send(200, (WEB_DIR / "style.css").read_bytes(), "text/css")
            return
        if route == "/state":
            with _LOCK:
                proj = STATE["project"]
                if proj is None:
                    self._json({"ready": False})
                    return
                try:
                    metrics = proj.metrics()
                    STATE["last_error"] = None
                except Exception as exc:  # 运算异常经界面提示，不返回 500 崩页
                    STATE["last_error"] = f"{type(exc).__name__}: {exc}"
                    metrics = None
                self._json({"ready": True, "params": proj.params.to_dict(),
                            "view": STATE["view"],
                            "base_color": STATE["base_color"],
                            "dark_color": STATE["dark_color"],
                            "metrics": metrics,
                            "fault": STATE.get("fault", "none"),
                            "last_error": STATE["last_error"],
                            "shape": list(proj.image.shape[:2])})
            return
        if route == "/view":
            name = self.query("name")
            with _LOCK:
                if STATE["project"] is None:
                    self._send(404, b"no project", "text/plain")
                    return
                try:
                    views = _current_views()
                    STATE["last_error"] = None
                except Exception as exc:  # 实际运算失败：显式提示而非黑屏
                    STATE["last_error"] = f"{type(exc).__name__}: {exc}"
                    self._json({"error": STATE["last_error"]}, 500)
                    return
                if name not in views:
                    self._send(404, b"unknown view", "text/plain")
                    return
                self._send(200, _png_bytes(views[name]), "image/png")
            return
        self._send(404, b"not found", "text/plain")

    def query(self, key: str) -> str | None:
        from urllib.parse import parse_qs
        q = parse_qs(urlparse(self.path).query)
        return q.get(key, [None])[0]

    def do_POST(self):
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        with _LOCK:
            if route == "/load/demo":
                data = generate(SyntheticSpec(seed=int(payload.get("seed", 7)),
                                              fringe=bool(payload.get("fringe", False))))
                proj = Project(image=data["image"], name="synthetic")
                h, w = data["image"].shape[:2]
                rl = RegionLayer(w, h)
                rl.add_polygon("sky", (0.62, 0.78, 0.92),
                               [(0, 0), (w, 0), (w, h * 0.45), (0, h * 0.5)])
                rl.add_polygon("ground", (0.78, 0.88, 0.62),
                               [(0, h * 0.5), (w, h * 0.45), (w, h), (0, h)])
                proj.regions = rl
                STATE["project"] = proj
                self._json({"ok": True})
                return
            if route == "/load/upload":
                import base64
                raw = base64.b64decode(payload["png_b64"])
                img = to_float(Image.open(io.BytesIO(raw)))
                STATE["project"] = Project(image=img, name="upload")
                STATE["triple"] = None
                STATE["fault"] = "none"
                STATE["last_error"] = None
                self._json({"ok": True})
                return
            if route == "/load/triple":
                # 回归样例：透印/淡墨/网纹三重重合图，带独立参照与分区
                scene = build_triple_scene(TripleSpec(
                    seed=int(payload.get("seed", 23))))
                paper = scene["paper_gt"]
                fault = payload.get("fault", "none")
                if fault == "paper_error":
                    paper = inject_paper_error(scene)
                proj = Project(image=scene["image"], paper_override=paper,
                               name="triple")
                STATE["project"] = proj
                STATE["triple"] = scene
                STATE["fault"] = fault
                STATE["last_error"] = None
                self._json({"ok": True, "fault": fault,
                            "zones": {str(int(k)): int((scene["zones"] == k).sum())
                                      for k in np.unique(scene["zones"])},
                            "coverage_agreement": coverage_agreement(
                                scene["analytic_base"],
                                scene["coverage_ss_geom"])})
                return
            if route == "/fault":
                # 在已载入的三重层样例上切换受控注错，重新分离
                scene = STATE.get("triple")
                if scene is None:
                    self._json({"error": "请先载入三重层回归样例"}, 400)
                    return
                fault = payload.get("fault", "none")
                if fault == "order_error":
                    # 次序注错改变的是参照重合成，观测不变；保存证据供前端查看
                    STATE["fault"] = fault
                    wrong = invert_blend_order(scene)
                    table = zone_residuals(
                        np.abs(scene["image"] - wrong).mean(axis=2),
                        scene["zones"])
                    STATE["last_error"] = None
                    self._json({"ok": True, "fault": fault,
                                "order_error_residual": table})
                    return
                paper = scene["paper_gt"]
                if fault == "paper_error":
                    paper = inject_paper_error(scene)
                proj = Project(image=scene["image"], paper_override=paper,
                               name="triple")
                STATE["project"] = proj
                STATE["fault"] = fault
                STATE["last_error"] = None
                self._json({"ok": True, "fault": fault})
                return
            proj = STATE["project"]
            if proj is None:
                self._json({"error": "no project"}, 400)
                return
            if route == "/params":
                try:
                    proj.params = Params.from_dict(payload)
                    proj.invalidate()
                    metrics = proj.metrics()
                    STATE["last_error"] = None
                except Exception as exc:
                    STATE["last_error"] = f"{type(exc).__name__}: {exc}"
                    self._json({"error": STATE["last_error"]}, 500)
                    return
                self._json({"ok": True, "metrics": metrics})
                return
            if route == "/mark":
                pts = [(float(x), float(y)) for x, y in payload.get("points", [])]
                radius = int(payload.get("radius", 10))
                if payload.get("kind") == "reliable":
                    proj.mark_reliable(pts, radius, additive=payload.get("add", True))
                else:
                    proj.mark_protect(pts, radius, additive=payload.get("add", True))
                self._json({"ok": True})
                return
            if route == "/marks/clear":
                proj.clear_marks()
                self._json({"ok": True})
                return
            if route == "/view":
                STATE["view"] = payload.get("view", "composite")
                self._json({"ok": True})
                return
            if route == "/base":
                if "base_color" in payload:
                    STATE["base_color"] = [float(v) for v in payload["base_color"]]
                if "dark_color" in payload:
                    STATE["dark_color"] = [float(v) for v in payload["dark_color"]]
                self._json({"ok": True})
                return
            if route == "/region":
                h, w = proj.image.shape[:2]
                rl = proj.regions or RegionLayer(w, h, tuple(STATE["base_color"]))
                rl.add_polygon(payload["name"], tuple(payload["color"]),
                               [tuple(p) for p in payload["points"]],
                               feather=float(payload.get("feather", 1.2)))
                proj.regions = rl
                proj.invalidate()
                self._json({"ok": True})
                return
            if route == "/save":
                target = Path(payload["path"])
                proj.save(target)
                self._json({"ok": True, "path": str(target)})
                return
        self._send(404, b"not found", "text/plain")


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"扫描线稿着色合成器界面: http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    serve(args.host, args.port)
