"""Native desktop UI for Real-ESRGAN video upscaling.

A pywebview window (native Win11 WebView2) loads a plain HTML/JS frontend from a
loopback HTTP server (127.0.0.1, no firewall prompt). No Node, no build, no
Electron. The upscale runs via inference_realesrgan.py using THIS interpreter, so
whatever Python you launch this with must have realesrgan installed.

Run:  uv run python ui/app.py        (from the repo root, in the env with realesrgan)
"""
import http.server
import json
import mimetypes
import os
import socketserver
import threading
import urllib.parse

import webview
from webview.dom import DOMEventHandler

from pipeline import Cancelled, Pipeline

UI_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(UI_DIR, "web")
REPO_ROOT = os.path.dirname(UI_DIR)  # ui/ -> repo root (has inference_realesrgan.py)
PORT = None  # set in main()
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
SETTINGS_DIR = os.path.join(os.environ.get("APPDATA", UI_DIR), "Real-ESRGAN Upscaler")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

# pywebview 5.x renamed the dialog enums; support both old and new.
_FD = getattr(webview, "FileDialog", None)
OPEN_DIALOG = _FD.OPEN if _FD else webview.OPEN_DIALOG
FOLDER_DIALOG = _FD.FOLDER if _FD else webview.FOLDER_DIALOG


# ── loopback static + media server (Range support so <video> can seek) ────────
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/media":
            qs = urllib.parse.parse_qs(parsed.query)
            path = (qs.get("path") or [None])[0]
            if not path or not os.path.isfile(path):
                self.send_error(404)
                return
            self._serve(path)
        else:
            rel = parsed.path.lstrip("/") or "index.html"
            full = os.path.normpath(os.path.join(WEB_DIR, rel))
            if not full.startswith(WEB_DIR) or not os.path.isfile(full):
                self.send_error(404)
                return
            self._serve(full)

    def _serve(self, path):
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            partial = True
            try:
                s, e = rng[len("bytes="):].split("-")
                start = int(s) if s else 0
                end = int(e) if e else size - 1
            except Exception:
                start, end = 0, size - 1
        end = min(end, size - 1)
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    do_HEAD = do_GET


def start_server():
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


# ── JS-facing API ─────────────────────────────────────────────────────────────
class Api:
    def __init__(self):
        self._window = None      # underscore: keep pywebview from serializing it (recursion)
        self._pipeline = None
        self._worker = None

    def _video_items(self, paths):
        out = []
        for p in paths:
            if not p or not os.path.isfile(p) or os.path.splitext(p)[1].lower() not in VIDEO_EXTS:
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                size = 0
            out.append({"path": p, "name": os.path.basename(p), "size": size, "url": self.media_url(p)})
        return out

    def pick_videos(self):
        types = ("Video files (*.mp4;*.mkv;*.avi;*.mov;*.webm;*.flv;*.wmv;*.m4v)", "All files (*.*)")
        res = self._window.create_file_dialog(OPEN_DIALOG, allow_multiple=True, file_types=types)
        return self._video_items(res or [])

    def pick_folder(self):
        res = self._window.create_file_dialog(FOLDER_DIALOG)
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else res

    def pick_model(self):
        types = ("Model weights (*.pth;*.safetensors;*.ckpt;*.bin)", "All files (*.*)")
        res = self._window.create_file_dialog(OPEN_DIALOG, allow_multiple=False, file_types=types)
        if not res:
            return None
        p = res[0] if isinstance(res, (list, tuple)) else res
        return {"path": p, "name": os.path.basename(p)}

    def media_url(self, path):
        try:
            version = str(int(os.path.getmtime(path)))
        except OSError:
            version = "0"
        return f"http://127.0.0.1:{PORT}/media?path={urllib.parse.quote(path)}&v={version}"

    def default_output_dir(self):
        return os.path.join(REPO_ROOT, "complete")

    def load_settings(self):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_settings(self, settings):
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings or {}, f, indent=2)
        return {"ok": True}

    def bind_drop(self):
        try:
            target = self._window.dom.get_element("#binBody")
            if target:
                target.on("drop", DOMEventHandler(self._on_drop, prevent_default=True, stop_propagation=True))
        except Exception:
            pass

    def _on_drop(self, event):
        files = (event.get("dataTransfer") or {}).get("files") or []
        paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
        items = self._video_items(paths)
        if items:
            self._emit("addDroppedVideos", items)
        else:
            self._emit("onStatus", {"text": "Drop did not include usable video paths", "className": "error"})

    def start_batch(self, payload):
        if self._worker and self._worker.is_alive():
            return {"ok": False, "error": "A batch is already running"}
        self._pipeline = Pipeline(REPO_ROOT)
        self._worker = threading.Thread(target=self._run, args=(payload,), daemon=True)
        self._worker.start()
        return {"ok": True}

    def cancel(self):
        if self._pipeline:
            self._pipeline.cancel()
        return {"ok": True}

    def _emit(self, fn, data):
        try:
            self._window.evaluate_js(f"window.{fn} && window.{fn}({json.dumps(data)})")
        except Exception:
            pass

    def _run(self, payload):
        jobs = payload.get("jobs", [])
        opts = payload.get("options", {})
        for job in jobs:
            jid = job.get("id")
            inp = job.get("input")

            def log(line, _jid=jid):
                self._emit("onLog", {"jobId": _jid, "line": str(line)})

            if not inp:
                log("ERROR: missing input path")
                self._emit("onError", {"jobId": jid, "error": "Missing input path"})
                continue

            def progress(stage, pct, msg, _jid=jid):
                self._emit("onProgress", {"jobId": _jid, "stage": stage, "progress": pct, "message": msg})

            try:
                log(f"=== {os.path.basename(inp)} ===")
                out = self._pipeline.run_job({"id": jid, "input": inp}, opts, progress, log)
                log(f"DONE -> {out}")
                self._emit("onDone", {"jobId": jid, "output": out, "outputUrl": self.media_url(out)})
            except Cancelled:
                log("Cancelled.")
                self._emit("onError", {"jobId": jid, "error": "Cancelled", "cancelled": True})
                break
            except Exception as e:  # surface the real reason to the UI + log
                log(f"ERROR: {e}")
                self._emit("onError", {"jobId": jid, "error": str(e)})


def main():
    global PORT
    PORT = start_server()
    api = Api()
    window = webview.create_window(
        "Real-ESRGAN Upscaler",
        url=f"http://127.0.0.1:{PORT}/index.html",
        js_api=api,
        width=1280, height=820, min_size=(960, 640),
        background_color="#0a0a0a",
    )
    api._window = window
    window.events.loaded += api.bind_drop
    webview.start()


if __name__ == "__main__":
    main()
