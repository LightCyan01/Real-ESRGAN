# Real-ESRGAN Upscaler: native UI

A simple native desktop app for upscaling videos with Real-ESRGAN. It extracts
frames with ffmpeg, upscales them on your GPU via `inference_realesrgan.py`, and
merges them back. No Node, no build step, no Electron. It's a Python window
(pywebview → native Windows WebView2) over plain HTML/JS.

## Requirements

- The repo's Python env must already work (CUDA torch + `realesrgan` installed;
  see `plans/001-uv-cuda-revival.md`). The UI upscales using **the same Python
  interpreter you launch it with**, so launch it from that env.
- `ffmpeg` and `ffprobe` on your PATH.
- `pywebview` (Windows: pulls the EdgeChromium/WebView2 backend, present on Win11).

## Install + run

From the repo root, in the env that has `realesrgan`:

```bash
uv pip install pywebview
uv run python ui/app.py
```

(Or activate the venv and `python ui/app.py`.)

## Use

1. **Input** → choose one or more videos (they appear as boxes in the bin).
2. Click a box to preview it; after upscaling, use **Original / Upscaled / Compare**
   (drag the wipe handle in Compare).
3. **Output** → choose where the finished videos go.
4. Set options (model, scale, audio, cleanup, etc.) and press **Start**.

Frames are written to your system temp by default (configurable in options);
they're deleted after a successful merge unless you turn off "Clean up frames".
On failure or cancel they're kept, so re-running resumes instead of redoing work.

## Notes

- Adding videos is via the **Input** button (native file picker). Drag-and-drop is
  best-effort: the WebView2 sandbox usually doesn't expose dropped file paths, so a
  drop opens the picker instead.
- `realesr-general-x4v3` is the only model that uses **Denoise**.
- Choose **webm** only if you want VP9/Opus (slower); mp4/mkv/mov/avi use H.264.
