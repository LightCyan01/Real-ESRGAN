"""Video upscale pipeline: ffmpeg extract -> Real-ESRGAN upscale -> ffmpeg merge.

All subprocesses run hidden (no console window on Windows). Importable; the GUI
drives it on a worker thread with progress + log callbacks and a cancel flag.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# CREATE_NO_WINDOW: never flash a console window for ffmpeg/python child processes.
_NOWINDOW = 0x08000000 if os.name == "nt" else 0

# ffmpeg -progress spews key=value lines; keep those out of the human log.
_PROGRESS_NOISE = re.compile(
    r"^(frame|fps|stream_\S+|bitrate|total_size|out_time\S*|dup_frames|drop_frames|speed|progress)=")


class Cancelled(Exception):
    pass


class Pipeline:
    def __init__(self, repo_root, python_exe=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
        self.repo_root = repo_root
        # Use the SAME interpreter running the GUI -> it already has realesrgan installed.
        self.python_exe = python_exe or sys.executable
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self._proc = None
        self._cancel = False
        self._log = lambda *_: None

    def cancel(self):
        self._cancel = True
        p = self._proc
        if p and p.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)],
                                   creationflags=_NOWINDOW, capture_output=True)
                else:
                    p.terminate()
            except Exception:
                pass

    # ── helpers ──────────────────────────────────────────────────────────────
    def probe(self, path):
        cmd = [self.ffprobe, "-v", "error",
               "-show_entries",
               "stream=codec_type,width,height,r_frame_rate,nb_frames,"
               "color_space,color_primaries,color_transfer,color_range",
               "-show_entries", "format=duration", "-of", "json", path]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NOWINDOW)
        except FileNotFoundError:
            raise RuntimeError(f"'{self.ffprobe}' not found — install ffmpeg and add it to PATH "
                               f"(or set its path in the code).")
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        fr = v.get("r_frame_rate", "30/1")
        try:
            num, den = fr.split("/")
            fps = (float(num) / float(den)) if float(den) else float(num)
        except Exception:
            fps = 30.0
        nb = v.get("nb_frames")
        if nb and str(nb).isdigit() and int(nb) > 0:
            nb_frames = int(nb)
        else:
            dur = float(data.get("format", {}).get("duration", 0) or 0)
            nb_frames = int(dur * fps) if dur else 0
        return {
            "width": v.get("width"), "height": v.get("height"),
            "fps": fps, "nb_frames": nb_frames, "has_audio": has_audio,
            "color_space": v.get("color_space"), "color_primaries": v.get("color_primaries"),
            "color_transfer": v.get("color_transfer"), "color_range": v.get("color_range"),
        }

    def _stream(self, cmd, on_line):
        if self._cancel:
            raise Cancelled()
        self._log("$ " + subprocess.list2cmdline(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, cwd=self.repo_root, creationflags=_NOWINDOW)
        except FileNotFoundError:
            self._proc = None
            raise RuntimeError(f"'{cmd[0]}' not found — is it installed / on PATH?")
        for line in self._proc.stdout:
            if self._cancel:
                break
            s = line.rstrip("\n")
            if on_line:
                on_line(line)
            if s and not _PROGRESS_NOISE.match(s):
                self._log(s)
        self._proc.wait()
        rc = self._proc.returncode
        self._proc = None
        if self._cancel:
            raise Cancelled()
        return rc

    # ── the job ──────────────────────────────────────────────────────────────
    def run_job(self, job, opts, progress, log=None):
        """job: {id, input}. opts: dict. progress(stage,pct,msg). log(str)."""
        self._log = log or (lambda *_: None)
        inp = job["input"]
        base = os.path.splitext(os.path.basename(inp))[0]
        temp_root = opts.get("temp_frames_dir") or os.path.join(tempfile.gettempdir(), "realesrgan_ui")
        out_root = opts.get("out_frames_dir") or os.path.join(tempfile.gettempdir(), "realesrgan_ui")
        in_frames = os.path.join(temp_root, base, "in")
        out_frames = os.path.join(out_root, base, "out")
        shutil.rmtree(in_frames, ignore_errors=True)
        shutil.rmtree(out_frames, ignore_errors=True)
        os.makedirs(in_frames, exist_ok=True)
        os.makedirs(out_frames, exist_ok=True)

        info = self.probe(inp)
        nb = info["nb_frames"] or 0
        self._log(f"[probe] {info['width']}x{info['height']} @ {info['fps']:.3f}fps, "
                  f"{nb} frames, audio={info['has_audio']}")
        progress("probe", 100, f"{info['width']}x{info['height']} @ {info['fps']:.2f}fps, {nb} frames")

        frame_re = re.compile(r"frame=\s*(\d+)")

        # 1) extract (lossless PNG). -y so a resumed run doesn't hang on overwrite prompt.
        progress("extract", 0, "Extracting frames")
        self._log(f"[extract] {inp} -> {in_frames}")

        def on_ext(line):
            m = frame_re.search(line)
            if m and nb:
                progress("extract", min(99, int(int(m.group(1)) / nb * 100)), None)

        ext_args = [self.ffmpeg, "-y", "-i", inp]
        if opts.get("denoise_temporal"):
            # temporal-denoise the LR BEFORE upscaling so the model doesn't amplify noise into shimmer
            ext_args += ["-vf", "hqdn3d=4:3:6:4.5"]
        ext_args += ["-vsync", "0", "-qscale:v", "1", "-nostats", "-progress", "pipe:1",
                     os.path.join(in_frames, "frame%08d.png")]
        self._stream(ext_args, on_ext)
        progress("extract", 100, None)

        # 2) upscale the extracted frame folder with this project's PyTorch CLI.
        progress("upscale", 0, "Upscaling")
        self._log(f"[upscale] model={opts['model']} scale={opts['scale']} -> {out_frames}")
        out_ext = "png"
        args = [self.python_exe, "-u", "inference_realesrgan.py",
                "-n", opts["model"], "-i", in_frames, "-o", out_frames,
                "-s", str(opts["scale"]), "--ext", out_ext, "--suffix", ""]
        if int(opts.get("tile") or 0) > 0:
            args += ["--tile", str(opts["tile"])]
        if opts["model"] == "realesr-general-x4v3":
            args += ["--denoise_strength", str(opts.get("denoise", 0.5))]
        if opts.get("face_enhance"):
            args += ["--face_enhance"]
        if opts.get("fp32"):
            args += ["--fp32"]
        if int(opts.get("gpu_id") or 0) != 0:
            args += ["-g", str(opts["gpu_id"])]
        # custom community model via spandrel (auto-detects arch); overrides -n's arch
        if opts.get("model_path"):
            args += ["--use_spandrel", "--model_path", opts["model_path"]]
        # fast mode: torch.compile + channels_last + cudnn.benchmark; reduce-overhead =
        # CUDA graphs, the win for a folder of same-size frames.
        if opts.get("fast"):
            args += ["--compile", "--compile_mode", "reduce-overhead"]

        test_re = re.compile(r"Testing\s+(\d+)")

        def on_up(line):
            m = test_re.search(line)
            if m and nb:
                progress("upscale", min(99, int((int(m.group(1)) + 1) / nb * 100)), None)

        rc = self._stream(args, on_up)
        if rc != 0:
            raise RuntimeError(f"Upscale exited with code {rc} — see the log above for the reason "
                               f"(model name? realesrgan installed in this interpreter? GPU memory?).")
        progress("upscale", 100, None)

        # 3) merge (codec matched to container; audio stream-copy unless webm).
        progress("merge", 0, "Merging")
        out_dir = opts.get("output_dir") or os.path.join(self.repo_root, "complete")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.repo_root, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        fmt = opts.get("output_format", "mp4")
        suffix = opts.get("suffix", "upscaled")
        name = f"{base}_{suffix}.{fmt}" if suffix else f"{base}.{fmt}"
        out_file = os.path.join(out_dir, name)
        is_webm = fmt == "webm"
        self._log(f"[merge] -> {out_file}")

        m = [self.ffmpeg, "-y", "-framerate", str(info["fps"]),
             "-i", os.path.join(out_frames, f"frame%08d.{out_ext}")]
        if opts.get("include_audio") and info["has_audio"]:
            m += ["-i", inp, "-map", "0:v:0", "-map", "1:a:0?",
                  "-c:a", "libopus" if is_webm else "copy"]
        if is_webm:
            m += ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"]
        else:
            m += ["-c:v", "libx264", "-crf", "17"]
        if opts.get("deflicker"):
            # post-merge temporal luminance deflicker to clean residual frame-to-frame pulsing
            m += ["-vf", "deflicker=size=5"]
        m += ["-pix_fmt", "yuv420p"]
        for key, flag in (("color_space", "-colorspace"), ("color_primaries", "-color_primaries"),
                          ("color_transfer", "-color_trc"), ("color_range", "-color_range")):
            if info.get(key):
                m += [flag, str(info[key])]
        m += ["-nostats", "-progress", "pipe:1", out_file]

        def on_mrg(line):
            mm = frame_re.search(line)
            if mm and nb:
                progress("merge", min(99, int(int(mm.group(1)) / nb * 100)), None)

        self._stream(m, on_mrg)
        progress("merge", 100, None)

        # 4) cleanup (only on success; on failure/cancel the frames stay -> resumable)
        if opts.get("cleanup", True):
            progress("cleanup", 0, "Cleaning up")
            self._log("[cleanup] removing frame dirs")
            shutil.rmtree(os.path.join(temp_root, base), ignore_errors=True)
            shutil.rmtree(os.path.join(out_root, base), ignore_errors=True)
            progress("cleanup", 100, None)

        return out_file
