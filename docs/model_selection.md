# Choosing a model & avoiding video flicker

## Which built-in model?
| Content | Model (`-n`) | Why |
|---|---|---|
| General photos | `RealESRGAN_x4plus` | strongest general RRDBNet |
| Anime / cartoon **video** | `realesr-animevideov3` | purpose-built for anime video; lowest flicker; fastest |
| Noisy / compressed real footage | `realesr-general-x4v3` `-dn <0..1>` | denoise knob (`--denoise_strength`); fewer GAN hallucinations |
| Anime stills | `RealESRGAN_x4plus_anime_6B` | small, anime-tuned |

`realesr-general-x4v3` blends a denoise sibling via `--denoise_strength`
(default 0.5). Set it explicitly per source.

## Video flicker (most common quality complaint)
Each frame is upscaled independently, so aggressive sharpening flickers frame to
frame. Mitigations (no retraining):
- Prefer `realesr-animevideov3` for animation.
- **Keep `--face_enhance` OFF for video** — per-frame GFPGAN re-estimates each
  face, so eyes/glasses/features jitter. Use face enhance for stills only.
- Optional, opt-in: ffmpeg `deflicker` on the merged output
  (`-vf deflicker=mode=median`). It equalizes luminance and can dim real flashes —
  check the result visually.

## Community models (advanced)
With `--use_spandrel --model_path <file>.pth` you can run community ESRGAN/RRDBNet
models (e.g. from OpenModelDB). Caveats:
- They produce a **different** look (often sharper) — not identical to the
  built-ins; validate per source, watch for over-sharpening/ringing/flicker on video.
- **Check the license.** Popular models like 4x-UltraSharp / 4x-AnimeSharp are
  CC-BY-NC-SA-4.0 (non-commercial). Do not use them in commercial work.
