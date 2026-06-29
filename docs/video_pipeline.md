# Lossless video upscaling pipeline (ffmpeg + Real-ESRGAN, CUDA path)

## 1. Inspect the source (so the merge matches it)
ffprobe -v error -select_streams v:0 \
  -show_entries stream=color_range,color_space,color_primaries,color_transfer,r_frame_rate,pix_fmt \
  -of default=noprint_wrappers=1 input.mp4

## 2. Extract frames LOSSLESSLY to PNG
ffmpeg -i input.mp4 -vsync 0 -qscale:v 1 frames_in/frame%08d.png

## 3. Upscale (CUDA path; PNG in, PNG out -> no recompression)
uv run python inference_realesrgan.py -n realesr-general-x4v3 -i frames_in -o frames_out -s 2 --ext png
#   - resumable: re-run the same command to continue after an interruption
#   - exits non-zero and lists frames if any failed; fix before merging

## 4. Merge back, matching the source color tags and a high-quality codec
#    Replace <colorspace>/<primaries>/<transfer>/<range> with the ffprobe values from step 1.
ffmpeg -framerate <r_frame_rate> -i frames_out/frame%08d.png -i input.mp4 \
  -map 0:v:0 -map 1:a:0? -c:a copy \
  -c:v libx264 -crf 17 -pix_fmt yuv420p \
  -colorspace <colorspace> -color_primaries <primaries> -color_trc <transfer> -color_range <range> \
  -movflags +faststart output_upscaled.mp4

# Notes:
# - yuv420p is the broadly compatible choice; use yuv444p + -profile:v high444 only if your
#   players support it (some hardware decoders reject 4:4:4).
# - crf 17 is visually near-lossless; lower = larger/better. Do NOT re-extract from a
#   compressed intermediate.
# - keep frames_in/ on disk until the merge is verified (resumability + crash safety).
