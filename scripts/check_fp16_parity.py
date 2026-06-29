import argparse
import cv2
import glob
import numpy as np
import os
import realesrgan._compat  # noqa: F401  shim before basicsr
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def build(model_name):
    if model_name == 'realesr-general-x4v3':
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'), 4, \
            'weights/realesr-general-x4v3.pth'
    if model_name == 'realesr-animevideov3':
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'), 4, \
            'weights/realesr-animevideov3.pth'
    # default to the x4plus RRDBNet
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4, \
        'weights/RealESRGAN_x4plus.pth'


def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', '--model_name', default='RealESRGAN_x4plus')
    p.add_argument('-i', '--input', default='inputs')
    p.add_argument('--model_path', default=None)
    p.add_argument('--threshold', type=float, default=50.0, help='min acceptable PSNR (dB) fast-vs-eager')
    args = p.parse_args()

    arch, netscale, default_path = build(args.model_name)
    model_path = args.model_path or default_path

    eager = RealESRGANer(scale=netscale, model_path=model_path, model=build(args.model_name)[0], half=True)
    fast = RealESRGANer(scale=netscale, model_path=model_path, model=arch, half=True,
                        cudnn_benchmark=True, channels_last=True, use_compile=True)

    paths = sorted(glob.glob(os.path.join(args.input, '*'))) if os.path.isdir(args.input) else [args.input]
    worst = float('inf')
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        out_e, _ = eager.enhance(img)
        out_f, _ = fast.enhance(img)
        d = psnr(out_e, out_f)
        worst = min(worst, d)
        print(f'{os.path.basename(path)}: PSNR(fast vs eager) = {d:.2f} dB')

    print(f'\nMIN PSNR = {worst:.2f} dB (threshold {args.threshold})')
    if worst < args.threshold:
        print('FAIL: fast path diverges from eager beyond fp16 noise. Do NOT use --compile/--channels_last for this model.')
        raise SystemExit(1)
    print('PASS: fast path is within fp16 noise of eager. Safe to use.')


if __name__ == '__main__':
    main()
