import argparse
import realesrgan._compat  # noqa: F401,E402  MUST precede the basicsr import below
import cv2
import glob
import os
import sys
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--gfpgan_version', type=str, default='1.3', choices=['1.3', '1.4'],
                        help='GFPGAN model version for --face_enhance (stills only; v1.4 often better on real faces)')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                        help='Autotune cuDNN conv algorithms (bit-identical; best for same-size frames)')
    parser.add_argument('--channels_last', action='store_true',
                        help='Use channels_last memory format (faster fp16; PSNR-gate before trusting)')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile the model (implies --channels_last and --cudnn_benchmark; '
                             'best for a folder of same-size frames; PSNR-gate before trusting)')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode; reduce-overhead enables CUDA graphs (best for fixed-size frames)')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument('--use_spandrel', action='store_true',
                        help='Load --model_path via spandrel (auto-detects arch for community models)')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    if args.compile:
        args.channels_last = True
        args.cudnn_benchmark = True

    # restorer
    if args.use_spandrel:
        assert args.model_path, '--use_spandrel requires --model_path'
        from spandrel import ModelLoader
        desc = ModelLoader().load_from_file(args.model_path)
        model = desc.model
        netscale = desc.scale
        upsampler = RealESRGANer(
            scale=netscale, model_path=args.model_path, model=model, model_loaded=True,
            tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, half=not args.fp32, gpu_id=args.gpu_id,
            cudnn_benchmark=args.cudnn_benchmark, channels_last=args.channels_last,
            use_compile=args.compile, compile_mode=args.compile_mode)
    else:
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            pre_pad=args.pre_pad,
            half=not args.fp32,
            gpu_id=args.gpu_id,
            cudnn_benchmark=args.cudnn_benchmark,
            channels_last=args.channels_last,
            use_compile=args.compile,
            compile_mode=args.compile_mode)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        gfpgan_url = f'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv{args.gfpgan_version}.pth'
        face_enhancer = GFPGANer(
            model_path=gfpgan_url,
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    if args.ext in ('jpg', 'jpeg', 'webp'):
        print(f'WARNING: --ext {args.ext} re-encodes lossily on top of the upscaled result. '
              'For video frames, extract to PNG and use --ext png to stay lossless.')
    elif args.ext == 'auto':
        print('NOTE: --ext auto keeps the input extension (JPEG in -> JPEG out, lossy). '
              'For video frames, extract to PNG and use --ext png to stay lossless.')

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    if (args.cudnn_benchmark or args.compile) and not args.face_enhance and len(paths) > 0:
        _w_img = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
        if _w_img is not None:
            print('Warming up (compile/autotune)...')
            upsampler.warmup(_w_img.shape[0], _w_img.shape[1])

    failures = []
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print('\tSkip (unreadable / not an image):', path)
            continue
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        # ---- compute the output path up front so we can skip already-done frames ----
        if args.ext == 'auto':
            out_ext = extension[1:]
        else:
            out_ext = args.ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            out_ext = 'png'
        if args.suffix == '':
            save_path = os.path.join(args.output, f'{imgname}.{out_ext}')
        else:
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{out_ext}')

        # ---- resumability: skip frames already produced ----
        if os.path.exists(save_path):
            print('\tSkip (already exists):', save_path)
            continue

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            failures.append((path, str(error)))
            continue

        # ---- atomic write so a half-written file is never mistaken for "done" ----
        root, ext = os.path.splitext(save_path)
        tmp_path = f'{root}.tmp{ext}'
        if not cv2.imwrite(tmp_path, output):
            raise RuntimeError(f'Failed to write image: {tmp_path}')
        os.replace(tmp_path, save_path)

    if failures:
        print(f'\n{len(failures)} image(s) FAILED and were not written:')
        for p, err in failures:
            print(f'  {p}: {err}')
        print('The output folder is INCOMPLETE — do not merge it into a video until these are resolved.')
        sys.exit(1)


if __name__ == '__main__':
    main()
