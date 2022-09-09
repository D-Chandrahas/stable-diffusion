from optimizedSD.optimized_txt2img import optimised_txt2img
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
import re

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render"
    )
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to (Default: outputs)", default="outputs")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps (Default: 50)"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often (Default: 1)"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space (Default: 512)"
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space (Default: 512)"
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels (Default: 4)"
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor (Default: 8)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1, ## changed from default value of 5
        help="how many samples to produce for each given prompt. A.k.a. batch size (Default: 1)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) (Default: 7.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="specify GPU (cuda:0/cuda:1/...) (Default: cuda:0)"
    )
    parser.add_argument(
        "--from_file", ## previously called --from-file
        type=str,
        help="if specified, load prompts from this file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="the seed (for reproducible sampling)"
    )
    parser.add_argument(
        "--unet_bs",
        type=int,
        default=1,
        help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended ) (Default: 1)"
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM"
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision (Default: autocast)",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="sampler (Default: plms)",
        choices=["ddim", "plms"],
        default="plms"
    )
    parser.add_argument(
        "--skip_log",
        action="store_true",
        help="Dosent log the command line arguments when this flag is used"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use full precision floats"
    )
    parser.add_argument(
        "--enhance_face",
        action="store_true",
        help="Enhance faces using GFPGAN"
    )
    parser.add_argument(
        "--enhance_image",
        action="store_true",
        help="Enhance image using RealESRGAN"
    )
    parser.add_argument(
        '--upscale',
        type=int,
        nargs="?",
        choices=[2,4],
        default=2,
        const=2,
        help='The final upsampling scale of the image. Default: 2'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = arguments()

    optimised_txt2img(args)

    if(args.enhance_image):

        if(args.upscale == 4):
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        else:
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=args.upscale
        )

        RealESRGAN = RealESRGANer(
            scale=args.upscale,
            model_path=model_path,
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=not args.fp32,
            gpu_id=args.device[5]
        )
    else:
        RealESRGAN = None

    if(args.enhance_face):
        
        GFPGAN = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
            upscale=args.upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=RealESRGAN
        )
    else:
        GFPGAN = None

    if(args.enhance_face):
        if not args.from_file:
            prompts = [args.prompt]
        else:
            with open(args.from_file, "r") as f:
                prompts = f.read().splitlines()

        for prompt in prompts:
            prompt_path = os.path.join(args.outdir, "_".join(re.split(":| ", prompt)))[:250]
            # for img in prompt_path:
            #     _, _, output = GFPGAN.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            #     save(output)

    elif(args.enhance_image):
        if not args.from_file:
            prompts = [args.prompt]
        else:
            with open(args.from_file, "r") as f:
                prompts = f.read().splitlines()

        for prompt in prompts:
            prompt_path = os.path.join(args.outdir, "_".join(re.split(":| ", prompt)))[:250]
            # for img in prompt_path:
            #     output, _ = RealESRGAN.enhance(img, outscale=args.upscale)
            #     save(output)
