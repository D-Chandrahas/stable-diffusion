from pyclbr import Class
import gradio as gr
import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from cv2 import imwrite,cvtColor,COLOR_RGB2BGR
logging.set_verbosity_error()

def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="a painting of a virus monster playing guitar", help="the prompt to render"
    )
    parser.add_argument("--outdir", type=str, help="dir to write results to (Default: outputs)", default="outputs")
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
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)"
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
        default="cuda",
        help="specify GPU (cuda/cuda:0/cuda:1/...) (Default: cuda)"
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
        help="the seed (for reproducible sampling), (Default: Random)"
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
        choices=["ddim", "plms","heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"],
        default="plms"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model (Default: models/ldm/stable-diffusion-v1/model.ckpt)",
        default="models/ldm/stable-diffusion-v1/model.ckpt"
    )
    parser.add_argument(
        "--skip_log",
        action="store_true",
        help="Dosent log the command line arguments when this flag is used"
    )
    parser.add_argument(
        "--enhance_face",
        action="store_true",
        help="Repair and upscale faces using GFPGAN"
    )
    parser.add_argument(
        "--enhance_image",
        action="store_true",
        help="Upscale image using RealESRGAN"
    )
    parser.add_argument(
        '--upscale',
        type=int,
        choices=[2,4],
        default=2,
        help='The final upscaling factor of the image. (Default: 2)'
    )

    return parser.parse_args()

def make_folders(args):

    os.makedirs(args.outdir, exist_ok=True)

    if not args.from_file:
        prompts = [args.prompt]
    else:
        with open(args.from_file, "r") as f:
            prompts = f.read().splitlines()

    prompt_paths = []

    for prompt in prompts:
        folder_base_count = 0
        ## make sure different prompts are stored in different folders
        base_path = os.path.join(args.outdir, "_".join(re.split(":| ", prompt)))[:100] ## changed folder name size limit from 150 to 100
        prompt_path = base_path
        while(os.path.exists(prompt_path)):
            with open(os.path.join(prompt_path, "prompt.txt"),'r') as f:
                if(f.read() == prompt):
                    break
                else:
                    folder_base_count += 1
                    prompt_path = base_path + "_" + str(folder_base_count)
        ##

        os.makedirs(prompt_path, exist_ok=True)

        ## store the prompt in file
        with open(os.path.join(prompt_path, "prompt.txt"),'w') as f:
            f.write(prompt)
        ##
        prompt_paths.append(prompt_path)
    
    return prompt_paths

def save_img(image,prompt_path,seed):
    base_count = len(os.listdir(prompt_path))
    while os.path.exists(os.path.join(prompt_path, f"seed_{str(seed)}_{base_count:05}.png")):
        base_count += 1
    imwrite(os.path.join(prompt_path, f"seed_{str(seed)}_{base_count:05}.png"),image)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def optimised_txt2img(opt):
    # Credit: https://github.com/basujindal

    config = "optimizedSD/v1-inference.yaml"

    DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt"
    # This variable is no longer used,
    # to change default path of checkpoint,
    # change the keyword argument `default` of `parser.add_argument("--ckpt",...)` in `arguments()`

    
    seed_everything(opt.seed)

    # Logging
    if not opt.skip_log:
        logger(vars(opt), log_csv = "logs/txt2img_logs.csv")

    sd = load_model_from_config(f"{opt.ckpt}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = opt.unet_bs
    model.cdevice = opt.device
    model.turbo = opt.turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = opt.device

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if opt.device != "cpu" and opt.precision == "autocast":
        model.half()
        modelCS.half()


    batch_size = opt.n_samples
    prompts_no = 0
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        prompts_no = 1
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            prompts_no = len(data)
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))


    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = ""
    all_images = [[[0 for _ in range(opt.n_samples)] for _ in range(prompts_no)] for _ in range(opt.n_iter)]
    with torch.no_grad():
        k=0
        for _ in trange(opt.n_iter, desc="Sampling"):
            l=0
            for prompts in tqdm(data, desc="data"):


                with precision_scope("cuda"):
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        seed=opt.seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=None,
                        sampler = opt.sampler,
                    )

                    modelFS.to(opt.device)

                    # print(samples_ddim.shape)
                    m=0
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        x_sample = cvtColor(x_sample.astype(np.uint8),COLOR_RGB2BGR)
                        all_images[k][l][m] = x_sample
                        
                        #----------------------------------------#
                        # convert all images to format accepted by the GANs and save them in a 3d list(all_images)

                        # Image.fromarray(x_sample.astype(np.uint8)).save(
                        #     os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.png")
                        # )
                        seeds += str(opt.seed) + ","
                        opt.seed += 1

                        #--------------------------------------#
                        m += 1

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)
                l += 1
            k += 1
            

    

    
    return all_images


def main(args):
    
    tic = time.time()

    # args = arguments()

    if args.seed == None:
        args.seed = randint(0, 1000000)
    seed = args.seed

    prompt_paths = make_folders(args)

    all_images = optimised_txt2img(args)

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
            half=(args.device != "cpu" and args.precision == "autocast"),
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
        for iter_images in all_images:
            for prompt_path,prompt_images in zip(prompt_paths,iter_images):
                for image in prompt_images:
                    _, _, output = GFPGAN.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
                    save_img(output,prompt_path,seed)
                    seed += 1

    elif(args.enhance_image):
        for iter_images in all_images:
            for prompt_path,prompt_images in zip(prompt_paths,iter_images):
                for image in prompt_images:
                    output, _ = RealESRGAN.enhance(image, outscale=args.upscale)
                    save_img(output,prompt_path,seed)
                    seed += 1

    else:
        for iter_images in all_images:
            for prompt_path,prompt_images in zip(prompt_paths,iter_images):
                for image in prompt_images:
                    save_img(image,prompt_path,seed)
                    seed += 1

    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(f"\nSamples finished in {time_taken:.2f} minutes")


class Arguments():
    def __init__(
            self,
            prompt,
            ddim_steps,
            ddim_eta,
            n_iter,
            H,
            W,
            C,
            f,
            n_samples,
            scale,
            device,
            from_file,
            seed,
            unet_bs,
            turbo,
            precision,
            sampler,
            ckpt,
            skip_log,
            enhance_face,
            enhance_image,
            upscale
    ):
        self.prompt = prompt
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.n_iter = n_iter
        self.H = H
        self.W = W
        self.C = C
        self.f = f
        self.n_samples = n_samples
        self.scale = scale
        self.device = device
        self.from_file = from_file
        self.seed = seed
        self.unet_bs = unet_bs
        self.turbo = turbo
        self.precision = precision
        self.sampler = sampler
        self.ckpt = ckpt
        self.skip_log = skip_log
        self.enhance_face = enhance_face
        self.enhance_image = enhance_image
        self.upscale = upscale

def prepare_args_and_call_main(...):
    args = Arguments(...)
    main(args)


if __name__ == "__main__":
    demo = gr.Interface(
        fn=prepare_args_and_call_main,
        inputs=[...],
        outputs=[...]
    )
    demo.launch()