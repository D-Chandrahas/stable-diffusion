### This fork provides integration with [GFPGAN](https://github.com/TencentARC/GFPGAN) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for facial restoration and image upscaling


# Optimized Stable Diffusion


This repo is a modified version of the Stable Diffusion repo, optimized to use less VRAM than the original by sacrificing inference speed.

To reduce the VRAM usage, the following opimizations are used:

- the stable diffusion model is fragmented into four parts which are sent to the GPU only when needed. After the calculation is done, they are moved back to the CPU.
- The attention calculation is done in parts.

# Installation

- Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html). Skip this step if you already have `conda` installed.
- Clone this repo. Ex: using `git clone https://github.com/D-Chandrahas/stable-diffusion.git`.
- Download the weights (`sd-v1-4.ckpt`) from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original). (Might have to create an account and accept the T&C).
- Move the `sd-v1-4.ckpt` file to the `stable-diffusion/models/ldm/stable-diffusion-v1/` folder and rename it to `model.ckpt`.
- Open the terminal and navigate into the `stable-diffusion/` folder.
- Create a conda environment using `conda env create -f environment.yaml`.
- Activate the conda environment with `conda activate ldm`.
 

# Usage

## img2img

- `img2img` can generate _512x512 images from a prior image and prompt using under 2.4GB VRAM in under 20 seconds per image_ on an RTX 2060.

- The maximum size that can fit on 6GB GPU (RTX 2060) is around 1152x1088.

- For example, the following command will generate 10 512x512 images:

`python optimizedSD/optimized_img2img.py --prompt "Austrian alps" --init-img ~/sketch-mountains-input.jpg --strength 0.8 --n_iter 2 --n_samples 5 --H 512 --W 512`

## txt2img

- `txt2img` can generate _512x512 images from a prompt using under 2.4GB GPU VRAM in under 24 seconds per image_ on an RTX 2060.

- For example, the following command will generate 10 512x512 images:

`python optimizedSD/optimized_txt2img.py --prompt "Cyberpunk style image of a Tesla car reflection in rain" --H 512 --W 512 --seed 27 --n_iter 2 --n_samples 5 --ddim_steps 50`

## inpainting

- `inpaint_gradio.py` can fill masked parts of an image based on a given prompt. It can inpaint 512x512 images while using under 2.5GB of VRAM.

- To launch the gradio interface for inpainting, run `python optimizedSD/inpaint_gradio.py`. The mask for the image can be drawn on the selected image using the brush tool.

- The results are not yet perfect but can be improved by using a combination of prompt weighting, prompt engineering and testing out multiple values of the `--strength` argument.


# Using the Gradio GUI

- You can also use the built-in gradio interface for `img2img`, `txt2img` & `inpainting` instead of the command line interface.

- Activate the conda environment and
  - Run img2img using `python optimizedSD/img2img_gradio.py`,
  - txt2img using `python optimizedSD/txt2img_gradio.py` and
  - inpainting using `python optimizedSD/inpaint_gradio.py`.

- img2img_gradio.py has a feature to crop input images. Look for the pen symbol in the image box after selecting the image.

# Arguments

## `--seed`

**Seed for image generation**, can be used to reproduce previously generated images. Defaults to a random seed if unspecified.

- The code will give the seed number along with each generated image. To generate the same image again, just specify the seed using `--seed` argument. Images are saved with its seed number as its name by default.

- For example if the seed number for an image is `1234` and it's the 55th image in the folder, the image name will be named `seed_1234_00055.png`.

## `--n_samples`

**Batch size/amount of images to generate at once.**

- To get the lowest inference time per image, use the maximum batch size `--n_samples` that can fit on the GPU. Inference time per image will reduce on increasing the batch size, but the required VRAM will increase.

- If you get a CUDA out of memory error, try reducing the batch size `--n_samples`. If it doesn't work, the other option is to reduce the image width `--W` or height `--H` or both.

## `--n_iter`

**Run _x_ amount of times**

- Equivalent to running the script n_iter number of times. Only difference is that the model is loaded only once per n_iter iterations. Unlike `n_samples`, reducing it doesn't have an effect on VRAM required or inference time.

## `--H` & `--W`

**Height & width of the generated image.**

- Both height and width should be a **multiple of 64**.

## `--turbo`

**Increases inference speed at the cost of extra VRAM usage.**

- Using this argument increases the inference speed by using around 700MB of extra GPU VRAM. It is especially effective when generating a small batch of images (~ 1 to 4) images. It takes under 20 seconds for txt2img and 15 seconds for img2img (on an RTX 2060, excluding the time to load the model). Use it on larger batch sizes if GPU VRAM available.

## `--precision autocast` or `--precision full`

**Whether to use `full` or `mixed` precision**

- Mixed Precision is enabled by default. If you don't have a GPU with tensor cores (any GTX 10 series card), you may not be able use mixed precision. Use the `--precision full` argument to disable it.

## `--unet_bs`

**Batch size for the unet model**

- Takes up a lot of extra RAM for **very little improvement** in inference time. `unet_bs` > 1 is not recommended!

- Should generally be a multiple of 2x(n_samples)

### `--scale` : unconditional guidance scale

### `--ddim_steps` : number of ddim sampling steps

### `--ddim_eta` : ddim eta (eta=0.0 corresponds to deterministic sampling)

### `--device` : specify GPU (cuda/cuda:0/cuda:1/...) or cpu

### `--outdir` : folder to save the output images to

## arguments specific to txt2img

### `--enhance_face` : use [GFPGAN](https://github.com/TencentARC/GFPGAN) to restore and upscale faces.

### `--enhance_image` : use [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to upscale image. Can be used in combination with `--enhance_face`.

### `--upscale` : The final upsampling scale of the image. Use with `--enhance_image` and/or `--enhance_face`. Value can be *2* or *4*.

## Weighted Prompts

- Prompts can also be weighted to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.

- The number followed by the colon represents the weight given to the words before the colon. The weights can be both fractions or integers.

## Troubleshooting

### Green colored output images

- If you have a Nvidia GTX series GPU, the output images maybe entirely green in color. This is because GTX series do not support half precision calculation, which is the default mode of calculation in this repository. To overcome the issue, use the `--precision full` argument. The downside is that it will lead to higher GPU VRAM usage.

###

## Changelog

- v1.0: Added support for multiple samplers for txt2img. Based on [crowsonkb](https://github.com/crowsonkb/k-diffusion)
- v0.9: Added support for calculating attention in parts. (Thanks to @neonsecret @Doggettx, @ryudrigo)
- v0.8: Added gradio interface for inpainting.
- v0.7: Added support for logging, jpg file format
- v0.6: Added support for using weighted prompts. (based on @lstein's [repo](https://github.com/lstein/stable-diffusion))
- v0.5: Added support for using gradio interface.
- v0.4: Added support for specifying image seed.
- v0.3: Added support for using mixed precision.
- v0.2: Added support for generating images in batches.
- v0.1: Split the model into multiple parts to run it on lower VRAM.
