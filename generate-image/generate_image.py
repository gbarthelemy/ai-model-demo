import sys

import torch
from diffusers import StableDiffusionPipeline

GEN_PREFIX = "gen_"
MODEL_ID = "runwayml/stable-diffusion-v1-5"


def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to("mps")  # pipe = pipe.to("cuda")

    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    images = pipe(prompt, num_inference_steps=30, num_images_per_prompt=4).images

    images[0].save(GEN_PREFIX + prompt.replace(" ", "_") + "_1.png")
    images[1].save(GEN_PREFIX + prompt.replace(" ", "_") + "_2.png")
    images[2].save(GEN_PREFIX + prompt.replace(" ", "_") + "_3.png")
    images[3].save(GEN_PREFIX + prompt.replace(" ", "_") + "_4.png")


if __name__ == '__main__':
    generate_image(sys.argv[1])
