import os
import sys
import urllib
import urllib.parse

import requests
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

GEN_PREFIX = "gen_"
MODEL_ID = "timbrooks/instruct-pix2pix"

def download_image(url):
    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.scheme == "file":
        # Si l'URL utilise le schéma "file://", téléchargement d'un fichier en local.
        local_path = urllib.parse.unquote(parsed_url.path)

        if os.path.exists(local_path):
            image = Image.open(local_path)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        else:
            raise FileNotFoundError(f"Le fichier '{local_path}' n'a pas été trouvé.")
    elif parsed_url.scheme in ["http", "https"]:
        # Si l'URL utilise le schéma "http://" ou "https://", téléchargement depuis Internet.
        response = requests.get(url, stream=True)
        response.raise_for_status()

        image = Image.open(response.raw)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image
    else:
        raise ValueError(f"Le schéma de l'URL '{url}' n'est pas pris en charge.")


def do_stuff(url, prompt):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                                                  safety_checker=None)
    pipe.to("mps")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

    image = download_image(url)

    images = pipe(prompt, image=image, num_inference_steps=30, image_guidance_scale=1).images
    images[0].save(GEN_PREFIX + prompt.replace(" ", "_") + "_0.png")


if __name__ == '__main__':
    # url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    # prompt = "turn him into cyborg"
    do_stuff(sys.argv[1], sys.argv[2])
