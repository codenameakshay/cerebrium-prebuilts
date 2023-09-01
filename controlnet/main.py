import base64
import io
from typing import Optional
from pydantic import BaseModel
import torch
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from io import BytesIO


class Item(BaseModel):
    prompt: str
    image: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 40
    num_images_per_prompt: Optional[int] = 1
    seed: Optional[int] = 0
    preprocessor_name: Optional[str] = "HED"

class Item(BaseModel):
    prompt: str
    hf_token: Optional[str]
    num_inference_steps: Optional[int] = 20
    num_samples: Optional[float] = 2
    height: Optional[int] = 512
    width: Optional[int] = 512
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str]
    num_images_per_prompt: Optional[str] = 1
    image_resolution: Optional[int] = 512
    strength: Optional[float] = 1.0
    guess_mode: Optional[bool] = False
    low_threshold: Optional[int] = 100
    high_threshold: Optional[int] = 200
    ddim_steps: Optional[int] = 20
    scale: Optional[float] = 9.0
    seed: Optional[int] = 1
    eta: Optional[float] = 0.0
    model_id: Optional[str]
    model: str
    image: Optional[str]
    image_url = Optional[str]

def download_image(image_url):
    image = Image.open(BytesIO(base64.b64decode(image_url)))
    image = image.convert("RGB")
    return image


checkpoint = "lllyasviel/control_v11p_sd15_softedge"

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, device_map="auto"
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")



def predict(item, run_id, logger):
    params = Item(**item)
    
    image = Image.open(BytesIO(base64.b64decode(item.image)))
    image = image.convert("RGB")
    image_width, image_height = image.size

    if(item.preprocessor_name=="HED"):
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    elif(item.preprocessor_name=="PidiNet"):
        processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    else:
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    

    control_image = processor(image, safe=True)


    images = pipe(
        prompt=item.prompt,
        image=control_image,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        num_images_per_prompt=params.num_images_per_prompt,
        negative_prompt=params.negative_prompt,
        generator=torch.manual_seed(params.seed),
    ).images
    
    finished_images = []
    for image in images:
        #image = image.resize((image_width, image_height))
        buffered = io.BytesIO()
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    
    return finished_images