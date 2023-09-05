import base64
import io
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from pydantic import BaseModel, HttpUrl
from typing import Optional


#######################################
# User-facing API Parameters
#######################################
class Item(BaseModel):
    hf_token: Optional[str] = None
    prompt: str
    hf_model_path: Optional[str] = "stabilityai/stable-diffusion-2-1"
    guidance_scale: float = 7.5
    height: int = 512
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    seed: Optional[int]
    width: int = 512
    webhook_endpoint: Optional[HttpUrl] = None


#######################################
# Model Setup
#######################################

set_auth_token = "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"
scheduler_1 = EulerDiscreteScheduler.from_pretrained(
    "SG161222/Realistic_Vision_V1.4_Fantasy.ai",
    subfolder="scheduler",
    use_auth_token=set_auth_token,
    cache_dir="/persistent-storage",
)
pipe_1 = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V1.4_Fantasy.ai",
    scheduler=scheduler_1,
    torch_dtype=torch.float16,
    use_auth_token=set_auth_token,
    cache_dir="/persistent-storage",
)

scheduler_2 = EulerDiscreteScheduler.from_pretrained(
    "GenZArt/jzli-DreamShaper-3.3-baked-vae",
    subfolder="scheduler",
    use_auth_token=set_auth_token,
    cache_dir="/persistent-storage",
)
pipe_2 = StableDiffusionPipeline.from_pretrained(
    "GenZArt/jzli-DreamShaper-3.3-baked-vae",
    scheduler=scheduler_2,
    torch_dtype=torch.float16,
    use_auth_token=set_auth_token,
    cache_dir="/persistent-storage",
)

scheduler_3 = EulerDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="scheduler",
    use_auth_token=set_auth_token,
    cache_dir="/persistent-storage",
)
pipe_3 = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    scheduler=scheduler_3,
    torch_dtype=torch.float16,
    use_auth_token=set_auth_token,
    cache_dir="/persistent-storage",
)


def run_model(pipe, params, logger):
    if params.seed is not None and int(params.seed) > 0:
        logger.info("Manual seed")
        generator = torch.Generator("cuda").manual_seed(int(params.seed))
    else:
        logger.info("Random seed")
        generator = torch.Generator("cuda")

    logger.info("Seed: {}".format(params.seed))

    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to("cuda")

    images = pipe(
        params.prompt,
        height=params.height,
        width=params.width,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        num_images_per_prompt=params.num_images_per_prompt,
        negative_prompt=params.negative_prompt,
        generator=generator,
    ).images

    return images


########################################
# Prediction
########################################
def predict(item, run_id, logger):
    params = Item(**item)
    hf_model_path = params.model_id if bool(params.model_id) else "stabilityai/stable-diffusion-2-1"
    if hf_model_path == "SG161222/Realistic_Vision_V1.4_Fantasy.ai":
        images = run_model(pipe=pipe_1, params=params, logger=logger)
    elif hf_model_path == "GenZArt/jzli-DreamShaper-3.3-baked-vae":
        images = run_model(pipe=pipe_2, params=params, logger=logger)
    elif hf_model_path == "stabilityai/stable-diffusion-2-1":
        images = run_model(pipe=pipe_3, params=params, logger=logger)
    else:
        auth_token = params.get("hf_token", False)
        scheduler = EulerDiscreteScheduler.from_pretrained(hf_model_path, subfolder="scheduler", use_auth_token=auth_token)
        pipe = StableDiffusionPipeline.from_pretrained(
            hf_model_path, scheduler=scheduler, torch_dtype=torch.float16, use_auth_token=auth_token
        )
        images = run_model(pipe=pipe, params=params, logger=logger)

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return finished_images
