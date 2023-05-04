from flask import Flask, request, jsonify
from fastapi import FastAPI, File, UploadFile
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from pipeline_stable_diffusion_controlnet_inpaint import *
from torch import autocast
import numpy as np
import io 
import base64
import cv2 
from PIL import Image
import torch

from pydantic import BaseModel

app = FastAPI()

class InpaintingRequest(BaseModel):
    prompt: str
    num_steps: int = 10
    image_file: UploadFile = File(...)

def inpaint_image(prompt: str, num_steps: int,image: Image) -> Image:
    image = Image.open(image)
    # Preprocess mask
    init_img_sd_array = np.array(image)
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] > 0 )] = [0,0,0,255]
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] <= 0 )] = [255,255,255,255]
    mask = Image.fromarray(init_img_sd_array)
    # preprocess canny image
    image=np.array(image)
    canny_image = cv2.Canny(image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    image=Image.fromarray(image)
    canny_image = Image.fromarray(canny_image)

     # Load ControlNet model and StableDiffusionInpaintPipeline
    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16)
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to('cuda')

    generator = torch.manual_seed(-1)
    output= pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=image,
        control_image=canny_image,
        controlnet_conditioning_scale = 1.0,
        mask_image=mask
    )
    out_img = output.images[0]

    img_byte_arr = io.BytesIO()
    out_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Return output image as response
    return jsonify({'image': base64.b64encode(img_byte_arr).decode('utf-8')})

@app.post("/inpaint")
def inpaint(request: InpaintingRequest):
    image = Image.open(request.image_file.file)

    # Inpaint image
    out_image = inpaint_image(request.prompt, request.num_steps, image)

    out_image.save("output.png")
    # Return output image as response
    return out_image