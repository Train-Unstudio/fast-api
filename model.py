import requests
import numpy as np
import time
import logging 
import os
import shutil
import boto3
import datetime
import requests
import io
import cv2
import base64
from io import BytesIO
from PIL import Image, ImageChops, ImageFile
from io import BytesIO
#from cv2 import dnn_superres
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
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_mask_image3(init_img_sd):
    init_img_sd_array = np.array(init_img_sd)
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] > 0 )] = [255,255,255,255]
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] <= 0 )] = [0,0,0,255]
    out_img = Image.fromarray(init_img_sd_array)
#     blur = GaussianBlur(11,20)
#     out_img = blur(out_img)
    return out_img

def process_init(init_image):
    init_img_sd_array = np.array(init_image)
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] <= 128 )] = [128,128,128,255]
    out_img = Image.fromarray(init_img_sd_array)
#     blur = GaussianBlur(11,20)
#     out_img = blur(out_img)
    return out_img

def get_canny(image): 
  # get canny image
  image=np.array(image)
  canny_image = cv2.Canny(image, 100, 200)
  canny_image = canny_image[:, :, None]
  canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

  image=Image.fromarray(image)
  #mask_image=Image.fromarray(mask_image)
  canny_image = Image.fromarray(canny_image)
  #canny_image= canny_image.resize((512,512))
  # task 2. rename the mask with the corresponding file name
  canny_image.save('canny_image_std_serum.png') # rename the file corresponding to the file name
  return canny_image

def sd_controlnet_inpainting(init_image,prompt):
    image = Image.open(init_image)

    mask_image=get_mask_image3(image)
    canny_image=get_canny(image)
    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )
# speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to('cpu')
    generator = torch.manual_seed(-1)
    output= pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=image,
        control_image=canny_image,
        controlnet_conditioning_scale = 1.0,
        mask_image=mask_image
    ).images[0]
    return output

def generate(withoutbg_img, prompt, randID, user_name, model_name = "model_1"):
    
    os.mkdir(randID)

    start_time = time.time()
    (w_in,h_in) = withoutbg_img.size
    if w_in > h_in:
        new_size = (w_in, w_in)
    else:
        new_size = (h_in, h_in)

    withoutbg_img_new = withoutbg_img    


    withoutbg_img_sd = withoutbg_img_new
    mask_img_sd = get_mask_image3(withoutbg_img_sd)
    canny_image= get_canny(withoutbg_img_sd)

    mask_img_sd_path = randID + "/" + "mask_img_sd_" + randID + ".png"
    canny_image_sd_path = randID + "/" + "canny_sd_" + randID + ".png"
    withoutbg_img_sd_path = randID + "/" + "withoutbg_img_sd_" + randID + ".png"

    canny_image.save(canny_image_sd_path)
    mask_img_sd.save(mask_img_sd_path)
    withoutbg_img_sd.save(withoutbg_img_sd_path)
    if model_name == "model_1":
        sd_output = sd_controlnet_inpainting(withoutbg_img_sd,prompt)
        sd_output_2 = sd_output
        sd_output_2.save(randID + "/" + "output_" + randID + ".jpeg")
        with open(randID + "/" + "output_" + randID + ".jpeg", "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    return image_base64
