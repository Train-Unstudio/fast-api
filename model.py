import requests
import numpy as np
import time
import logging 
import os
import shutil
import datetime
import requests
import io
import cv2
import base64
from io import BytesIO
from PIL import Image, ImageChops, ImageFile
from io import BytesIO
from pipeline_stable_diffusion_controlnet_inpaint import *
#from cv2 import dnn_superres
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
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
    out_img.save('mask_image_std_serum.png') # rename the file corresponding to the file name

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

def custom_model(b64string,prompt):
   img_bytes=base64.b64decode(b64string)
   img_stream=io.BytesIO(img_bytes)
   img=Image.open(img_stream)
   init_img=img
   canny_image=get_canny(init_img)
   mask_image=get_mask_image3(init_img)
   init_img.save("init_img.png")
   controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
   pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )
   pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
   pipe.enable_xformers_memory_efficient_attention()
   pipe.to('cuda')
   generator = torch.manual_seed(-1)
   output= pipe(
        prompt,
        num_inference_steps=20,
        generator=generator,
        image=init_img,
        control_image=canny_image,
        controlnet_conditioning_scale = 1.0,
        mask_image=mask_image
    ).images[0]
   output.save("output.png")
   with open ("output.png","rb") as image_file:
    encoded_string=base64.b64encode(image_file.read()).decode("utf-8")
   return encoded_string
