# main.py
from model import *
import requests
import numpy as np
import time
import logging 

from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Response
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
#import imageio as iio

app = FastAPI(title="Project Arch")

origins = [
    "http://localhost",
    "http://localhost:3000",
]
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=origins,
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/design/sd", status_code=200)

# this accepts the base64encoded string 
def gen_output(base64_string:str=Form(...),prompt:str=Form(...)):
   custom_model(base64_string,prompt)

#test.py 
#def img2base64():
#    with open("/content/serum_standard.png","rb") as img_file:
#      encoded_string=base64.b64encode(img_file.read()).decode("utf-8")
#    return encoded_string
#b64_string=img2base64()
#prompt="serum on hills"
#gen_output(b64_string,prompt)
