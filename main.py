
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Response
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from base64 import b64encode
import os
from model import *
from typing import Union
from io import BytesIO
import base64
import time
import shutil


#import imageio as iio

#app = FastAPI(title="Project Arch")

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


#@app.get("/")
def read_root():
    return {"Hello": "World"}


#@app.post("/design/sd", status_code=200)
def predict(withoutbg_image_base64,prompt):
#def predict(withoutbg_image_base64 : str = Form(...), prompt : str = Form(...), randID: str = Form(...), username: str = Form(...), model_name: str = Form(...)):
    #withoutbg_image_base64 = b'=' * (-len(withoutbg_image_base64) % 4)
    #withoutbg_image_encoded = withoutbg_image_base64.encode('utf-8')
    withoutbg_image_bytes = BytesIO(base64.b64decode(withoutbg_image_base64))
    #withoutbg_image = Image.open(withoutbg_image_bytes)
    image_base64 = generate(withoutbg_image_bytes, prompt, model_name = "model_1")
    #image_base64 = generate(withoutbg_image, prompt, model_name = model_name)

    return {'image_base64': image_base64}


def imgToBase64String(filename):
    img = Image.open(filename)
    im_file = BytesIO()
    img.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return im_b64

withoutbg_image_string = imgToBase64String("fastapi/serum_standard.png") 

predict(withoutbg_image_string,"serum on mountain")

