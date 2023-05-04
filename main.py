
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
def get_mask_image3(init_img_sd):
    init_img_sd_array = np.array(init_img_sd)
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] > 0 )] = [255,255,255,255]
    init_img_sd_array[np.where(init_img_sd_array[:, :, 3] <= 0 )] = [0,0,0,255]
    out_img = Image.fromarray(init_img_sd_array)
#     blur = GaussianBlur(11,20)
#     out_img = blur(out_img)
    return out_img


app = FastAPI(title="Project Arch")

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/design/sd", status_code=200)

def predict(withoutbg_image_base64 : str = Form(...), prompt : str = Form(...), randID: str = Form(...), username: str = Form(...), model_name: str = Form(...)):
    #withoutbg_image_base64 = b'=' * (-len(withoutbg_image_base64) % 4)
    #withoutbg_image_encoded = withoutbg_image_base64.encode('utf-8')
    withoutbg_image_bytes = BytesIO(base64.b64decode(withoutbg_image_base64))
    withoutbg_image = Image.open(withoutbg_image_bytes)
    print(model_name)
    if model_name == "model_3":
        try:
            print("Guys")
        except:
            shutil.rmtree(randID)
            image_base64 = generate(withoutbg_image, prompt,randID, username, model_name = "model_1")
    else:
        image_base64 = generate(withoutbg_image, prompt,randID, username, model_name = model_name)

    return {'image_base64': image_base64}
