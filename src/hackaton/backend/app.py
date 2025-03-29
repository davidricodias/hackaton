from fastapi import FastAPI, File, UploadFile
from typing import Union
from transformers import pipeline
from PIL import Image
import os

pipe = pipeline("image-classification", model="Organika/sdxl-detector")

app = FastAPI()

@app.get("/")
def hello_world():
    return {"response": "Hello World"}

@app.post("/process_img/")
async def process_img(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(content)
    results = pipe(image)
    return {"filename": results}
