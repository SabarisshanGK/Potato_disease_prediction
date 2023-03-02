import numpy as np
from fastapi import FastAPI,File,UploadFile
import uvicorn
from PIL import  Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("../models/1")
Class_name = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hi da Sabarisshan"

def read_from_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file:UploadFile = File(...)
):
    image = read_from_image(await file.read())
    image_batch = np.expand_dims(image,0)

    predictions = MODEL.predict(image_batch)
    predicted_class = Class_name[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class_name":predicted_class,
        "confidence":float(confidence),
    }


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)