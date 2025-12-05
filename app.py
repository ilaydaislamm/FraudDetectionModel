import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import math

app = FastAPI()

templates = Jinja2Templates(directory="templates")

with open(r"C:\Users\omen\PycharmProjects\Fraud_Detection\.venv\fraud_detection_model.pkl" , "rb") as f:
    data = pickle.load(f)

    model = data["model"]
    encoders = data["encoders"]
    freq_city = data["freq_city"]
    freq_state = data["freq_state"]


class DetectionFeatures(BaseModel):
    Year:int
    Month:int
    Day:int
    Amount:float
    Use_Chip:str
    Zip:float
    MCC:int
    hours:int
    minute:int
    Merchant_City:str
    Merchant_State:str


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})


@app.post("/predict")
async def predict(features: DetectionFeatures):
    input_data = pd.DataFrame([features.model_dump()])

    input_data["Use_Chip"] = encoders["Use_Chip"].transform([input_data["Use_Chip"].iloc[0]])
    total_minutes = input_data["hours"][0] * 60 + input_data["minute"][0]

    input_data["sin_time"] = math.sin(2 * math.pi * total_minutes / 1440)
    input_data["cos_time"] = math.cos(2 * math.pi * total_minutes / 1440)

    merchant_city = input_data["Merchant_City"].map(freq_city).fillna(0)
    merchant_state = input_data["Merchant_State"].map(freq_state).fillna(0)

    input_data["Merchant_City_Freq"] = merchant_city.iloc[0]
    input_data["Merchant_State_Freq"] = merchant_state.iloc[0]

    input_data_new = input_data.drop(columns=["hours","minute","Merchant_City","Merchant_State"],axis = 1)

    
    df = pd.DataFrame(input_data_new)


    prediction = model.predict(df)

    return {"predicted_price": int(prediction[0])}
