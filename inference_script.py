# app.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()


# model = pickle.load("model.pkl")

class Input(BaseModel):
    instances: list

@app.get("/health_check")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(input):
    # data = np.array(input.instances)
    preds = [1, 2, 3]
    return {"predictions": preds}

