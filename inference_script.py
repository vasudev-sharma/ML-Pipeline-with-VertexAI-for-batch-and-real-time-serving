# app.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

app = FastAPI()


# model = pickle.load("model.pkl")

class Input(BaseModel):
    instances: list

@app.get("/health_check")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(input: BaseModel):
    # data = np.array(input.instances)
    preds = [4.60042024e-03, 3.33172912e-01, 5.26700015e+00, 9.20000000e+01,0.00000000e+00, 0.00000000e+00, 2.80000000e+01, 4.00000000e+00,5.10000000e+01]
    predictions = None
    with open("/app/model.pkl", 'rb') as file:
        model = pickle.load(file)
        predictions = model(input)
    
    if predictions:
        return {"predictions": predictions}
    return {"predictions": preds}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)