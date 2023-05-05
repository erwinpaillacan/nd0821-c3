# Put the code for your API here.
# Use relative imports

from src.ml.model import train_model
from src.ml.model import inference
from src.ml.data import process_data

import joblib
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import pandas as pd
import asyncio

app = FastAPI()


def load_model():
    model = joblib.load("data/model/rf.joblib")
    lb = joblib.load("data/model/lb.joblib")
    encoder = joblib.load("data/model/encoder.joblib")

    return model, lb, encoder


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome message"},
    )


@app.post("/model")
async def predict(data):
    model, lb, encoder = load_model()

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    df = pd.DataFrame(data.dict(by_alias=True), index=[0])
    X, *_ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    predictions = inference(model, X)
    prediction = lb.inverse_transform(predictions)[0]
    return prediction


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
