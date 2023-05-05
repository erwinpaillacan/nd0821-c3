import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.ml.data import process_data
from src.ml.model import inference
import pathlib

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


def load_model():
    model = joblib.load(
        (pathlib.Path(__file__).parent.resolve() / "data" / "model" / "rf.joblib")
    )
    lb = joblib.load(
        (pathlib.Path(__file__).parent.resolve() / "data" / "model" / "lb.joblib")
    )
    encoder = joblib.load(
        (pathlib.Path(__file__).parent.resolve() / "data" / "model" / "encoder.joblib")
    )
    return model, lb, encoder


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 40,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 1000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }


class Prediction(BaseModel):
    prediction: str


@app.get("/")
async def root():
    return JSONResponse(content={"message": "Welcome message"})


@app.post("/model")
async def predict(data: CensusData):
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
    return Prediction(prediction=prediction)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
