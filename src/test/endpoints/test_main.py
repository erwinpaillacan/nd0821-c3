import pytest
from fastapi.testclient import TestClient

from src.endpoints.main import app, CensusData


@pytest.fixture()
def client():
    return TestClient(app)


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome message"}


# TEST CASES BINARY
@pytest.fixture()
def below_50k_example():
    return CensusData(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )


def test_predict_below_50k(client: TestClient, below_50k_example: CensusData):
    response = client.post("/model", json=below_50k_example.dict())
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


@pytest.fixture()
def above_50k_example():
    return CensusData(
        age=50,
        workclass="Self-emp-not-inc",
        fnlgt=83311,
        education="Bachelors",
        education_num=13,
        marital_status="Married-civ-spouse",
        occupation="Exec-managerial",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=60,
        native_country="United-States",
    )


def test_predict_above_50k(client: TestClient, above_50k_example: CensusData):
    response = client.post("/model", json=above_50k_example.dict())
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
