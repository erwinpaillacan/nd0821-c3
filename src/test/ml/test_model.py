import numpy as np
from src.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier


def test_train_model():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference():
    X = np.array([[1, 2], [3, 4]])
    model = train_model(X, np.array([0, 1]))
    preds = inference(model, X)

    assert all(
        isinstance(p, (int, float)) or np.issubdtype(p, np.number) for p in preds
    )
