# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import joblib

# Add the necessary imports for the starter code.
# Add code to load in the data.
import pandas as pd

from .data import process_data
from .model import train_model, compute_model_metrics, inference, evaluate_model
import pathlib


data = pd.read_csv(
    pathlib.Path(__file__).parent.resolve() / ".." / ".." / "data" / "census.csv"
)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
label = "salary"
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)


joblib.dump(
    model,
    (
        pathlib.Path(__file__).parent.resolve()
        / ".."
        / ".."
        / "data"
        / "model"
        / "rf.joblib"
    ),
)

joblib.dump(
    lb,
    (
        pathlib.Path(__file__).parent.resolve()
        / ".."
        / ".."
        / "data"
        / "model"
        / "lb.joblib"
    ),
)

joblib.dump(
    encoder,
    (
        pathlib.Path(__file__).parent.resolve()
        / ".."
        / ".."
        / "data"
        / "model"
        / "encoder.joblib"
    ),
)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")


performance_df = evaluate_model(data, cat_features, label, model, encoder, lb)
print(performance_df.head())


performance_df.to_csv(
    (pathlib.Path(__file__).parent.resolve() / ".." / ".." / "slice_output.txt"),
    index=False,
)
