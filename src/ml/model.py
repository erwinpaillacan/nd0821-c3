from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import pandas as pd
import os


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = RandomForestClassifier(max_depth=10, random_state=0)

    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def evaluate_model(data, cat_features, label, model, encoder, lb):
    # Create an empty dataframe to store the model's performance metrics
    performance_df = pd.DataFrame(
        columns=["feature", "category", "precision", "recall", "fbeta"]
    )

    # Loop over each categorical feature
    for feature in cat_features:
        feature_performance = []

        # Loop over each category within the feature
        for category in data[feature].unique():
            # Subset the data to only include the current category
            subset = data.loc[data[feature] == category]

            # Process the data using a function called "process_data"
            X_test, y_test, *_ = process_data(
                subset,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            # Make predictions using the model
            y_pred = model.predict(X_test)

            # Compute the precision, recall, and fbeta score for the predictions
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            # Append the performance metrics to the feature_performance list
            feature_performance.append(
                {
                    "feature": feature,
                    "category": category,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )

        # Convert the feature_performance list to a dataframe
        feature_performance_df = pd.DataFrame(feature_performance)

        # Concatenate the feature_performance_df with the performance_df
        performance_df = pd.concat([performance_df, feature_performance_df], axis=0)

    return performance_df
