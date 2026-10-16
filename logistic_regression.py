"""
Logistic Regression with two scalers  (from your Logistic_regression_scalar.py)

Task: predict whether a booking will be CANCELLED (1) or not (0).
The same model is trained twice - once with StandardScaler and once with
MinMaxScaler - so you can compare the results.
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
from data_loader import load_data
from utils import banner, finish_plot


def load_xy():
    data = load_data()
    features = config.LOGISTIC_FEATURES
    data = data[features + [config.TARGET]].dropna()   # remove missing values
    return data[features], data[config.TARGET]


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_and_evaluate(X_train, y_train, X_test, y_test, scaler, scaler_name):
    banner(scaler_name)

    # Fit scaler on TRAINING data only, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    print("Training Accuracy:", round(accuracy_score(y_train, train_pred), 4))
    print("Testing Accuracy:", round(accuracy_score(y_test, test_pred), 4))

    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix:")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not cancelled", "Cancelled"]
    ).plot()
    plt.title("Logistic Regression - " + scaler_name)
    finish_plot("logistic_regression_" + scaler_name.lower())

    return model


def main():
    X, y = load_xy()
    print("Data loaded successfully!")
    print("Number of rows:", len(X))
    print("Number of features:", X.shape[1])

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nTraining data:", X_train.shape)
    print("Testing data:", X_test.shape)

    train_and_evaluate(X_train, y_train, X_test, y_test,
                       StandardScaler(), "StandardScaler")
    train_and_evaluate(X_train, y_train, X_test, y_test,
                       MinMaxScaler(), "MinMaxScaler")

    banner("Both models completed successfully!")


if __name__ == "__main__":
    main()
