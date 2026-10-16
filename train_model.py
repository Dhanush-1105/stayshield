"""
Train the models used by the website and save them to outputs/.

Trains Logistic Regression, AdaBoost and Gradient Boosting on the same
split, saves their scores (model_results.json) and saves the Gradient
Boosting model (model.joblib) that the website uses for predictions.
The website runs this automatically the first time if the files are missing.
"""
import json

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from data_loader import get_X_y, split_feature_types

MODEL_PATH = config.OUTPUT_DIR / "model.joblib"
RESULTS_PATH = config.OUTPUT_DIR / "model_results.json"


def make_preprocessor(num, cat, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat),
    ])


def train_and_save():
    X, y = get_X_y()
    num, cat = split_feature_types(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(num, cat)),
            ("clf", LogisticRegression(max_iter=1000))]),
        "AdaBoost": Pipeline([
            ("prep", make_preprocessor(num, cat, scale=False)),
            ("clf", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1,
                                                 random_state=config.RANDOM_STATE),
                n_estimators=100, random_state=config.RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([
            ("prep", make_preprocessor(num, cat)),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=config.RANDOM_STATE))]),
    }

    results = {}
    for name, model in models.items():
        print("Training", name, "...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, pos_label=1,
                                                     average="binary")
        results[name] = {"accuracy": accuracy_score(y_test, pred),
                         "precision": p, "recall": r, "f1": f}

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # typical value for every input column (used to fill what the user skips)
    defaults = {c: float(X[c].median()) for c in num}
    defaults.update({c: X[c].mode().iloc[0] for c in cat})
    options = {c: sorted(X[c].dropna().unique().tolist()) for c in cat}

    joblib.dump({"model": models["Gradient Boosting"], "defaults": defaults,
                 "options": options, "columns": X.columns.tolist(),
                 "sklearn_version": sklearn.__version__}, MODEL_PATH)
    print("Saved", MODEL_PATH.name, "and", RESULTS_PATH.name)


if __name__ == "__main__":
    train_and_save()
