"""
AdaBoost  (from your ADAboost.py)

Many very small decision trees ("stumps", depth 1). Each new stump
focuses on the bookings the previous stumps got wrong.
Task: predict booking cancellation.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import config
from data_loader import get_X_y, split_feature_types
from utils import banner

print("AdaBoost Model")

X, y = get_X_y()   # leakage columns already removed
numerical_features, categorical_features = split_feature_types(X)
print("\nNumber of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=config.RANDOM_STATE),
    n_estimators=100,
    learning_rate=1.0,
    random_state=config.RANDOM_STATE,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y,
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

banner("ADABOOST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Not cancelled", "Cancelled"]))
