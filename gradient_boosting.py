"""
Gradient Boosting  (from your gradient_boosting.py)

Trees are built one after another; each tree corrects the mistakes of
the ones before it. Task: predict booking cancellation.
Also saves the feature importances and the predictions to outputs/.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from data_loader import get_X_y, split_feature_types
from utils import banner, finish_plot

X, y = get_X_y()   # leakage columns already removed
numerical_features, categorical_features = split_feature_types(X)

numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric, numerical_features),
    ("cat", categorical, categorical_features),
])

gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=config.RANDOM_STATE,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", gradient_boosting),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y,
)

print("Training Gradient Boosting (this takes about a minute)...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

gb = model.named_steps["classifier"]

banner("GRADIENT BOOSTING RESULTS")
print("Total Boosting Trees:", len(gb.estimators_))

print("\nGradient Boosting Parameters:")
print("Number of Estimators:", gb.n_estimators)
print("Learning Rate:", gb.learning_rate)
print("Maximum Tree Depth:", gb.max_depth)
print("Random State:", gb.random_state)

print("\nFirst 10 boosting trees:")
for i, stage in enumerate(gb.estimators_[:10], 1):
    tree = stage[0]
    print(f"Tree {i}: Depth={tree.get_depth()}, Leaves={tree.get_n_leaves()}")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 4))
print("Accuracy Percentage:", f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Not cancelled", "Cancelled"]))

# ---- feature importance ------------------------------------------------
banner("FEATURE IMPORTANCE")
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": gb.feature_importances_,
}).sort_values(by="Importance", ascending=False)

print("Top 20 Important Features:")
print(importance_df.head(20).to_string(index=False))

importance_path = config.OUTPUT_DIR / "gradient_boosting_feature_importance.csv"
importance_df.to_csv(importance_path, index=False)

top = importance_df.head(15).iloc[::-1]
plt.figure(figsize=(8, 5))
plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Importance")
plt.title("Gradient Boosting - Top 15 Features")
finish_plot("gradient_boosting_feature_importance")

# ---- actual vs predicted ----------------------------------------------
comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
banner("ACTUAL VS PREDICTED (first 20)")
print(comparison.head(20).to_string(index=False))

predictions_path = config.OUTPUT_DIR / "gradient_boosting_predictions.csv"
comparison.to_csv(predictions_path, index=False)

print("\nSaved:", importance_path.name, "and", predictions_path.name)
banner("Gradient Boosting implementation completed!")
