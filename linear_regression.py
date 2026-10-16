"""
Linear Regression  (from your Linear_Regression_Sklearn.py)

Task: predict the price of a booking (adr = average daily rate, EUR)
from booking details. Your original file stopped in the middle of the
residual histogram; it is completed here.
"""
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import config
from data_loader import load_data
from utils import banner, finish_plot

# 1. Load dataset
data = load_data()

# 2. Select features and target
features = config.LINEAR_FEATURES
X = data[features]
y = data[config.REG_TARGET]

# 3. Remove missing values
data = pd.concat([X, y], axis=1).dropna()
X = data[features]
y = data[config.REG_TARGET]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
)

# 5-6. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict test data
y_pred = model.predict(X_test)

# 8. Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

banner("MODEL EVALUATION")
print("Mean Squared Error (MSE):", round(mse, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))
print("Mean Absolute Error (MAE):", round(mae, 3))
print("R Square (R2):", round(r2, 4))

banner("MODEL COEFFICIENTS (effect on adr in EUR)")
print("Intercept:", round(model.intercept_, 3))
for name, coef in zip(features, model.coef_):
    print(f"{name:30s} {coef:+.4f}")

# 9. Predict a new booking
banner("PRICE PREDICTION FOR A NEW BOOKING")
if config.INTERACTIVE:
    values = [float(input(f"Enter {f}: ")) for f in features]
else:
    values = X_train.median().tolist()
    print("(non-interactive mode: using a typical booking)")

new_booking = pd.DataFrame([values], columns=features)
price = model.predict(new_booking)
print("Predicted adr (price per night):", round(price[0], 2), "EUR")

# 10. Residual histogram (actual - predicted)
residuals = y_test - y_pred
plt.figure(figsize=(7, 4))
plt.hist(residuals, bins=60, edgecolor="black")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residual (actual - predicted adr)")
plt.ylabel("Number of bookings")
plt.title("Linear Regression - Residual Histogram")
finish_plot("linear_regression_residuals")

# 11. Actual vs predicted
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.2, s=8)
lim = [0, max(y_test.max(), y_pred.max())]
plt.plot(lim, lim, color="red", linestyle="--")
plt.xlabel("Actual adr")
plt.ylabel("Predicted adr")
plt.title("Linear Regression - Actual vs Predicted")
finish_plot("linear_regression_actual_vs_predicted")
