# -*- coding: utf-8 -*-
"""
Keogram -> Satellite Rayleigh
Random Forest & XGBoost alternative for Step 1
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --------------------------
# 1. Load data
# --------------------------
data_file = "final_weather_filtered.csv"
df = pd.read_csv(data_file, parse_dates=['time'])

# --------------------------
# 2. Train / Validation / Test split
# --------------------------
df_train = df[(df['time'].dt.year >= 2012) & (df['time'].dt.year <= 2017)]
df_val   = df[df['time'].dt.year == 2018]
df_test  = df[(df['time'].dt.year >= 2019) & (df['time'].dt.year <= 2020)]

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# --------------------------
# 3. Features & target
# --------------------------
features = ['keogram_mean', 'keogram_median', 'keogram_max']
X_train = df_train[features].values
y_train = df_train['satellite_mean'].values

X_val = df_val[features].values
y_val = df_val['satellite_mean'].values

X_test = df_test[features].values
y_test = df_test['satellite_mean'].values

# --------------------------
# 4. Train Random Forest
# --------------------------
rf_model = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=42)
rf_model.fit(X_train, y_train)

# --------------------------
# 5. Train XGBoost
# --------------------------
xgb_model = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# --------------------------
# 6. Predictions
# --------------------------
y_train_pred_rf = rf_model.predict(X_train)
y_val_pred_rf   = rf_model.predict(X_val)
y_test_pred_rf  = rf_model.predict(X_test)

y_train_pred_xgb = xgb_model.predict(X_train)
y_val_pred_xgb   = xgb_model.predict(X_val)
y_test_pred_xgb  = xgb_model.predict(X_test)

# --------------------------
# 7. Evaluation function
# --------------------------
def print_metrics(y_true, y_pred, name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

print("----- Random Forest -----")
print_metrics(y_train, y_train_pred_rf, "Train")
print_metrics(y_val, y_val_pred_rf, "Validation")
print_metrics(y_test, y_test_pred_rf, "Test")

print("\n----- XGBoost -----")
print_metrics(y_train, y_train_pred_xgb, "Train")
print_metrics(y_val, y_val_pred_xgb, "Validation")
print_metrics(y_test, y_test_pred_xgb, "Test")

# --------------------------
# 8. Scatter plot: True vs Predicted
# --------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred_rf, c='blue', alpha=0.6, label='RF Test')
plt.scatter(y_test, y_test_pred_xgb, c='red', alpha=0.6, label='XGB Test')
plt.plot([0, max(y_test.max(), y_test_pred_rf.max(), y_test_pred_xgb.max())],
         [0, max(y_test.max(), y_test_pred_rf.max(), y_test_pred_xgb.max())],
         'k--', lw=2)
plt.xlabel("True Satellite Mean (Rayleigh)")
plt.ylabel("Predicted (Rayleigh)")
plt.title("Keogram -> Satellite (RF & XGBoost)")
plt.legend()
plt.grid(True)
plt.show()
