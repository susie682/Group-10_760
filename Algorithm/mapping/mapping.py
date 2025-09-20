# -*- coding: utf-8 -*-
"""
Keogram -> Satellite Rayleigh Linear Regression with feature selection & transformation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from itertools import combinations

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
# 3. Candidate features and transformations
# --------------------------
base_features = ['keogram_mean', 'keogram_median', 'keogram_max']

# 支持 log / sqrt 转换
def transform_features(df, features, method=None):
    X = df[features].copy()
    if method == "log":
        X = np.log1p(X)
    elif method == "sqrt":
        X = np.sqrt(X)
    return X.values

# --------------------------
# 4. Log-transform target
# --------------------------
y_train_log = np.log1p(df_train['satellite_mean'].values)
y_val_log   = np.log1p(df_val['satellite_mean'].values)
y_test_log  = np.log1p(df_test['satellite_mean'].values)

# --------------------------
# 5. Feature selection & model fitting
# --------------------------
best_r2 = -np.inf
best_features = None
best_transform = None
best_model = None

transform_methods = [None, "log", "sqrt"]

for trans in transform_methods:
    # 遍历单特征 + 多特征组合
    for k in range(1, len(base_features)+1):
        for feats in combinations(base_features, k):
            X_train = transform_features(df_train, list(feats), method=trans)
            X_val   = transform_features(df_val, list(feats), method=trans)
            
            model = LinearRegression()
            model.fit(X_train, y_train_log)
            y_val_pred_log = model.predict(X_val)
            y_val_pred = np.expm1(y_val_pred_log)
            r2 = r2_score(df_val['satellite_mean'].values, y_val_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_features = feats
                best_transform = trans
                best_model = model

print(f"Best feature set: {best_features}, transform: {best_transform}, val R2: {best_r2:.4f}")

# --------------------------
# 6. Evaluate on all sets
# --------------------------
def evaluate(model, df_train, df_val, df_test, features, transform):
    X_train = transform_features(df_train, list(features), transform)
    X_val   = transform_features(df_val, list(features), transform)
    X_test  = transform_features(df_test, list(features), transform)
    
    y_train_pred = np.expm1(model.predict(X_train))
    y_val_pred   = np.expm1(model.predict(X_val))
    y_test_pred  = np.expm1(model.predict(X_test))
    
    def print_metrics(y_true, y_pred, name="Dataset"):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{name}: R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    print_metrics(df_train['satellite_mean'].values, y_train_pred, "Train")
    print_metrics(df_val['satellite_mean'].values, y_val_pred, "Validation")
    print_metrics(df_test['satellite_mean'].values, y_test_pred, "Test")
    
    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(df_train['satellite_mean'], y_train_pred, c='blue', alpha=0.6, label='Train')
    plt.scatter(df_val['satellite_mean'], y_val_pred, c='green', alpha=0.6, label='Validation')
    plt.scatter(df_test['satellite_mean'], y_test_pred, c='red', alpha=0.6, label='Test')
    plt.plot([0, max(df_train['satellite_mean'].max(), df_val['satellite_mean'].max(), df_test['satellite_mean'].max())],
             [0, max(df_train['satellite_mean'].max(), df_val['satellite_mean'].max(), df_test['satellite_mean'].max())],
             'k--', lw=2)
    plt.xlabel("True Satellite Mean (Rayleigh)")
    plt.ylabel("Predicted (Rayleigh)")
    plt.title("Keogram -> Satellite Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Residuals
    plt.figure(figsize=(6,4))
    residuals = df_test['satellite_mean'] - y_test_pred
    plt.scatter(df_test['satellite_mean'], residuals, c='red', alpha=0.6)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("True Satellite Mean (Rayleigh)")
    plt.ylabel("Residuals")
    plt.title("Test Residuals")
    plt.grid(True)
    plt.show()
    
    # Coefficients
    coef_dict = {feat: coef for feat, coef in zip(features, model.coef_)}
    print("Linear Regression coefficients (log-target):")
    print(coef_dict)
    print(f"Intercept: {model.intercept_}")

evaluate(best_model, df_train, df_val, df_test, best_features, best_transform)
