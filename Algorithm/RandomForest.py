# RandomForest.py
# -------------------------------------------------------------
# Baseline implementation of Random Forest for aurora intensity prediction
# Author: Susie + Group 10 (COMPSCI 760)
# -------------------------------------------------------------
# This script:
#   1. Loads the dataset (final.csv)
#   2. Selects features and defines the target column
#   3. Cleans missing values (drop NaN targets, impute X features)
#   4. Splits the dataset into train/validation/test by year
#   5. Trains a Random Forest Regressor
#   6. Evaluates model performance (MSE, MAE, R²)
#   7. Outputs top feature importances for interpretation
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -------------------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------------------
# Build the absolute path to final.csv, assuming directory structure:
#   Group-10_760/
#     ├─ datasets/final.csv
#     └─ Algorithm/RandomForest.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path of this script
CSV_PATH = os.path.join(BASE_DIR, "..", "datasets", "final.csv")

df = pd.read_csv(CSV_PATH, parse_dates=["time"])
print("Loaded:", CSV_PATH)
print("Shape before drop:", df.shape)
print("Time range:", df["time"].min(), "->", df["time"].max(), flush=True)

# -------------------------------------------------------------
# 2. Define target and features
# -------------------------------------------------------------
# Current target = satellite aurora intensity mean (temporary baseline)
# Later, replace TARGET_COL = "keogram_mean" when ground data is added.
TARGET_COL = "mean"

# Columns that should not be used as input features
# time   -> timestamp
# mean/median/max -> currently target columns from satellite (used later as features)
# origin -> hemisphere label (0 = North, 1 = South), may be useful later as input
drop_cols = ["time", "mean", "median", "max", "origin"]

# Candidate features (all other numeric columns)
features = [c for c in df.columns if c not in drop_cols]

# Drop rows where the target is NaN (Random Forest cannot train on NaN y)
before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
after = len(df)
print(f"Dropped rows with NaN target ({TARGET_COL}): {before - after}", flush=True)

# Extract features (X) and target (y)
X_all = df[features]
y_all = df[TARGET_COL].values

# -------------------------------------------------------------
# 3. Time-based train/val/test split
# -------------------------------------------------------------
# For dataset covering 2012–2015:
#   Train = 2012–2013
#   Validation = 2014
#   Test = 2015
train_idx = df[(df["time"] < "2014-01-01")].index
val_idx   = df[(df["time"] >= "2014-01-01") & (df["time"] < "2015-01-01")].index
test_idx  = df[(df["time"] >= "2015-01-01") & (df["time"] < "2016-01-01")].index

print("Split sizes:",
      "train =", len(train_idx),
      "val =", len(val_idx),
      "test =", len(test_idx), flush=True)

# -------------------------------------------------------------
# 4. Handle missing values for features (X)
# -------------------------------------------------------------
# Random Forest cannot handle NaN in features.
# Strategy: replace missing values with median from training set.
imputer = SimpleImputer(strategy="median")

# Fit imputer only on training set, then transform all splits
X_train = imputer.fit_transform(X_all.loc[train_idx])
X_val   = imputer.transform(X_all.loc[val_idx])
X_test  = imputer.transform(X_all.loc[test_idx])

y_train = y_all[df.index.get_indexer(train_idx)]
y_val   = y_all[df.index.get_indexer(val_idx)]
y_test  = y_all[df.index.get_indexer(test_idx)]

# Safety check: ensure no NaN left in targets
assert not np.isnan(y_train).any(), "y_train still has NaN!"
assert not np.isnan(y_val).any(),   "y_val still has NaN!"
assert not np.isnan(y_test).any(),  "y_test still has NaN!"

# -------------------------------------------------------------
# 5. Train Random Forest Regressor
# -------------------------------------------------------------
# Key hyperparameters:
#   n_estimators = number of trees
#   max_depth = maximum depth of each tree (None = unlimited)
#   n_jobs = number of parallel jobs (-1 = use all CPUs)
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# -------------------------------------------------------------
# 6. Evaluate performance on validation and test sets
# -------------------------------------------------------------
def eval_and_print(split_name, y_true, y_pred):
    """Compute and print MSE, MAE, and R² for a given dataset split."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{split_name} -> MSE: {mse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}", flush=True)

print("\n=== Evaluation ===", flush=True)
eval_and_print("VAL ", y_val, rf.predict(X_val))
eval_and_print("TEST", y_test, rf.predict(X_test))

# -------------------------------------------------------------
# 7. Feature importance analysis
# -------------------------------------------------------------
# Random Forest provides a measure of how important each feature is
# (based on total decrease in impurity).
importances = rf.feature_importances_

# Sort and select top 15 features
order = np.argsort(importances)[::-1][:15]
top_feats = [(features[i], float(importances[i])) for i in order]

print("\nTop-15 feature importances:")
for name, score in top_feats:
    print(f"{name:20s}  {score:.4f}")
