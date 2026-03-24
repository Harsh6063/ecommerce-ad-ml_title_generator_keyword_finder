import pandas as pd
import numpy as np
import os

# 👉 Detect CI environment
IS_CI = os.getenv("CI") == "true"

if not IS_CI:
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri("file:./mlruns")
    os.makedirs("mlruns", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
from pathlib import Path

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/processed/final_dataset.csv")

features = [
    "price","rating","review_count",
    "frequency","avg_rank","avg_rating","avg_reviews",
    "title_len","log_reviews"
]

target = "suggested_bid"

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, features)
])

models = {
    "rf": RandomForestRegressor(),
    "xgb": XGBRegressor(),
    "lr": LinearRegression()
}

best = None
best_score = -1

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    print(f"{name}: {r2:.3f}")

    # 👉 Only log in LOCAL (not CI)
    if not IS_CI:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(pipe, artifact_path="model")

    if r2 > best_score:
        best_score = r2
        best = pipe

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
joblib.dump(best, "models/best_model.pkl")

print("✅ Best model saved → models/best_model.pkl")