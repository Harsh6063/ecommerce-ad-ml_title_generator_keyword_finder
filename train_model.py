import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
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

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# preprocessing
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

mlflow.set_experiment("ads_model")

best = None
best_score = -1

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(pipe, name)

        print(name, r2)

        if r2 > best_score:
            best_score = r2
            best = pipe

Path("models").mkdir(exist_ok=True)
joblib.dump(best, "models/best_model.pkl")

print("Saved best model")