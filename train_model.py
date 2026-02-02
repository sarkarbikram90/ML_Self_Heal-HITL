import mlflow
import mlflow.sklearn
import joblib
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from generate_data import generate_customer_data
from preprocess import preprocess

# Generate training data
data = generate_customer_data(n=500)
X = preprocess(data)

# Fake labels (for demo only)
y = (X["monthly_charges"] > 80).astype(int)

# Train model with MLflow
mlflow.set_experiment("ml-self-heal-hitl")

with mlflow.start_run() as run:
    model = LogisticRegression()
    model.fit(X, y)

    accuracy = model.score(X, y)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id
    print(f"MLflow run_id: {run_id}")

    # Save fallback model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fallback_model.pkl")
