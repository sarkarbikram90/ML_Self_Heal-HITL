import mlflow.sklearn
import joblib

"""
Loads the last known good model as a self-healing action.
"""

def load_model_from_mlflow(run_id):
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)

def switch_to_fallback_model():
    return joblib.load("models/fallback_model.pkl")

