import joblib
from model.model import SimpleChurnModel
import os

os.makedirs("models", exist_ok=True)

model = SimpleChurnModel()
joblib.dump(model, "models/fallback_model.pkl")

print("Fallback model saved.")
