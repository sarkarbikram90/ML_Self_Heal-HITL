import numpy as np

class SimpleChurnModel:
    def predict_proba(self, X):
        # Simple rule-based probability (believable)
        score = (
            0.3
            + 0.01 * X["monthly_charges"]
            - 0.005 * X["tenure"]
            + 0.1 * X["support_calls"]
        )

        prob = 1 / (1 + np.exp(-score))
        prob = np.clip(prob, 0, 1)

        return np.vstack([1 - prob, prob]).T
