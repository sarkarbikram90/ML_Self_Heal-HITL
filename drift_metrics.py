import numpy as np

def prediction_drift_score(pred_probs):
    """
    Measures instability in prediction confidence
    """
    return np.std(pred_probs)
