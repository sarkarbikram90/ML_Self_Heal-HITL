import numpy as np

def inject_data_drift(df, drift_strength=0.4):
    df = df.copy()

    df["monthly_charges"] *= (1 + drift_strength)
    df["support_calls"] += np.random.poisson(1, len(df))

    return df
