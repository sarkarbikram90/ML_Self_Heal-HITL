import pandas as pd

def preprocess(df):
    df = df.copy()
    df["contract_type"] = df["contract_type"].map({
        "Month-to-Month": 0,
        "One Year": 1,
        "Two Year": 2
    })
    return df
