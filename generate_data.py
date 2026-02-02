import pandas as pd
import numpy as np

def generate_customer_data(n=200):
    return pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "contract_type": np.random.choice(
            ["Month-to-Month", "One Year", "Two Year"], n
        ),
        "support_calls": np.random.poisson(2, n)
    })
