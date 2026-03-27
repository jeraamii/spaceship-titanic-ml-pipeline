import pandas as pd

def load_data():
    df = pd.read_csv("train.csv")
    return df
