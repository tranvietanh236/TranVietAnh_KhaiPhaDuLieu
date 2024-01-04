import pandas as pd

def load_data(url):
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)
    return data