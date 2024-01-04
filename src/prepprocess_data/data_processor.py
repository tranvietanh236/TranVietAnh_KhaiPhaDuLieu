from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    data = data.dropna()
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data
