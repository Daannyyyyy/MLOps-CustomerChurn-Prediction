import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load_data(path):
    df = pd.read_csv(path)
    print("Data Loaded successfully")
    print(f"Shape:{df.shape}")
    return df

def split_features_target(df, target_column):
    X = df.drop(target_column, axis = 1)
    y = df[target_column]
    print("Features and target are separated")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
    return X,y

def preprocess_data(X):
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes
        else:
            X[col] = X[col].fillna(0)

    print("Data preprocessing completed")
    return X
