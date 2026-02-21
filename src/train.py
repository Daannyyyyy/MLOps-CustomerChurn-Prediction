import joblib
import os

from utils import load_data, split_features_target, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  



def main():
    df = load_data("data/churn.csv")
    print("Pipeline started")
    X, y  = split_features_target(df,"Churn")
    X = preprocess_data(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)   
    print("Model accuracy:", accuracy)
    

    os.makedirs("models", exist_ok=True)

    feature_names = X.columns
    joblib.dump(feature_names, "models/feature_names.pkl")
    joblib.dump(model, "models/churn_model.pkl")

    print("model saved successfully")


if __name__=="__main__":
    main()