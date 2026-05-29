import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_pickle():
    # Find data.csv relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), "Data", "data.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Cannot find dataset at {data_path}")
        return

    # Load dataset
    df = pd.read_csv(data_path)
    df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    feature_defaults = X.mean().to_dict()

    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Wrap up model and defaults
    payload = {"model": model, "defaults": feature_defaults}

    # Save to the exact same folder as train.py
    model_output_path = os.path.join(current_dir, "cancer_model.pkl")
    with open(model_output_path, "wb") as file:
        pickle.dump(payload, file)

    print(f"🎉 Success! Generated model at: {model_output_path}")

if __name__ == "__main__":
    train_and_pickle()