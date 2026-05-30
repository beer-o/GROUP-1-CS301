import os
import pandas as p


def clean_data():
    # Load data relative to this file for consistent paths.
    data_path = os.path.join(os.path.dirname(__file__), "Data", "data.csv")
    data = p.read_csv(data_path)
    data = data.drop(['id','Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data
