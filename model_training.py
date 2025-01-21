import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):

        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
        
    return np.array(X), np.array(y)

def scale_features(X, y,feature_columns):
    feature_scaler = StandardScaler()
    features = X[feature_columns]
    scaled_features = feature_scaler.fit_transform(features)

    target_scaler = MinMaxScaler()
    scaled_target = target_scaler.fit_transform([y])

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):

        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
        
    return np.array(X), np.array(y)

def check_data_quality(df):
  
    print("Missing values:\n", df.isnull().sum())
    
    print("\nInfinite values:\n", np.isinf(df).sum())
    
    z_scores = stat.zscore(df)
    print("\nExtreme outliers:\n", (abs(z_scores) > 3).sum())

def clean_data(df):
    
    df = df.fillna(method='ffill')
    
    df = df.fillna(method='bfill')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill')
    
    return df
