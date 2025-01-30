import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import statistics as stats
import time
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class StockPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPricePredictor, self).__init__()
     
        self.fc1 = nn.Linear(input_size, 128)
        self.act1 = nn.LeakyReLU(0.01)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.LeakyReLU(0.01)
        self.dropout2 = nn.Dropout(p=0.35)
        self.fc3 = nn.Linear(64, 32)
        self.act3 = nn.LeakyReLU(0.01)
        self.dropout3 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(32, 32)
        self.act4 = nn.LeakyReLU(0.01)
        self.dropout4 = nn.Dropout(p=0.25)
        self.fc5 = nn.Linear(32,16)
        self.act5 = nn.LeakyReLU(0.01)
        self.fc6 = nn.Linear(16,1)
    
    def forward(self, x):
            
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.act5(x)
        x = self.fc6(x)
        return x


class StockPrice_Model:
    
    def __init__(self, model=None, lr=0.001, epochs=25, batch_size=64, l2_lam=0.1):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lam = l2_lam
        
        if model is not None:
            self.metric = nn.SmoothL1Loss()
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lam)
        
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
    def prepare_data(self, X_train, y_train):
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True
        )
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in self.train_loader:
            y_batch = y_batch.view(-1, 1)
            predictions = self.model(X_batch)
            loss = self.metric(predictions, y_batch)
            
            epoch_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return epoch_loss / len(self.train_loader)

    def train(self, X_train, y_train):
        self.prepare_data(X_train, y_train)
        patience_counter = 0
        
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch()
            self.train_losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0 and epoch != (self.epochs - 1):
                time.sleep(45)
                
            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.2f}")
    
    def save_model(self, file_name='model.pt'):
        path = '/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/saved_models/'
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        full_path = os.path.join(path, file_name)
        torch.save(self.model, full_path)
        print(f'Model saved to {full_path}')
        
    
    def load_model(self, file_name='model.pt'):
        
        path = '/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/saved_models/'
        
        full_path = os.path.join(path, file_name)
        self.model = torch.load(full_path)
        self.model.eval()
        print('Model loaded successfully')
    
    def predict(self, X):
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions
        

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):

        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
        
    return np.array(X), np.array(y)

def scale_features(data,feature_columns, target_feature):
    feature_scaler = StandardScaler()
    features = data[feature_columns]
    scaled_x = feature_scaler.fit_transform(features)

    target_scaler = MinMaxScaler()
    scaled_y = target_scaler.fit_transform(data[target_feature])
    
    return scaled_x, scaled_y

def check_data_quality(df):
  
    print("Missing values:\n", df.isnull().sum())
    
    print("\nInfinite values:\n", np.isinf(df).sum())
    
    z_scores = stats.zscore(df)
    print("\nExtreme outliers:\n", (abs(z_scores) > 3).sum())

def clean_data(df):
    
    df = df.fillna(method='ffill')
    
    df = df.fillna(method='bfill')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill')
    
    return df
