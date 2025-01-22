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

class StockPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPricePredictor, self).__init__()
        
        self.ltsm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 1)
    
    def forward(self, x, lengths=None):
        
        if len(x.shape) == 2:
            x = x.view(-1, self.sequence_length, self.input_size)
        
        if lengths is not None:
            # Pack the padded sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            packed_lstm_out, _ = self.lstm(packed_x)
            
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_lstm_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)
            
            
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class StockPrice_Model:
    
    def __init__(self, model, lr=0.001, epoch=25, batch_size=64, l2_lam=0.1):
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.l2_lam = l2_lam
        
        self.metric = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
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
            
            # L2 regularization
            l2_reg = sum(param.norm(2) for param in self.model.parameters())
            loss += self.l2_lambda * l2_reg
            
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
