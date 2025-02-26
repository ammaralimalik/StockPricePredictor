from model import StockPricePredictor, StockPrice_Model, scale_features, train_scale_features
import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import torch 

data = pd.read_csv('/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/data/indexData.csv')

stock_data = metrics.calculate_all_features(data)

feature_columns = ['Open','High','Low','Volume','Returns','MA10', 'MA50', 'RSI', 'ATR', 'Volume_Norm', 'Volatility']
target_feature = ['Close']

X, y = train_scale_features(stock_data, feature_columns, target_feature)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    shuffle=False
)

X_train[X_train == 0] = 0.1

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

nn_model = StockPrice_Model(
    model=StockPricePredictor(len(feature_columns)),
    lr=0.0001,
    epochs=100,
    batch_size=64,
    l2_lam=0.001
)

nn_model.train(X_train_tensor, y_train_tensor)

nn_model.save_model()