import pandas as pd
import numpy as np

def calculate_returns(stock_data):
    return stock_data['Adj Close'].pct_change()

def calculate_moving_averages(stock_data, windows=[10, 50]):
 
    ma_dict = {}
    for window in windows:
        ma_dict[f'MA{window}'] = stock_data['Adj Close'].rolling(window=window).mean()
    return pd.DataFrame(ma_dict)

def calculate_volatility(returns, window=20):
    return returns.rolling(window=window).std()

def calculate_rsi(stock_data, periods=14):
    delta = stock_data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(stock_data, window=14):
    high_low = stock_data['High'] - stock_data['Low']
    high_close = abs(stock_data['High'] - stock_data['Adj Close'].shift())
    low_close = abs(stock_data['Low'] - stock_data['Adj Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def normalize_volume(stock_data, window=20):
    return (stock_data['Volume'] - stock_data['Volume'].rolling(window=window).mean()) / \
           stock_data['Volume'].rolling(window=window).std()

def calculate_price_gap(stock_data):
    return stock_data['Open'] - stock_data['Adj Close'].shift(1)

def calculate_hl_range(stock_data):
   
    return (stock_data['High'] - stock_data['Low']) / stock_data['Adj Close']

def prepare_stock_data(df, stock_id_column='Index'):
   
    if stock_id_column not in df.columns:
        raise KeyError(f"Missing stock identifier column: {stock_id_column}")
    
    if not df.index.nlevels == 2:
        df = df.set_index([stock_id_column, 'Date'])
    return df

def calculate_single_stock_features(stock_data):
    features = stock_data.copy()
    
    # Calculate each feature
    features['Returns'] = calculate_returns(stock_data)
    features = features.join(calculate_moving_averages(stock_data))
    features['Volatility'] = calculate_volatility(features['Returns'])
    features['RSI'] = calculate_rsi(stock_data)
    features['ATR'] = calculate_atr(stock_data)
    features['Volume_Norm'] = normalize_volume(stock_data)
    features['Gap'] = calculate_price_gap(stock_data)
    features['HL_Range'] = calculate_hl_range(stock_data)
    
    return features

def calculate_all_features(df, stock_id_column='Index'):

    try:
        # Prepare data
        df_features = prepare_stock_data(df.copy(), stock_id_column)
        
        # Calculate features for each stock
        df_features = df_features.groupby(level=0, group_keys=False).apply(calculate_single_stock_features)
        
        # Clean up NaN values
        df_features = df_features.dropna()
        
        return df_features
    
    except KeyError as e:
        print(f"Error: Missing required column - {str(e)}")
        print("Required columns: Symbol/Index, Date, Open, High, Low, Close, Adj Close, Volume")
        return None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None
