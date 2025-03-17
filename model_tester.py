import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
from model import StockPrice_Model, train_scale_features
from sklearn.model_selection import train_test_split
import metrics
import torch
import warnings
import os
warnings.filterwarnings("ignore")


class ModelTester:
    def __init__(self, model):
        self.model = model
        self.save_path = '/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/model_artifacts'
        # Create directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate various regression metrics
        """
        # Convert to numpy arrays and flatten
        y_true = y_true.numpy().flatten()
        y_pred = y_pred.numpy().flatten()
        
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Direction Accuracy': self.direction_accuracy(y_true, y_pred)
        }
        
        print('Calculated metrics')
        return metrics
    
    def direction_accuracy(self, y_true, y_pred):
        """
        Calculate the directional accuracy (up/down movement prediction)
        """
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        print('Found directional accuracy')
        return np.mean(true_direction == pred_direction) * 100
    
    def plot_predictions(self, y_true, y_pred, title="Actual vs Predicted Values"):
        """
        Plot actual vs predicted values
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        save_path = os.path.join(self.save_path, 'actual_vs_predicted.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")
        
        plt.show()
        
    def plot_regression(self, y_true, y_pred):
        """
        Create a regression plot with confidence intervals
        """
        plt.figure(figsize=(10, 10))
        
        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
        line = slope * y_true + intercept
        
        # Plot scatter and regression line
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(y_true, line, color='red', label=f'R² = {r_value**2:.3f}')
        
        plt.title('Actual vs Predicted Regression Plot')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        
        # Add perfect prediction line
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                '--', color='gray', label='Perfect Prediction')
        
        plt.grid(True)
        
        # Save the plot
        save_path = os.path.join(self.save_path, 'regression_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved regression plot to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """
        Plot residuals to check for patterns
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        
        # Save the plot
        save_path = os.path.join(self.save_path, 'residual_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved residual plot to {save_path}")
        
        plt.show()
        
        # Plot residual distribution
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.75)
        plt.title('Residual Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Save the plot
        save_path = os.path.join(self.save_path, 'residual_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved residual distribution plot to {save_path}")
        
        plt.show()

# Example usage code:
def test_model(X_test, y_test, model_path):
    """
    Complete testing procedure
    """
    # Load model
    stock_model = StockPrice_Model()
    stock_model.load_model(model_path)
    
    # Make predictions
    predictions = stock_model.predict(X_test)
    
    # Initialize tester
    tester = ModelTester(stock_model)
    
    # Calculate metrics
    metrics = tester.calculate_metrics(y_test, predictions)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    tester.plot_predictions(y_test, predictions)
    tester.plot_regression(y_test.numpy().flatten(), predictions.numpy().flatten())
    tester.plot_residuals(y_test.numpy().flatten(), predictions.numpy().flatten())
    
    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_save_path = os.path.join(tester.save_path, 'model_metrics.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Saved metrics to {metrics_save_path}")
    
    print('Testing Completed')
    return metrics, predictions


stock_data = pd.read_csv('data/indexProcessed.csv')
stock_data = metrics.calculate_all_features(stock_data)

feature_columns = ['Open','High','Low','Volume','Returns','MA10', 'MA50', 'RSI', 'ATR', 'Volume_Norm', 'Volatility']
target_feature = ['Close']

X, y = train_scale_features(stock_data, feature_columns, target_feature)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.95,
    shuffle=False
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

X_test = X_test_tensor
y_test = y_test_tensor  
model_path = '/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/saved_models/model.pt'      

y_test[y_test == 0] = 0.01
metrics, predictions = test_model(X_test, y_test, model_path)