# Simple Linear Regression Implementation
# Based on Day2_Simple_Linear_Regression.md

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

def main():
    print("=== Simple Linear Regression ===")
    
    # Step 1: Data Preprocessing
    print("\nStep 1: Data Preprocessing")
    
    try:
        # Try relative path first, then absolute path
        data_paths = [
            '../../datasets/studentscores.csv',
            '/Users/dongdinh/Documents/Learning/100-Days-Of-ML-Code/datasets/studentscores.csv'
        ]
        
        dataset = None
        for path in data_paths:
            if os.path.exists(path):
                dataset = pd.read_csv(path)
                print(f"✅ Dataset loaded from: {path}")
                break
        
        if dataset is None:
            raise FileNotFoundError("studentscores.csv not found in any of the expected locations")
        
        # Display dataset info
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset columns: {list(dataset.columns)}")
        print("\nFirst 5 rows:")
        print(dataset.head())
        
        # Extract features and target
        X = dataset.iloc[:, :-1].values  # All columns except last (features)
        Y = dataset.iloc[:, -1].values   # Last column (target)
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {Y.shape}")
        
        # Split the dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=0
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return
    
    # Step 2: Fitting Simple Linear Regression Model to the training set
    print("\nStep 2: Training Simple Linear Regression Model")
    
    try:
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)
        
        # Display model parameters
        print(f"✅ Model trained successfully!")
        print(f"Coefficient (slope): {regressor.coef_[0]:.4f}")
        print(f"Intercept: {regressor.intercept_:.4f}")
        print(f"Equation: Y = {regressor.coef_[0]:.4f} * X + {regressor.intercept_:.4f}")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    # Step 3: Predicting the Results
    print("\nStep 3: Making Predictions")
    
    try:
        Y_pred = regressor.predict(X_test)
        
        # Display some predictions
        print("Sample predictions vs actual values:")
        for i in range(min(5, len(Y_test))):
            print(f"Predicted: {Y_pred[i]:.2f}, Actual: {Y_test[i]:.2f}, Difference: {abs(Y_pred[i] - Y_test[i]):.2f}")
        
        # Calculate performance metrics
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, Y_pred)
        
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return
    
    # Step 4: Visualization
    print("\nStep 4: Creating Visualizations")
    
    try:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set visualization
        ax1.scatter(X_train, Y_train, color='red', alpha=0.7, label='Training Data')
        ax1.plot(X_train, regressor.predict(X_train), color='blue', linewidth=2, label='Regression Line')
        ax1.set_title('Simple Linear Regression (Training Set)')
        ax1.set_xlabel('Hours Studied')
        ax1.set_ylabel('Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test set visualization
        ax2.scatter(X_test, Y_test, color='red', alpha=0.7, label='Test Data')
        ax2.plot(X_test, regressor.predict(X_test), color='blue', linewidth=2, label='Regression Line')
        ax2.set_title('Simple Linear Regression (Test Set)')
        ax2.set_xlabel('Hours Studied')
        ax2.set_ylabel('Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations created successfully!")
        
        # Additional visualization: Residual plot
        plt.figure(figsize=(10, 6))
        residuals = Y_test - Y_pred
        plt.scatter(Y_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
    
    print("\n=== Simple Linear Regression Complete! ===")

# Additional utility functions
def predict_single_value(regressor, hours):
    """Predict score for a single input"""
    prediction = regressor.predict([[hours]])
    return prediction[0]

def display_dataset_info(dataset):
    """Display comprehensive dataset information"""
    print("\n=== Dataset Information ===")
    print(f"Shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")
    print("\nStatistical Summary:")
    print(dataset.describe())
    print("\nMissing Values:")
    print(dataset.isnull().sum())

if __name__ == "__main__":
    main()