#!/usr/bin/env python3
"""
Example script demonstrating how to use the Traffic Prediction Model
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from Src.Day4.traffic_prediction_model import TrafficPredictionModel

def main():
    """
    Main function to demonstrate the usage of the Traffic Prediction Model
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and use the Traffic Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--data-path', default=None, help='Path to the CSV data file')
    parser.add_argument('--models-dir', default='Src/Day4/models', help='Directory to save/load models')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    
    args = parser.parse_args()
    
    print("Traffic Prediction Model Example")
    print("===============================")
    
    # Create model
    model = TrafficPredictionModel(data_path=args.data_path)
    
    # Train or load models
    if args.train:
        print("\n=== Training new models ===")
        results = model.train_and_save()
        print("\nTraining completed. Models saved to:", args.models_dir)
    else:
        print("\n=== Loading pre-trained models ===")
        if not os.path.exists(args.models_dir):
            os.makedirs(args.models_dir, exist_ok=True)
            print(f"Models directory {args.models_dir} does not exist or is empty.")
            print("Training new models instead...")
            results = model.train_and_save()
        else:
            success = model.load_models(args.models_dir)
            if not success:
                print("Failed to load models. Training new models instead...")
                results = model.train_and_save()
    
    # Make predictions for different scenarios
    print("\n=== Making predictions for different scenarios ===")
    
    # Scenario 1: Regular weekday traffic
    prediction1 = model.predict(
        current_tpm=500,
        hour=14,  # 2 PM
        day_of_week=2,  # Wednesday
        response_time=100,
        previous_tpm_values=[480, 450, 420, 400, 380],
        push_notification_active=0
    )
    
    print("\nScenario 1: Regular weekday traffic (Wednesday 2 PM)")
    print("Current TPM: 500")
    for target, value in prediction1.items():
        print(f"  {target}: {value:.2f} TPM")
    
    # Scenario 2: Weekend traffic
    prediction2 = model.predict(
        current_tpm=800,
        hour=15,  # 3 PM
        day_of_week=5,  # Saturday
        response_time=120,
        previous_tpm_values=[750, 700, 650, 600, 550],
        push_notification_active=0
    )
    
    print("\nScenario 2: Weekend traffic (Saturday 3 PM)")
    print("Current TPM: 800")
    for target, value in prediction2.items():
        print(f"  {target}: {value:.2f} TPM")
    
    # Scenario 3: With active push notification
    prediction3 = model.predict(
        current_tpm=500,
        hour=14,
        day_of_week=2,  # Wednesday
        response_time=100,
        previous_tpm_values=[480, 450, 420, 400, 380],
        push_notification_active=1,
        minutes_since_push=0
    )
    
    print("\nScenario 3: With active push notification (just sent)")
    print("Current TPM: 500")
    for target, value in prediction3.items():
        print(f"  {target}: {value:.2f} TPM")
        
    # Calculate percentage increase due to push notification
    print("\nPercentage increase due to push notification:")
    for target in prediction1.keys():
        increase = (prediction3[target] - prediction1[target]) / prediction1[target] * 100
        print(f"  {target}: {increase:.1f}%")
    
    # Scenario 4: 5 minutes after push notification
    prediction4 = model.predict(
        current_tpm=550,  # Increased due to push notification
        hour=14,
        day_of_week=2,  # Wednesday
        response_time=110,
        previous_tpm_values=[500, 480, 450, 420, 400],
        push_notification_active=0,
        minutes_since_push=5
    )
    
    print("\nScenario 4: 5 minutes after push notification")
    print("Current TPM: 550")
    for target, value in prediction4.items():
        print(f"  {target}: {value:.2f} TPM")
    
    # Visualize predictions if requested
    if args.visualize:
        print("\n=== Visualizing predictions ===")
        model.visualize_predictions()
        
        # Create a simple bar chart comparing the scenarios
        scenarios = ['Regular', 'Weekend', 'Push Notification', '5min After Push']
        targets = list(prediction1.keys())
        
        # Create figure with subplots for each target
        fig, axes = plt.subplots(len(targets), 1, figsize=(10, 12))
        
        for i, target in enumerate(targets):
            values = [prediction1[target], prediction2[target], 
                     prediction3[target], prediction4[target]]
            
            axes[i].bar(scenarios, values, color=['blue', 'green', 'red', 'orange'])
            axes[i].set_title(f'Predicted TPM for {target}')
            axes[i].set_ylabel('TPM')
            
            # Add value labels on top of bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 5, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        plt.savefig('Src/Day4/scenario_comparison.png')
        print("Scenario comparison chart saved to Src/Day4/scenario_comparison.png")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()