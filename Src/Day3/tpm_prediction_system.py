from main import load_and_predict, predict_current_period
import pandas as pd

def main():
    print("=== TPM Prediction System ===")
    
    # Option 1: Predict từ file
    print("\n1. Predicting from file...")
    try:
        predictions = load_and_predict('../../datasets/gamification_traffic_data.csv')
        high_tpm_times = predictions[predictions['probability_high_tpm'] > 0.8]
        print(f"Found {len(high_tpm_times)} high TPM periods")
        print(high_tmp_times[['timestamp', 'probability_high_tpm']].head())
    except Exception as e:
        print(f"Error: {e}")
    
    # Option 2: Real-time prediction
    print("\n2. Real-time prediction...")
    current_prediction = predict_current_period(
        hour=15,  # 3PM
        minute=30,
        day_of_week=1,  # Tuesday
        current_tpm=500,
        previous_tpm_values=[480, 460, 440],
        play_counted=1500
    )
    
    print(f"Current prediction: {current_prediction['recommendation']}")
    print(f"Probability: {current_prediction['probability_high_tpm']:.2%}")

if __name__ == "__main__":
    main()