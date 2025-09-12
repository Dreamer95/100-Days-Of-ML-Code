import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ImprovedTrafficPredictionModel import ImprovedTrafficPredictor

def test_enhanced_push_notification_effects():
    """Test the enhanced push notification effects with daytime vs nighttime logic"""
    print("🧪 Testing Enhanced Push Notification Effects")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedTrafficPredictor()
    
    # Load and train model (if not already trained)
    try:
        print("📊 Loading data and training model...")
        data = predictor.load_real_data()
        results = predictor.train_and_evaluate_optimized()
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Daytime Push Effect (10 AM)',
            'when': datetime(2024, 12, 23, 10, 0),  # 10 AM - daytime
            'current_tpm': 500.0,
            'previous_tpms': [480.0, 460.0, 450.0, 440.0, 420.0],
            'push_notification_active': 1,
            'minutes_since_push': 0,
            'expected_effect': 'Strong increase due to daytime push'
        },
        {
            'name': 'Daytime Push Decay (10:05 AM)',
            'when': datetime(2024, 12, 23, 10, 5),  # 10:05 AM - 5 minutes after push
            'current_tpm': 600.0,
            'previous_tpms': [580.0, 560.0, 540.0, 520.0, 500.0],
            'push_notification_active': 0,
            'minutes_since_push': 5,
            'expected_effect': 'Moderate increase with decay'
        },
        {
            'name': 'Daytime Push Fully Decayed (10:20 AM)',
            'when': datetime(2024, 12, 23, 10, 20),  # 10:20 AM - 20 minutes after push
            'current_tpm': 450.0,
            'previous_tpms': [460.0, 470.0, 480.0, 490.0, 500.0],
            'push_notification_active': 0,
            'minutes_since_push': 20,
            'expected_effect': 'No push effect (beyond 15 minutes)'
        },
        {
            'name': 'Nighttime Push No Effect (3 AM)',
            'when': datetime(2024, 12, 23, 3, 0),  # 3 AM - nighttime
            'current_tpm': 50.0,
            'previous_tpms': [48.0, 46.0, 44.0, 42.0, 40.0],
            'push_notification_active': 1,
            'minutes_since_push': 0,
            'expected_effect': 'No effect during nighttime (1-6 AM)'
        },
        {
            'name': 'Evening Push Minimal Effect (22:00)',
            'when': datetime(2024, 12, 23, 22, 0),  # 10 PM - outside daytime window
            'current_tpm': 200.0,
            'previous_tpms': [190.0, 180.0, 170.0, 160.0, 150.0],
            'push_notification_active': 1,
            'minutes_since_push': 0,
            'expected_effect': 'Minimal effect outside daytime window'
        }
    ]
    
    print("\n🎯 Testing Push Notification Scenarios:")
    print("-" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Time: {scenario['when'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Expected: {scenario['expected_effect']}")
        
        try:
            results = predictor.predict_from_inputs(
                when=scenario['when'],
                current_tpm=scenario['current_tpm'],
                previous_tpms=scenario['previous_tpms'],
                push_notification_active=scenario['push_notification_active'],
                minutes_since_push=scenario['minutes_since_push'],
                minutes_ahead=(5, 10, 15)
            )
            
            predictions = results['predictions']
            print(f"   Results:")
            print(f"     5min:  {predictions['tpm_5min']:.1f}")
            print(f"     10min: {predictions['tpm_10min']:.1f}")
            print(f"     15min: {predictions['tpm_15min']:.1f}")
            
            # Calculate effect magnitude
            base_tpm = scenario['current_tpm']
            effect_5min = (predictions['tpm_5min'] - base_tpm) / base_tpm * 100
            print(f"     Effect: {effect_5min:+.1f}% at 5min")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

def test_3hour_history_prediction():
    """Test the predict_from_3hour_history function"""
    print("\n\n🕐 Testing 3-Hour History Prediction")
    print("=" * 60)
    
    predictor = ImprovedTrafficPredictor()
    
    # Create sample 3-hour historical data
    current_time = datetime(2024, 12, 23, 14, 30)  # 2:30 PM
    start_time = current_time - timedelta(hours=3)
    
    # Generate sample data points every 5 minutes for 3 hours
    timestamps = []
    tpm_values = []
    response_times = []
    push_active_values = []
    minutes_since_push_values = []
    
    for i in range(36):  # 3 hours * 12 (5-minute intervals)
        ts = start_time + timedelta(minutes=i * 5)
        timestamps.append(ts)
        
        # Simulate TPM pattern with some randomness
        base_tpm = 300 + 100 * np.sin(2 * np.pi * ts.hour / 24) + np.random.normal(0, 20)
        tpm_values.append(max(10, base_tpm))
        
        response_times.append(100 + np.random.normal(0, 20))
        
        # Simulate push notification at 1:00 PM (13:00)
        if ts.hour == 13 and ts.minute == 0:
            push_active_values.append(1)
            minutes_since_push_values.append(0)
        else:
            push_active_values.append(0)
            # Calculate minutes since 1:00 PM push
            push_time = datetime(2024, 12, 23, 13, 0)
            if ts >= push_time:
                minutes_since = int((ts - push_time).total_seconds() / 60)
                minutes_since_push_values.append(minutes_since)
            else:
                minutes_since_push_values.append(999)
    
    # Create DataFrame
    historical_data = pd.DataFrame({
        'timestamp': timestamps,
        'tpm': tpm_values,
        'response_time': response_times,
        'push_notification_active': push_active_values,
        'minutes_since_push': minutes_since_push_values
    })
    
    print(f"📊 Created historical data: {len(historical_data)} points over 3 hours")
    print(f"   Time range: {historical_data['timestamp'].min()} to {historical_data['timestamp'].max()}")
    print(f"   TPM range: {historical_data['tpm'].min():.1f} to {historical_data['tpm'].max():.1f}")
    print(f"   Push events: {historical_data['push_notification_active'].sum()}")
    
    try:
        # Load model first
        data = predictor.load_real_data()
        results = predictor.train_and_evaluate_optimized()
        
        # Test the 3-hour history prediction
        prediction_results = predictor.predict_from_3hour_history(
            current_time=current_time,
            historical_data=historical_data,
            minutes_ahead=(5, 10, 15)
        )
        
        print(f"\n✅ 3-Hour History Prediction Results:")
        predictions = prediction_results['predictions']
        print(f"   5min ahead:  {predictions['tpm_5min']:.1f}")
        print(f"   10min ahead: {predictions['tpm_10min']:.1f}")
        print(f"   15min ahead: {predictions['tpm_15min']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in 3-hour history prediction: {e}")
        return False

def test_tpm_list_with_push_events():
    """Test the predict_from_tpm_list_with_push_events function"""
    print("\n\n🎯 Testing TPM List with Push Events Prediction")
    print("=" * 60)
    
    predictor = ImprovedTrafficPredictor()
    
    # Create sample TPM chronological list
    tpm_list = [
        400.0, 420.0, 450.0, 480.0, 500.0, 520.0, 550.0, 580.0,
        600.0, 620.0, 640.0, 660.0, 680.0, 700.0, 720.0, 740.0
    ]
    
    # Create push events
    current_time = datetime(2024, 12, 23, 15, 30)  # 3:30 PM
    push_events = [
        datetime(2024, 12, 23, 15, 25),  # 5 minutes ago
        datetime(2024, 12, 23, 14, 30),  # 1 hour ago
    ]
    
    print(f"📈 TPM chronological list: {len(tpm_list)} values")
    print(f"   Range: {min(tpm_list):.1f} to {max(tpm_list):.1f}")
    print(f"   Latest: {tpm_list[-1]:.1f}")
    print(f"🔔 Push events: {len(push_events)}")
    for i, event in enumerate(push_events):
        minutes_ago = int((current_time - event).total_seconds() / 60)
        print(f"   {i+1}. {event.strftime('%H:%M:%S')} ({minutes_ago} minutes ago)")
    
    try:
        # Load model first
        data = predictor.load_real_data()
        results = predictor.train_and_evaluate_optimized()
        
        # Test the TPM list with push events prediction
        prediction_results = predictor.predict_from_tpm_list_with_push_events(
            current_time=current_time,
            tpm_chronological_list=tpm_list,
            push_events=push_events,
            minutes_ahead=(5, 10, 15),
            response_time=120.0
        )
        
        print(f"\n✅ TPM List with Push Events Prediction Results:")
        predictions = prediction_results['predictions']
        print(f"   5min ahead:  {predictions['tpm_5min']:.1f}")
        print(f"   10min ahead: {predictions['tpm_10min']:.1f}")
        print(f"   15min ahead: {predictions['tpm_15min']:.1f}")
        
        # Test with dict format push events
        push_events_dict = [
            {'timestamp': datetime(2024, 12, 23, 15, 25), 'active': True},
            {'timestamp': datetime(2024, 12, 23, 14, 30), 'active': False},  # Inactive event
        ]
        
        print(f"\n🔄 Testing with dict format push events...")
        prediction_results_2 = predictor.predict_from_tpm_list_with_push_events(
            current_time=current_time,
            tpm_chronological_list=tpm_list,
            push_events=push_events_dict,
            minutes_ahead=(5, 10, 15)
        )
        
        predictions_2 = prediction_results_2['predictions']
        print(f"   5min ahead:  {predictions_2['tpm_5min']:.1f}")
        print(f"   10min ahead: {predictions_2['tpm_10min']:.1f}")
        print(f"   15min ahead: {predictions_2['tpm_15min']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in TPM list with push events prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced Push Notification Prediction Tests")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Enhanced push notification effects
    test_results.append(test_enhanced_push_notification_effects())
    
    # Test 2: 3-hour history prediction
    test_results.append(test_3hour_history_prediction())
    
    # Test 3: TPM list with push events prediction
    test_results.append(test_tpm_list_with_push_events())
    
    # Summary
    print("\n\n📋 Test Summary")
    print("=" * 40)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
    
    return passed == total

if __name__ == "__main__":
    main()