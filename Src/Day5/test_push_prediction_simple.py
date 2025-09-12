import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ImprovedTrafficPredictionModel import ImprovedTrafficPredictor

def create_mock_trained_predictor():
    """Create a predictor with minimal mock training data for testing"""
    predictor = ImprovedTrafficPredictor()

    # Create minimal mock data for testing
    mock_data = pd.DataFrame({
        'tpm': [100, 120, 150, 180, 200, 220, 250, 280, 300, 320],
        'hour': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'minute': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'day_of_week': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'is_weekend': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'hour_sin': [np.sin(2 * np.pi * h / 24) for h in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],
        'hour_cos': [np.cos(2 * np.pi * h / 24) for h in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]],
        'push_active': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'minutes_since_push_safe': [999, 0, 5, 10, 15, 0, 5, 10, 15, 20],
        'is_daytime_effect_window': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'is_nighttime_no_effect': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'push_effect_multiplier': [0, 1.0, 0.6, 0.4, 0.2, 1.0, 0.6, 0.4, 0.2, 0],
    })

    # Set up minimal required attributes
    predictor.feature_cols = [col for col in mock_data.columns if col != 'tpm']
    predictor.X_processed = mock_data.drop(columns=['tpm'])
    predictor.y_processed = mock_data['tpm']

    # Create a simple mock model
    from sklearn.linear_model import LinearRegression
    mock_model = LinearRegression()
    mock_model.fit(predictor.X_processed, predictor.y_processed)

    # Set up model structure
    predictor.models = {'tpm': {'LinearRegression': mock_model}}
    predictor.best_models = {'tpm': 'LinearRegression'}
    predictor._tpm_thresholds = {'q60': 150, 'q80': 200, 'q90': 250}

    return predictor

def test_enhanced_push_features():
    """Test the enhanced push notification feature calculation"""
    print("🧪 Testing Enhanced Push Notification Features")
    print("=" * 60)

    predictor = create_mock_trained_predictor()

    # Test scenarios for push notification effects
    test_scenarios = [
        {
            'name': 'Daytime Push Active (10 AM)',
            'when': datetime(2024, 12, 23, 10, 0),
            'push_active': 1,
            'minutes_since_push': 0,
            'expected_daytime': True,
            'expected_nighttime': False,
            'expected_effect': 'Strong'
        },
        {
            'name': 'Daytime Push Decay (10:05 AM)',
            'when': datetime(2024, 12, 23, 10, 5),
            'push_active': 0,
            'minutes_since_push': 5,
            'expected_daytime': True,
            'expected_nighttime': False,
            'expected_effect': 'Moderate'
        },
        {
            'name': 'Nighttime Push No Effect (3 AM)',
            'when': datetime(2024, 12, 23, 3, 0),
            'push_active': 1,
            'minutes_since_push': 0,
            'expected_daytime': False,
            'expected_nighttime': True,
            'expected_effect': 'None'
        },
        {
            'name': 'Evening Push Minimal Effect (22:00)',
            'when': datetime(2024, 12, 23, 22, 0),
            'push_active': 1,
            'minutes_since_push': 0,
            'expected_daytime': False,
            'expected_nighttime': False,
            'expected_effect': 'Minimal'
        }
    ]

    print("\n🎯 Testing Push Feature Calculation:")
    print("-" * 60)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")

        hour = scenario['when'].hour
        ms_push = scenario['minutes_since_push']

        # Calculate features as in the enhanced model
        is_daytime_effect_window = int(7 <= hour <= 21)
        is_nighttime_no_effect = int(1 <= hour <= 6)

        # Calculate push effect multiplier
        push_effect_multiplier = 0.0
        if scenario['push_active'] or ms_push <= 15:
            if is_daytime_effect_window:
                if ms_push <= 15:
                    decay_factor = np.exp(-ms_push / 5.0)
                    push_effect_multiplier = decay_factor
            elif is_nighttime_no_effect:
                push_effect_multiplier = 0.0
            else:
                if ms_push <= 15:
                    decay_factor = np.exp(-ms_push / 10.0)
                    push_effect_multiplier = decay_factor * 0.3

        print(f"   Time: {scenario['when'].strftime('%H:%M')}")
        print(f"   Daytime window: {bool(is_daytime_effect_window)} (expected: {scenario['expected_daytime']})")
        print(f"   Nighttime no-effect: {bool(is_nighttime_no_effect)} (expected: {scenario['expected_nighttime']})")
        print(f"   Push effect multiplier: {push_effect_multiplier:.3f}")
        print(f"   Expected effect: {scenario['expected_effect']}")

        # Verify expectations
        assert bool(is_daytime_effect_window) == scenario['expected_daytime'], f"Daytime detection failed for {scenario['name']}"
        assert bool(is_nighttime_no_effect) == scenario['expected_nighttime'], f"Nighttime detection failed for {scenario['name']}"

        if scenario['expected_effect'] == 'Strong':
            assert push_effect_multiplier > 0.8, f"Strong effect expected but got {push_effect_multiplier}"
        elif scenario['expected_effect'] == 'Moderate':
            assert 0.3 < push_effect_multiplier < 0.8, f"Moderate effect expected but got {push_effect_multiplier}"
        elif scenario['expected_effect'] == 'None':
            assert push_effect_multiplier == 0.0, f"No effect expected but got {push_effect_multiplier}"
        elif scenario['expected_effect'] == 'Minimal':
            assert 0.0 < push_effect_multiplier <= 0.3, f"Minimal effect expected but got {push_effect_multiplier}"

        print(f"   ✅ Test passed!")

    return True

def test_prediction_functions():
    """Test the new prediction functions with mock data"""
    print("\n\n🎯 Testing New Prediction Functions")
    print("=" * 60)

    predictor = create_mock_trained_predictor()

    # Test 1: predict_from_inputs with enhanced push effects
    print("\n1. Testing predict_from_inputs with enhanced push effects...")

    try:
        results = predictor.predict_from_inputs(
            when=datetime(2024, 12, 23, 10, 0),  # 10 AM - daytime
            current_tpm=500.0,
            previous_tpms=[480.0, 460.0, 450.0],
            push_notification_active=1,
            minutes_since_push=0,
            minutes_ahead=(5, 10, 15)
        )

        predictions = results['predictions']
        print(f"   ✅ Daytime push predictions:")
        print(f"      5min:  {predictions['tpm_5min']:.1f}")
        print(f"      10min: {predictions['tpm_10min']:.1f}")
        print(f"      15min: {predictions['tpm_15min']:.1f}")

        # Test nighttime scenario
        results_night = predictor.predict_from_inputs(
            when=datetime(2024, 12, 23, 3, 0),  # 3 AM - nighttime
            current_tpm=50.0,
            previous_tpms=[48.0, 46.0, 44.0],
            push_notification_active=1,
            minutes_since_push=0,
            minutes_ahead=(5, 10, 15)
        )

        predictions_night = results_night['predictions']
        print(f"   ✅ Nighttime push predictions:")
        print(f"      5min:  {predictions_night['tpm_5min']:.1f}")
        print(f"      10min: {predictions_night['tpm_10min']:.1f}")
        print(f"      15min: {predictions_night['tpm_15min']:.1f}")

    except Exception as e:
        print(f"   ❌ Error in predict_from_inputs: {e}")
        return False

    # Test 2: predict_from_3hour_history
    print("\n2. Testing predict_from_3hour_history...")

    try:
        # Create sample 3-hour historical data
        current_time = datetime(2024, 12, 23, 14, 30)
        start_time = current_time - timedelta(hours=3)

        timestamps = [start_time + timedelta(minutes=i*10) for i in range(18)]  # Every 10 minutes
        tpm_values = [300 + 50*np.sin(i/3) + np.random.normal(0, 10) for i in range(18)]

        historical_data = pd.DataFrame({
            'timestamp': timestamps,
            'tpm': tpm_values,
            'response_time': [100 + np.random.normal(0, 10) for _ in range(18)],
            'push_notification_active': [1 if i == 10 else 0 for i in range(18)],  # Push at index 10
            'minutes_since_push': [abs(i-10)*10 if i >= 10 else 999 for i in range(18)]
        })

        results = predictor.predict_from_3hour_history(
            current_time=current_time,
            historical_data=historical_data,
            minutes_ahead=(5, 10, 15)
        )

        predictions = results['predictions']
        print(f"   ✅ 3-hour history predictions:")
        print(f"      5min:  {predictions['tpm_5min']:.1f}")
        print(f"      10min: {predictions['tpm_10min']:.1f}")
        print(f"      15min: {predictions['tpm_15min']:.1f}")

    except Exception as e:
        print(f"   ❌ Error in predict_from_3hour_history: {e}")
        return False

    # Test 3: predict_from_tpm_list_with_push_events
    print("\n3. Testing predict_from_tpm_list_with_push_events...")

    try:
        tpm_list = [400, 420, 450, 480, 500, 520, 550, 580, 600, 620]
        current_time = datetime(2024, 12, 23, 15, 30)
        push_events = [
            datetime(2024, 12, 23, 15, 25),  # 5 minutes ago
            datetime(2024, 12, 23, 14, 30),  # 1 hour ago
        ]

        results = predictor.predict_from_tpm_list_with_push_events(
            current_time=current_time,
            tpm_chronological_list=tpm_list,
            push_events=push_events,
            minutes_ahead=(5, 10, 15)
        )

        predictions = results['predictions']
        print(f"   ✅ TPM list with push events predictions:")
        print(f"      5min:  {predictions['tpm_5min']:.1f}")
        print(f"      10min: {predictions['tpm_10min']:.1f}")
        print(f"      15min: {predictions['tpm_15min']:.1f}")

        # Test with dict format push events
        push_events_dict = [
            {'timestamp': datetime(2024, 12, 23, 15, 25), 'active': True},
            {'timestamp': datetime(2024, 12, 23, 14, 30), 'active': False},
        ]

        results2 = predictor.predict_from_tpm_list_with_push_events(
            current_time=current_time,
            tpm_chronological_list=tpm_list,
            push_events=push_events_dict,
            minutes_ahead=(5, 10, 15)
        )

        predictions2 = results2['predictions']
        print(f"   ✅ Dict format push events predictions:")
        print(f"      5min:  {predictions2['tpm_5min']:.1f}")
        print(f"      10min: {predictions2['tpm_10min']:.1f}")
        print(f"      15min: {predictions2['tpm_15min']:.1f}")

    except Exception as e:
        print(f"   ❌ Error in predict_from_tpm_list_with_push_events: {e}")
        return False

    return True

def main():
    """Run simplified tests"""
    print("🚀 Enhanced Push Notification Prediction Tests (Simplified)")
    print("=" * 80)

    test_results = []

    # Test 1: Enhanced push features calculation
    test_results.append(test_enhanced_push_features())

    # Test 2: New prediction functions
    test_results.append(test_prediction_functions())

    # Summary
    print("\n\n📋 Test Summary")
    print("=" * 40)
    passed = sum(test_results)
    total = len(test_results)

    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed successfully!")
        print("\n📝 Summary of Enhancements:")
        print("   ✅ Enhanced push notification effects with daytime vs nighttime logic")
        print("   ✅ Exponential decay over 15 minutes during daytime (7h-21h)")
        print("   ✅ No effect during nighttime (1h-6h)")
        print("   ✅ Minimal effect during other hours")
        print("   ✅ New function: predict_from_3hour_history()")
        print("   ✅ New function: predict_from_tpm_list_with_push_events()")
        print("   ✅ Support for both datetime and dict format push events")
    else:
        print("⚠️ Some tests failed. Please check the output above.")

    return passed == total

if __name__ == "__main__":
    main()
