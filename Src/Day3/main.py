import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# 1. Load và preprocessing data
def load_and_preprocess_data(file_path):
    """Load và tiền xử lý dữ liệu"""
    df = pd.read_csv(file_path)

    # Chuyển đổi timestamp - sử dụng format tự động hoặc ISO8601
    try:
        # Thử với format ISO8601 trước
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    except ValueError:
        try:
            # Nếu không được, thử với format ban đầu
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
        except ValueError:
            # Nếu cả hai đều không được, để pandas tự động detect
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


# 2. Feature Engineering
def create_features(df):
    """Tạo các features từ dữ liệu gốc"""
    df_features = df.copy()

    # Trích xuất features từ timestamp
    df_features['hour'] = df_features['timestamp'].dt.hour
    df_features['minute'] = df_features['timestamp'].dt.minute
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

    # Tạo features về thời gian trong ngày
    df_features['time_of_day'] = df_features['hour'] + df_features['minute'] / 60

    # Tạo features về chu kỳ trong ngày
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

    # Features về các giờ cao điểm đã biết
    df_features['is_peak_hour'] = ((df_features['hour'] >= 0) & (df_features['hour'] <= 1) |
                                   (df_features['hour'] >= 9) & (df_features['hour'] <= 10) |
                                   (df_features['hour'] >= 12) & (df_features['hour'] <= 13) |
                                   (df_features['hour'] >= 19) & (df_features['hour'] <= 20)).astype(int)

    # Lag features (giá trị trước đó)
    df_features['tpm_lag_1'] = df_features['tpm'].shift(1)  # 5 phút trước
    df_features['tpm_lag_2'] = df_features['tpm'].shift(2)  # 10 phút trước
    df_features['tpm_lag_3'] = df_features['tpm'].shift(3)  # 15 phút trước

    # Rolling average features
    df_features['tpm_rolling_mean_6'] = df_features['tpm'].rolling(window=6, min_periods=1).mean()  # 30 phút
    df_features['tpm_rolling_mean_12'] = df_features['tpm'].rolling(window=12, min_periods=1).mean()  # 1 giờ

    # Features về xu hướng tăng/giảm
    df_features['tpm_trend'] = df_features['tpm'] - df_features['tpm_lag_1']

    # Features về play_counted
    df_features['play_counted_rate'] = df_features['play_counted'].pct_change().fillna(0)

    # Features về push notification effect (45 phút sau push)
    df_features['minutes_since_push'] = 0
    push_indices = df_features[df_features['is_push_notification'] == 1].index

    for push_idx in push_indices:
        # Đánh dấu 45 phút (9 records) sau push notification
        end_idx = min(push_idx + 9, len(df_features))
        for i in range(push_idx, end_idx):
            df_features.loc[i, 'minutes_since_push'] = (i - push_idx) * 5

    df_features['is_post_push_period'] = (df_features['minutes_since_push'] <= 45).astype(int)

    return df_features


# 3. Định nghĩa target variable
def create_target_variable(df, threshold_percentile=75):
    """
    Tạo target variable: 1 nếu tpm cao, 0 nếu tpm thấp
    Sử dụng percentile để định nghĩa 'cao'
    """
    threshold = df['tpm'].quantile(threshold_percentile / 100)
    df['is_high_tpm'] = (df['tpm'] >= threshold).astype(int)

    print(f"Threshold for high TPM: {threshold:.0f}")
    print(f"Percentage of high TPM records: {df['is_high_tpm'].mean() * 100:.1f}%")

    return df, threshold


# 4. Train models
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models và so sánh performance"""

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Better parameters for TPM prediction
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,          # More trees for better stability
            max_depth=12,              # Deeper for complex patterns
            min_samples_split=5,       # Prevent overfitting
            random_state=42
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,          # More boosting stages
            max_depth=6,               # Keep moderate depth
            learning_rate=0.1,         # Control overfitting
            subsample=0.8,             # Bootstrap sampling
            random_state=42
        ),
        
        'Logistic Regression': LogisticRegression(
            C=1.0,                     # Regularization strength
            max_iter=2000,             # More iterations
            solver='liblinear',        # Good for small datasets
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate
        accuracy = model.score(X_test_scaled if name == 'Logistic Regression' else X_test, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        trained_models[name] = {
            'model': model,
            'scaler': scaler if name == 'Logistic Regression' else None
        }

        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return results, trained_models


# 5. Feature importance analysis
def analyze_feature_importance(model, feature_names, model_name):
    """Phân tích độ quan trọng của features"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"\n--- Top 10 Important Features for {model_name} ---")
        print(feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()

        return feature_importance
    else:
        print(f"Model {model_name} doesn't have feature_importances_ attribute")
        return None


# 6. Prediction function cho thời gian thực
def predict_high_tpm_periods(model, scaler, new_data, threshold):
    """Dự đoán thời điểm tpm cao cho dữ liệu mới"""

    # Tạo features cho dữ liệu mới
    new_features = create_features(new_data)

    # Lọc ra các columns cần thiết (tương tự như training)
    feature_columns = [col for col in new_features.columns
                       if col not in ['timestamp', 'tpm', 'is_high_tpm']]

    X_new = new_features[feature_columns].fillna(0)

    # Predict
    if scaler is not None:
        X_new_scaled = scaler.transform(X_new)
        probabilities = model.predict_proba(X_new_scaled)[:, 1]
        predictions = model.predict(X_new_scaled)
    else:
        probabilities = model.predict_proba(X_new)[:, 1]
        predictions = model.predict(X_new)

    # Tạo kết quả
    results = new_features[['timestamp']].copy()
    results['predicted_high_tpm'] = predictions
    results['probability_high_tpm'] = probabilities
    results['actual_tpm'] = new_features['tpm']

    return results


# 7. Main execution
def main():
    # Load data
    print("Loading data...")
    df = load_and_preprocess_data('../../datasets/gamification_traffic_data.csv')

    # Create features
    print("Creating features...")
    df_with_features = create_features(df)

    # Create target variable
    print("Creating target variable...")
    df_final, tpm_threshold = create_target_variable(df_with_features, threshold_percentile=75)

    # Prepare features và target
    feature_columns = [col for col in df_final.columns
                       if col not in ['timestamp', 'tpm', 'is_high_tpm']]

    X = df_final[feature_columns].fillna(0)  # Fill NaN values
    y = df_final['is_high_tpm']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {len(feature_columns)}")

    # Train models
    print("\nTraining models...")
    results, trained_models = train_models(X_train, X_test, y_train, y_test)

    # Tìm model tốt nhất
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
    best_model = results[best_model_name]['model']

    print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.3f})")

    # Analyze feature importance
    feature_importance = analyze_feature_importance(
        best_model, feature_columns, best_model_name
    )

    # Save model để sử dụng sau
    import joblib
    joblib.dump(trained_models[best_model_name], 'best_tpm_prediction_model.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    joblib.dump(tpm_threshold, 'tpm_threshold.pkl')

    print(f"\nModel saved as 'best_tpm_prediction_model.pkl'")
    print(f"TPM threshold for 'high': {tpm_threshold:.0f}")

    return trained_models, feature_columns, tpm_threshold


# 8. Function để sử dụng model đã train
def load_and_predict(new_data_file):
    """Load model đã train và predict cho dữ liệu mới"""
    import joblib

    # Load model và các thông tin cần thiết
    model_info = joblib.load('best_tpm_prediction_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    tpm_threshold = joblib.load('tpm_threshold.pkl')

    # Load dữ liệu mới
    new_data = load_and_preprocess_data(new_data_file)

    # Predict
    predictions = predict_high_tpm_periods(
        model_info['model'],
        model_info['scaler'],
        new_data,
        tpm_threshold
    )

    # Tìm các thời điểm có khả năng cao tpm sẽ tăng
    high_prob_periods = predictions[predictions['probability_high_tpm'] > 0.7]

    print("Thời điểm dự đoán TPM cao (probability > 70%):")
    print(high_prob_periods[['timestamp', 'probability_high_tpm', 'actual_tpm']])

    return predictions


# Chạy training
if __name__ == "__main__":
    trained_models, feature_columns, tpm_threshold = main()


def predict_current_period(hour, minute, day_of_week, current_tpm, 
                          previous_tpm_values, play_counted, is_push_notification=0):
    """
    Dự đoán TPM cho thời điểm hiện tại
    
    Parameters:
    - hour: Giờ hiện tại (0-23)
    - minute: Phút hiện tại (0-59)  
    - day_of_week: Thứ trong tuần (0=Monday, 6=Sunday)
    - current_tpm: TPM hiện tại
    - previous_tpm_values: List các giá trị TPM trước đó [tpm_1_period_ago, tpm_2_periods_ago, tpm_3_periods_ago]
    - play_counted: Số lượt chơi hiện tại
    - is_push_notification: Có push notification không (0 hoặc 1)
    """
    import joblib
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Load model và thông tin cần thiết
    model_info = joblib.load('best_tpm_prediction_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    tpm_threshold = joblib.load('tpm_threshold.pkl')
    
    # Tạo DataFrame với dữ liệu input
    current_time = datetime.now().replace(hour=hour, minute=minute)
    
    data = {
        'timestamp': [current_time],
        'tpm': [current_tpm],
        'play_counted': [play_counted],
        'is_push_notification': [is_push_notification]
    }
    
    df = pd.DataFrame(data)
    
    # Tạo features thủ công
    df['hour'] = hour
    df['minute'] = minute
    df['day_of_week'] = day_of_week
    df['is_weekend'] = 1 if day_of_week >= 5 else 0
    df['time_of_day'] = hour + minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Peak hour features
    df['is_peak_hour'] = 1 if (0 <= hour <= 1) or (9 <= hour <= 10) or (12 <= hour <= 13) or (19 <= hour <= 20) else 0
    
    # Lag features
    df['tpm_lag_1'] = previous_tpm_values[0] if len(previous_tpm_values) > 0 else current_tpm
    df['tpm_lag_2'] = previous_tpm_values[1] if len(previous_tpm_values) > 1 else current_tpm
    df['tpm_lag_3'] = previous_tpm_values[2] if len(previous_tpm_values) > 2 else current_tpm
    
    # Rolling means (simplified)
    recent_tpm = [current_tpm] + previous_tpm_values[:5]
    df['tpm_rolling_mean_6'] = np.mean(recent_tpm[:6])
    df['tpm_rolling_mean_12'] = np.mean(recent_tpm[:12]) if len(recent_tpm) >= 12 else np.mean(recent_tpm)
    
    # Trend
    df['tpm_trend'] = current_tpm - df['tpm_lag_1']
    
    # Play counted rate (simplified)
    df['play_counted_rate'] = 0  # Cần thêm logic nếu có dữ liệu trước đó
    
    # Push notification effect
    df['minutes_since_push'] = 0  # Cần logic tính toán
    df['is_post_push_period'] = is_push_notification
    
    # Chuẩn bị features cho prediction
    X_features = []
    for col in feature_columns:
        if col in df.columns:
            X_features.append(df[col].iloc[0])
        else:
            X_features.append(0)  # Default value cho missing features
    
    X_new = np.array(X_features).reshape(1, -1)
    
    # Predict
    model = model_info['model']
    scaler = model_info['scaler']
    
    if scaler is not None:
        X_new = scaler.transform(X_new)
    
    probability = model.predict_proba(X_new)[0, 1]
    prediction = model.predict(X_new)[0]
    
    result = {
        'predicted_high_tpm': bool(prediction),
        'probability_high_tpm': probability,
        'tpm_threshold': tpm_threshold,
        'recommendation': 'TPM sẽ cao' if probability > 0.7 else 'TPM bình thường'
    }
    
    return result

# Ví dụ sử dụng:
# Dự đoán cho 2PM, thứ 2, TPM hiện tại = 450, có dữ liệu TPM trước đó
prediction = predict_current_period(
    hour=14, 
    minute=0, 
    day_of_week=0,  # Monday
    current_tpm=450,
    previous_tpm_values=[420, 380, 350],  # TPM của 3 periods trước
    play_counted=1200,
    is_push_notification=0
)

print(f"Dự đoán: {prediction['recommendation']}")
print(f"Xác suất TPM cao: {prediction['probability_high_tpm']:.2%}")