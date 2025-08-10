import os
import sys
import argparse
import glob
import joblib
from datetime import datetime
import pytz

# Ensure relative imports work when running from project root or Src/Day5
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from advanced_traffic_predictor import AdvancedTrafficPredictor
from newrelic_traffic_collector import get_recent_3h_1min_dataframe


def predict_from_inputs(
    when,
    current_tpm: float,
    previous_tpms: list = None,
    response_time: float = None,
    push_notification_active: int = 0,
    minutes_since_push: int = 999999,
    minutes_ahead=(5, 10, 15),
    extra_features: dict = None,
    models_dir: str = None,
    timestamp: str = None,
):
    """
    Dự đoán TPM cho các horizon (5/10/15 phút) từ input tuỳ chỉnh, sử dụng
    các model đã được train và lưu trước đó (không cần train lại).

    Tham số:
    - when: datetime (naive hoặc timezone-aware)
    - current_tpm: TPM tại thời điểm 'when'
    - previous_tpms: list TPM quá khứ, phần tử 0 là 1 phút trước, [1] là 2 phút trước, ...
    - response_time: response time hiện tại (nếu có)
    - push_notification_active: cờ 0/1 cho biết có push tại thời điểm 'when'
    - minutes_since_push: số phút kể từ lần push gần nhất
    - minutes_ahead: tuple/list các horizon cần dự đoán (mặc định 5/10/15)
    - extra_features: dict các feature bổ sung nếu model có
    - models_dir: thư mục chứa models (mặc định Src/Day5/models)
    - timestamp: timestamp model cụ thể để load; nếu None sẽ tự lấy bản mới nhất

    Trả về:
    - dict: { 'tpm_5min': float, 'tpm_10min': float, 'tpm_15min': float }
    """
    # Xác định thư mục models mặc định theo cấu trúc project
    if models_dir is None:
        models_dir = os.path.join(CURRENT_DIR, 'models')

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}. Hãy chạy training để tạo models trước.")

    # Khởi tạo predictor và nạp models theo timestamp mới nhất (hoặc theo tham số)
    predictor = AdvancedTrafficPredictor()

    # Tìm timestamp mới nhất nếu chưa cung cấp
    if not timestamp:
        ts = find_latest_timestamp(models_dir)
        if not ts:
            raise FileNotFoundError("No saved models found. Please run training to create models.")
        timestamp = ts

    ok = load_models_into_predictor(predictor, models_dir, timestamp)
    if not ok:
        raise RuntimeError("Failed to load models/scalers/metadata. Cannot perform prediction.")

    # Gọi vào hàm suy luận điểm-thời-gian của predictor
    return predictor.predict_from_inputs(
        when=when,
        current_tpm=current_tpm,
        previous_tpms=previous_tpms,
        response_time=response_time,
        push_notification_active=push_notification_active,
        minutes_since_push=minutes_since_push,
        minutes_ahead=minutes_ahead,
        extra_features=extra_features,
    )


def find_latest_timestamp(models_dir: str) -> str:
    """Find the latest metadata_<timestamp>.joblib and return its timestamp string."""
    pattern = os.path.join(models_dir, 'metadata_*.joblib')
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by timestamp parsed from filename or by mtime as fallback
    def extract_ts(path):
        base = os.path.basename(path)
        ts = base.replace('metadata_', '').replace('.joblib', '')
        return ts
    files_sorted = sorted(files, key=lambda p: extract_ts(p))
    return extract_ts(files_sorted[-1])


def load_models_into_predictor(predictor: AdvancedTrafficPredictor, models_dir: str, timestamp: str) -> bool:
    """Load best models, scalers, and metadata into the predictor for a given timestamp.
    Returns True if successful, False otherwise.
    """
    metadata_path = os.path.join(models_dir, f'metadata_{timestamp}.joblib')
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return False

    metadata = joblib.load(metadata_path)
    best_models = metadata.get('best_models', {})
    feature_columns = metadata.get('feature_columns')
    target_columns = metadata.get('target_columns')

    if not best_models:
        print("❌ No best_models found in metadata; cannot load models.")
        return False

    # Initialize containers matching training structure
    predictor.models = {t: {} for t in target_columns}
    predictor.scalers = {}
    predictor.best_models = best_models
    predictor.model_performance = metadata.get('model_performance', {})

    # If feature/target columns differ, update to match saved artifacts
    if feature_columns:
        predictor.feature_columns = feature_columns
    if target_columns:
        predictor.target_columns = target_columns

    # Load best model per target and its scaler
    for target, best_name in best_models.items():
        model_path = os.path.join(models_dir, f'{target}_{best_name}_{timestamp}.joblib')
        if not os.path.exists(model_path):
            print(f"❌ Missing model file: {model_path}")
            return False
        predictor.models[target][best_name] = joblib.load(model_path)

        # Load scaler for the target (always saved, used particularly for SVR)
        scaler_path = os.path.join(models_dir, f'scaler_{target}_{timestamp}.joblib')
        if not os.path.exists(scaler_path):
            print(f"⚠️ Scaler not found for {target}: {scaler_path}. Proceeding without scaler.")
        else:
            predictor.scalers[target] = joblib.load(scaler_path)

    print(f"✅ Loaded models and metadata from timestamp {timestamp}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run demo predictions without retraining (using live New Relic data)')
    parser.add_argument('--hours-ahead', type=int, default=6, help='Hours ahead to scan for high TPM periods')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset CSV as a fallback if API fails')
    parser.add_argument('--timestamp', type=str, default=None, help='Specific model timestamp to load (e.g., 20250809_224101)')
    parser.add_argument('--api-key', type=str, default=os.environ.get('NEWRELIC_API_KEY') or os.environ.get('NEW_RELIC_API_KEY'), help='New Relic API key (default from env NEWRELIC_API_KEY)')
    parser.add_argument('--app-id', type=str, default=os.environ.get('NEWRELIC_APP_ID') or os.environ.get('NEW_RELIC_APP_ID'), help='New Relic Application ID (default from env NEWRELIC_APP_ID)')
    parser.add_argument('--metric', type=str, default='HttpDispatcher', help='Metric name to query (default: HttpDispatcher)')
    args = parser.parse_args()

    models_dir = os.path.join(CURRENT_DIR, 'models')
    if not os.path.isdir(models_dir):
        print(f"❌ Models directory not found: {models_dir}\nPlease run the training pipeline first to generate models.")
        sys.exit(1)

    # Determine timestamp to load
    timestamp = args.timestamp or find_latest_timestamp(models_dir)
    if not timestamp:
        print("❌ No saved models found. Please run training to create models.")
        sys.exit(1)

    # Initialize predictor
    predictor = AdvancedTrafficPredictor()

    # Prefer live New Relic data (last 3 hours, 1-min granularity)
    print("🌐 Fetching recent 3h data (1-min) from New Relic API...")
    live_df = get_recent_3h_1min_dataframe(api_key=args.api_key, app_id=args.app_id, metric_name=args.metric)

    if live_df is None or live_df.empty:
        if args.dataset:
            print("⚠️ API data unavailable. Falling back to dataset CSV provided via --dataset.")
            predictor.data_file_path = args.dataset
            df = predictor.load_and_preprocess_data()
        else:
            print("❌ No live data fetched and no dataset fallback provided. Aborting.")
            sys.exit(1)
    else:
        # Create features from live data for prediction (keep latest rows even if some targets are NaN)
        df = predictor.create_features(live_df)

    # Load models into predictor
    ok = load_models_into_predictor(predictor, models_dir, timestamp)
    if not ok:
        print("❌ Failed to load models. Aborting.")
        sys.exit(1)

    # Run the demo predictions without retraining
    print("\n🚀 Running independent demo predictions...")
    predictor.demonstrate_predictions(df, hours_ahead=args.hours_ahead)


def main2():
    # Lấy ngày hôm nay 09:00 tại GMT+7
    tz = pytz.timezone('Asia/Bangkok')
    date_str = '2025-08-15 09:00'
    naive_dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
    test_when = tz.localize(naive_dt)

    # Giá trị mẫu để test
    current_tpm = 200.0
    previous_tpms = [190.0, 100.0, 123.0, 124.0, 130.0]  # 1→5 phút trước
    response_time = 120.0

    # Mô phỏng vừa push lúc 09:00
    push_notification_active = 1
    minutes_since_push = 0

    print("\n🧪 Testing predict_from_inputs at 09:00 GMT+7 ...")
    preds = predict_from_inputs(
        when=test_when,
        current_tpm=current_tpm,
        previous_tpms=previous_tpms,
        response_time=response_time,
        push_notification_active=push_notification_active,
        minutes_since_push=minutes_since_push,
        minutes_ahead=(5, 10, 15),
        # models_dir=None, timestamp=None  # dùng mặc định: lấy model mới nhất từ thư mục models
    )
    print(f"🧪 Predictions at {test_when.strftime('%Y-%m-%d %H:%M %Z')}: {preds}")

if __name__ == '__main__':
    main2()
