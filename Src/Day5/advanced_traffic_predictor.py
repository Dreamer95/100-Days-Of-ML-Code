import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class AdvancedTrafficPredictor:
    def __init__(self):
        """Initialize the Advanced Traffic Predictor"""
        print("🚀 Initializing Advanced Traffic Predictor...")

        # Model configuration
        self.models = {}
        self.scalers = {}
        self.best_models = {}
        self.model_performance = {}
        self.tpm_thresholds = None
        self.tpm_thresholds_stats = None

        # Data file path
        self.data_file_path = "/Users/dinhngocdong/Documents/learning/100-Days-Of-ML-Code/Src/Day5/datasets/newrelic_weekend_traffic_enhanced.csv"

        # Feature and target columns
        self.feature_columns = [
            'hour', 'minute', 'day_of_week', 'is_weekend',
            'tpm_lag_1', 'tpm_lag_5', 'tpm_lag_10', 'tpm_lag_15',
            'tpm_rolling_mean_5', 'tpm_rolling_mean_15', 'tpm_rolling_mean_30',
            'tpm_rolling_std_5', 'tpm_rolling_std_15',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
            'day_sin', 'day_cos',
            'push_notification_active',
            'minutes_since_push',
        ]

        self.target_columns = ['tpm_5min', 'tpm_10min', 'tpm_15min']

        # Model algorithms to try
        self.algorithms = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }

        # Python
        # self.algorithms = {
        #     'RandomForest': RandomForestRegressor(
        #         n_estimators=500,
        #         max_depth=16,
        #         min_samples_leaf=4,
        #         min_samples_split=8,
        #         max_features='sqrt',
        #         bootstrap=True,
        #         oob_score=True,
        #         random_state=42,
        #         n_jobs=-1
        #     ),
        #     'GradientBoosting': GradientBoostingRegressor(
        #         loss='huber',  # hoặc 'absolute_error'
        #         alpha=0.9,  # chỉ dùng khi loss='huber' hoặc 'quantile'
        #         learning_rate=0.05,
        #         n_estimators=600,
        #         max_depth=3,
        #         subsample=0.8,
        #         max_features=None,
        #         min_samples_leaf=10,
        #         random_state=42,
        #         validation_fraction=0.1,
        #         n_iter_no_change=20,
        #         tol=1e-4
        #     ),
        #     'LinearRegression': LinearRegression(
        #         fit_intercept=True
        #     ),
        #     'SVR': SVR(
        #         kernel='rbf',
        #         C=10.0,  # bắt đầu nhỏ hơn, rồi tune
        #         gamma='scale',  # để thích nghi theo variance của dữ liệu đã scale
        #         epsilon=0.2
        #     ),
        #     # (Tùy chọn) Thêm vài mô hình robust/regularized để so sánh:
        #     # 'Ridge': Ridge(alpha=1.0, random_state=42),
        #     # 'Huber': HuberRegressor(epsilon=1.5, alpha=1e-3)
        # }

        print("✅ Predictor initialized successfully!")

    def compute_tpm_thresholds_from_df(self, df):
        """
        Tính các ngưỡng quantile để phân loại TPM theo yêu cầu
        và in ra phân phối số lượng bản ghi theo từng nhóm.

        - 0-60%: low
        - 60-80%: medium
        - 80-90%: high
        - 90-100%: very_high
        """
        series = df['tpm'].dropna()
        total = int(len(series))
        if total == 0:
            print("⚠️ No TPM data available to compute thresholds.")
            self.tpm_thresholds_stats = None
            return {'q60': np.nan, 'q80': np.nan, 'q90': np.nan}

        q60 = float(series.quantile(0.60))
        q80 = float(series.quantile(0.80))
        q90 = float(series.quantile(0.90))

        thresholds = {'q60': q60, 'q80': q80, 'q90': q90}

        # Phân loại theo bins dựa trên thresholds
        labels = ['low', 'medium', 'high', 'very_high']
        bins = [-np.inf, q60, q80, q90, np.inf]

        cat = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
        counts = cat.value_counts().reindex(labels, fill_value=0).astype(int)
        percents = (counts / total * 100).round(2)

        # Range hiển thị cho từng nhóm
        s_min = float(series.min())
        s_max = float(series.max())
        ranges = {
            'low': f"[{s_min:.2f}, {q60:.2f}]",
            'medium': f"({q60:.2f}, {q80:.2f}]",
            'high': f"({q80:.2f}, {q90:.2f}]",
            'very_high': f"({q90:.2f}, {s_max:.2f}]",
        }

        # Lưu stats vào class để dùng sau
        stats = {
            k: {'count': int(counts[k]), 'percent': float(percents[k]), 'range': ranges[k]}
            for k in labels
        }
        self.tpm_thresholds_stats = {
            'total_records': total,
            'thresholds': thresholds,
            'distribution': stats
        }

        # In ra tổng hợp để đánh giá
        print("\n📊 TPM Thresholds (training-based):")
        print(f"  - q60: {q60:.2f}, q80: {q80:.2f}, q90: {q90:.2f}")
        print(f"  - Total training records: {total}")
        print("  - Distribution by category:")
        for k in labels:
            info = stats[k]
            print(f"    • {k:10s} | {info['count']:6d} recs "
                  f"({info['percent']:6.2f}%) | range {info['range']}")

        return thresholds

    def classify_tpm_value(self, tpm, thresholds=None):
        """
        Phân loại một giá trị TPM theo thresholds đã lưu trong class.
        Nếu truyền thresholds thì dùng thresholds đó; ngược lại dùng self.tpm_thresholds.
        """
        th = thresholds or self.tpm_thresholds
        if th is None:
            return 'unknown'
        if tpm <= th['q60']:
            return 'low'
        if tpm <= th['q80']:
            return 'medium'
        if tpm <= th['q90']:
            return 'high'
        return 'very_high'

    def load_real_data(self):
        """Load data from the New Relic CSV file"""
        print(f"📊 Loading data from {self.data_file_path}...")

        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}")

        # Load the CSV file
        df = pd.read_csv(self.data_file_path)
        print(f"✅ Loaded {len(df):,} records from CSV")

        # Display basic info about the data
        print(f"📋 Data columns: {list(df.columns)}")
        print(f"📋 Data shape: {df.shape}")
        print(f"📋 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Determine the TPM column name (could be requests_per_minute, tpm, etc.)
        tpm_column = None
        possible_tpm_columns = ['tpm', 'requests_per_minute', 'rpm', 'throughput']

        for col in possible_tpm_columns:
            if col in df.columns:
                tpm_column = col
                break

        if tpm_column is None:
            # Look for any column that might contain traffic data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            likely_tpm_columns = [col for col in numeric_columns if any(keyword in col.lower()
                                                                        for keyword in
                                                                        ['request', 'tpm', 'traffic', 'minute', 'rpm'])]
            if likely_tpm_columns:
                tpm_column = likely_tpm_columns[0]
                print(f"⚠️ Using column '{tpm_column}' as TPM data")
            else:
                raise ValueError("Could not find TPM/requests column in the data")

        # Rename to standardized 'tpm' column
        if tpm_column != 'tpm':
            df = df.rename(columns={tpm_column: 'tpm'})
            print(f"📝 Renamed column '{tpm_column}' to 'tpm'")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Handle missing values
        df['tpm'] = df['tpm'].fillna(df['tpm'].median())

        # Basic data validation
        if df['tpm'].min() < 0:
            print("⚠️ Found negative TPM values, setting them to 0")
            df['tpm'] = df['tpm'].clip(lower=0)

        # Display summary statistics
        print(f"\n📊 TPM Statistics:")
        print(f"   • Min: {df['tpm'].min():.2f}")
        print(f"   • Max: {df['tpm'].max():.2f}")
        print(f"   • Mean: {df['tpm'].mean():.2f}")
        print(f"   • Median: {df['tpm'].median():.2f}")
        print(f"   • Std: {df['tpm'].std():.2f}")

        return df

    def create_features(self, df):
        """Create features from timestamp and historical data"""
        print("🔧 Creating features...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Lag features
        df['tpm_lag_1'] = df['tpm'].shift(1)
        df['tpm_lag_5'] = df['tpm'].shift(5)
        df['tpm_lag_10'] = df['tpm'].shift(10)
        df['tpm_lag_15'] = df['tpm'].shift(15)

        # Rolling statistics
        df['tpm_rolling_mean_5'] = df['tpm'].rolling(window=5, min_periods=1).mean()
        df['tpm_rolling_mean_15'] = df['tpm'].rolling(window=15, min_periods=1).mean()
        df['tpm_rolling_mean_30'] = df['tpm'].rolling(window=30, min_periods=1).mean()

        df['tpm_rolling_std_5'] = df['tpm'].rolling(window=5, min_periods=1).std()
        df['tpm_rolling_std_15'] = df['tpm'].rolling(window=15, min_periods=1).std()

        # Handle NaN values in rolling statistics
        df['tpm_rolling_std_5'] = df['tpm_rolling_std_5'].fillna(0)
        df['tpm_rolling_std_15'] = df['tpm_rolling_std_15'].fillna(0)

        # Cyclic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Push-notification features (read from CSV if present, else default)
        if 'push_notification_active' in df.columns:
            df['push_notification_active'] = df['push_notification_active'].fillna(0).astype(int)
        else:
            df['push_notification_active'] = 0

        if 'minutes_since_push' in df.columns:
            df['minutes_since_push'] = df['minutes_since_push'].fillna(999999)
            # ensure non-negative
            df['minutes_since_push'] = df['minutes_since_push'].clip(lower=0)
        else:
            df['minutes_since_push'] = 999999

        # Optional: a decayed strength feature to help model learn push impact
        # Strongest right after push and decays over ~15 minutes, 0 afterward
        df['push_effect_strength'] = np.where(
            df['minutes_since_push'] <= 15,
            np.exp(-df['minutes_since_push'] / 10.0),
            0.0
        )

        # Target variables (future TPM values)
        df['tpm_5min'] = df['tpm'].shift(-5)
        df['tpm_10min'] = df['tpm'].shift(-10)
        df['tpm_15min'] = df['tpm'].shift(-15)

        print("✅ Features created successfully!")
        return df

    def load_and_preprocess_data(self):
        """Load and preprocess the data from New Relic CSV file"""
        print("📈 Loading and preprocessing real New Relic data...")

        # Load real data from CSV
        df = self.load_real_data()

        # Create features
        df = self.create_features(df)

        # Remove rows with NaN values (due to shifting and initial lag features)
        initial_length = len(df)
        df = df.dropna()
        final_length = len(df)

        print(f"📊 Data preprocessing complete:")
        print(f"   • Initial rows: {initial_length:,}")
        print(f"   • Final rows: {final_length:,}")
        print(f"   • Features: {len(self.feature_columns)}")
        print(f"   • Targets: {len(self.target_columns)}")

        if final_length < 100:
            print("⚠️ Warning: Very few data points remaining after preprocessing. Consider adjusting lag features.")

        # TÍNH VÀ LƯU THRESHOLDS cho phân loại từ dữ liệu huấn luyện
        try:
            self.tpm_thresholds = self.compute_tpm_thresholds_from_df(df)
            print(f"✅ Stored TPM thresholds for classification: {self.tpm_thresholds}")
        except Exception as e:
            print(f"⚠️ Could not compute TPM thresholds: {e}")
            self.tpm_thresholds = None

        return df

    def train_models(self, df):
        """Train models for each target variable"""
        print("🤖 Training models...")

        # Prepare features
        X = df[self.feature_columns].fillna(0)

        # Initialize storage
        self.models = {target: {} for target in self.target_columns}
        self.scalers = {target: StandardScaler() for target in self.target_columns}
        self.model_performance = {target: {} for target in self.target_columns}
        self.best_models = {}

        # Train models for each target
        for target in self.target_columns:
            print(f"\n🎯 Training models for {target}...")

            # Prepare target variable
            y = df[target].fillna(df['tpm'])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features for SVR
            X_train_scaled = self.scalers[target].fit_transform(X_train)
            X_test_scaled = self.scalers[target].transform(X_test)

            best_score = -np.inf

            # Train each algorithm
            for name, algorithm in self.algorithms.items():
                print(f"   Training {name}...")

                try:
                    # Use scaled features for SVR, original for others
                    if name == 'SVR':
                        model = algorithm.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model = algorithm.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Store model and performance
                    self.models[target][name] = model
                    self.model_performance[target][name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }

                    # Track best model
                    if r2 > best_score:
                        best_score = r2
                        self.best_models[target] = name

                    print(f"      ✅ {name}: R² = {r2:.3f}, RMSE = {rmse:.2f}")

                except Exception as e:
                    print(f"      ❌ {name}: Failed - {str(e)}")

            print(f"   🏆 Best model for {target}: {self.best_models[target]} (R² = {best_score:.3f})")

        print("\n✅ Model training completed!")

    def save_models(self):
        """Save trained models"""
        print("💾 Saving models...")

        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for target in self.target_columns:
            for model_name, model in self.models[target].items():
                filename = f"models/{target}_{model_name}_{timestamp}.joblib"
                joblib.dump(model, filename)

        # Save scalers
        for target, scaler in self.scalers.items():
            filename = f"models/scaler_{target}_{timestamp}.joblib"
            joblib.dump(scaler, filename)

        # Save metadata
        metadata = {
            'best_models': self.best_models,
            'model_performance': self.model_performance,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'timestamp': timestamp,
            'data_source': 'New Relic Real Data'
        }

        joblib.dump(metadata, f"models/metadata_{timestamp}.joblib")

        print(f"✅ Models saved with timestamp: {timestamp}")

    def visualize_model_performance_on_training_data(self, df):
        """
        Vẽ charts để đánh giá độ hiệu quả của model bằng cách so sánh
        giá trị dự đoán và giá trị thực tế trên training data
        """
        print("📊 Creating model performance evaluation charts...")

        if not self.models or not self.best_models:
            print("❌ No trained models found. Please train models first.")
            return

        # Prepare data for evaluation
        X = df[self.feature_columns].fillna(0)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))

        # Create a grid layout: 3 rows x 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Colors for different targets
        colors = {'tpm_5min': 'blue', 'tpm_10min': 'green', 'tpm_15min': 'red'}

        for i, target in enumerate(self.target_columns):
            print(f"📈 Evaluating {target}...")

            # Get actual values (remove NaN)
            y_actual = df[target].fillna(df['tpm'])

            # Get best model for this target
            best_model_name = self.best_models[target]
            model = self.models[target][best_model_name]

            # Make predictions
            if best_model_name in ['SVR']:
                X_scaled = self.scalers[target].transform(X)
                y_pred = model.predict(X_scaled)
            else:
                y_pred = model.predict(X)

            # Calculate metrics
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)

            row = i  # Each target gets a full row

            # Plot 1: Actual vs Predicted Scatter Plot
            ax1 = fig.add_subplot(gs[row, 0])

            # Sample data for better visualization if dataset is large
            if len(y_actual) > 5000:
                sample_idx = np.random.choice(len(y_actual), 5000, replace=False)
                y_actual_sample = y_actual.iloc[sample_idx]
                y_pred_sample = y_pred[sample_idx]
            else:
                y_actual_sample = y_actual
                y_pred_sample = y_pred

            ax1.scatter(y_actual_sample, y_pred_sample, alpha=0.5, color=colors[target], s=20)

            # Perfect prediction line (y = x)
            min_val = min(y_actual_sample.min(), y_pred_sample.min())
            max_val = max(y_actual_sample.max(), y_pred_sample.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

            ax1.set_xlabel('Actual TPM')
            ax1.set_ylabel('Predicted TPM')
            ax1.set_title(f'{target} - Actual vs Predicted\n'
                          f'Model: {best_model_name}\n'
                          f'R² = {r2:.3f}, RMSE = {rmse:.1f}')
            ax1.grid(True, alpha=0.3)

            # Add correlation coefficient
            correlation = np.corrcoef(y_actual_sample, y_pred_sample)[0, 1]
            ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                     transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot 2: Residuals Plot
            ax2 = fig.add_subplot(gs[row, 1])
            residuals = y_actual - y_pred

            if len(residuals) > 5000:
                residuals_sample = residuals.iloc[sample_idx]
                y_pred_sample_for_residuals = y_pred[sample_idx]
            else:
                residuals_sample = residuals
                y_pred_sample_for_residuals = y_pred

            ax2.scatter(y_pred_sample_for_residuals, residuals_sample,
                        alpha=0.5, color=colors[target], s=20)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax2.set_xlabel('Predicted TPM')
            ax2.set_ylabel('Residuals (Actual - Predicted)')
            ax2.set_title(f'{target} - Residuals Plot\n'
                          f'MAE = {mae:.1f}')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Time Series Comparison (last 500 points)
            ax3 = fig.add_subplot(gs[row, 2])

            # Take last 500 points for time series visualization
            last_n = min(500, len(y_actual))
            time_idx = range(len(y_actual) - last_n, len(y_actual))

            ax3.plot(time_idx, y_actual.iloc[-last_n:],
                     label='Actual', alpha=0.7, linewidth=1.5, color='black')
            ax3.plot(time_idx, y_pred[-last_n:],
                     label='Predicted', alpha=0.7, linewidth=1.5, color=colors[target])

            ax3.set_xlabel('Time Index')
            ax3.set_ylabel('TPM')
            ax3.set_title(f'{target} - Time Series Comparison\n'
                          f'(Last {last_n} points)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Highlight significant differences
            diff = np.abs(y_actual.iloc[-last_n:] - y_pred[-last_n:])
            high_error_threshold = np.percentile(diff, 90)  # Top 10% errors
            high_error_idx = np.where(diff > high_error_threshold)[0]

            if len(high_error_idx) > 0:
                ax3.scatter(np.array(time_idx)[high_error_idx],
                            y_actual.iloc[-last_n:].iloc[high_error_idx],
                            color='red', s=50, marker='x', alpha=0.8,
                            label='High Error Points')
                ax3.legend()

        # Add overall performance summary
        fig.suptitle('Model Performance Evaluation - New Relic Real Data',
                     fontsize=20, fontweight='bold', y=0.98)

        # Add performance summary text
        summary_text = "📊 Performance Summary (Real Data):\n"
        for target in self.target_columns:
            best_model = self.best_models[target]
            perf = self.model_performance[target][best_model]
            summary_text += f"{target}: {best_model} (R²={perf['r2']:.3f}, RMSE={perf['rmse']:.1f})\n"

        fig.text(0.02, 0.02, summary_text, fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                 verticalalignment='bottom')

        # Save plot
        os.makedirs('charts', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'charts/model_performance_evaluation_real_data_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        # Print detailed performance analysis
        print(f"\n{'=' * 60}")
        print("🎯 DETAILED PERFORMANCE ANALYSIS (REAL DATA)")
        print(f"{'=' * 60}")

        for target in self.target_columns:
            print(f"\n📊 {target.upper()}")
            print("-" * 40)

            best_model_name = self.best_models[target]

            for model_name, performance in self.model_performance[target].items():
                status = "🏆 BEST" if model_name == best_model_name else "  "
                print(f"{status} {model_name:15} | "
                      f"R²: {performance['r2']:6.3f} | "
                      f"RMSE: {performance['rmse']:7.1f} | "
                      f"MAE: {performance['mae']:6.1f}")

        # Performance interpretation
        print(f"\n{'=' * 60}")
        print("🔍 PERFORMANCE INTERPRETATION (REAL DATA)")
        print(f"{'=' * 60}")

        overall_performance = []
        for target in self.target_columns:
            best_model_name = self.best_models[target]
            r2_score_value = self.model_performance[target][best_model_name]['r2']
            overall_performance.append(r2_score_value)

            if r2_score_value >= 0.8:
                interpretation = "Excellent - Model explains >80% of variance"
            elif r2_score_value >= 0.6:
                interpretation = "Good - Model explains >60% of variance"
            elif r2_score_value >= 0.4:
                interpretation = "Moderate - Model explains >40% of variance"
            elif r2_score_value >= 0.2:
                interpretation = "Poor - Model explains >20% of variance"
            else:
                interpretation = "Very Poor - Model explains <20% of variance"

            print(f"{target}: R² = {r2_score_value:.3f} - {interpretation}")

        avg_r2 = np.mean(overall_performance)
        print(f"\n🎯 Overall Model Performance: R² = {avg_r2:.3f}")

        if avg_r2 >= 0.7:
            print("✅ EXCELLENT: Models are performing very well!")
        elif avg_r2 >= 0.5:
            print("👍 GOOD: Models are performing adequately")
        elif avg_r2 >= 0.3:
            print("⚠️ MODERATE: Models need improvement")
        else:
            print("❌ POOR: Models need significant improvement")

    def plot_learning_curves(self, df):
        """Plot learning curves for each model"""
        print("📈 Creating learning curves...")

        if not self.models or not self.best_models:
            print("❌ No trained models found. Please train models first.")
            return

        # Prepare data
        X = df[self.feature_columns].fillna(0)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Learning Curves for Best Models (Real Data)', fontsize=16, fontweight='bold')

        for i, target in enumerate(self.target_columns):
            print(f"📊 Creating learning curve for {target}...")

            # Get target values
            y = df[target].fillna(df['tpm'])

            # Get best model
            best_model_name = self.best_models[target]
            model = self.algorithms[best_model_name]

            # Prepare data for learning curve
            if best_model_name == 'SVR':
                X_processed = self.scalers[target].fit_transform(X)
            else:
                X_processed = X

            # Calculate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_processed, y,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5, scoring='r2', n_jobs=-1
            )

            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Plot
            ax = axes[i]
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

            ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

            ax.set_title(f'{target}\n{best_model_name}')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('R² Score')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'charts/learning_curves_real_data_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Learning curves created successfully!")

    def visualize_model_comparison(self):
        """Create comprehensive model comparison visualizations"""
        print("📊 Creating model comparison visualizations...")

        if not self.model_performance:
            print("❌ No model performance data found. Please train models first.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison (Real Data)', fontsize=16, fontweight='bold')

        # Colors for models
        model_colors = {'RandomForest': 'green', 'GradientBoosting': 'blue',
                        'LinearRegression': 'orange', 'SVR': 'red'}

        # 1. R² Score Comparison
        ax1 = axes[0, 0]
        targets = []
        models_data = {model: [] for model in self.algorithms.keys()}

        for target in self.target_columns:
            targets.append(target.replace('tpm_', '').replace('min', ' min'))
            for model_name in self.algorithms.keys():
                if model_name in self.model_performance[target]:
                    models_data[model_name].append(self.model_performance[target][model_name]['r2'])
                else:
                    models_data[model_name].append(0)

        x = np.arange(len(targets))
        width = 0.2

        for i, (model_name, scores) in enumerate(models_data.items()):
            ax1.bar(x + i * width, scores, width, label=model_name, color=model_colors[model_name])

        ax1.set_xlabel('Target Variable')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(targets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. RMSE Comparison
        ax2 = axes[0, 1]
        models_rmse = {model: [] for model in self.algorithms.keys()}

        for target in self.target_columns:
            for model_name in self.algorithms.keys():
                if model_name in self.model_performance[target]:
                    models_rmse[model_name].append(self.model_performance[target][model_name]['rmse'])
                else:
                    models_rmse[model_name].append(0)

        for i, (model_name, rmse_values) in enumerate(models_rmse.items()):
            ax2.bar(x + i * width, rmse_values, width, label=model_name, color=model_colors[model_name])

        ax2.set_xlabel('Target Variable')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE Comparison (Lower is Better)')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(targets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Best Models Distribution
        ax3 = axes[1, 0]
        best_model_counts = {}
        for target in self.target_columns:
            best_model = self.best_models[target]
            best_model_counts[best_model] = best_model_counts.get(best_model, 0) + 1

        wedges, texts, autotexts = ax3.pie(best_model_counts.values(),
                                           labels=best_model_counts.keys(),
                                           autopct='%1.1f%%',
                                           colors=[model_colors[model] for model in best_model_counts.keys()])
        ax3.set_title('Best Models Distribution')

        # 4. Performance Summary Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')

        # Create performance summary
        summary_data = []
        for target in self.target_columns:
            best_model = self.best_models[target]
            perf = self.model_performance[target][best_model]
            summary_data.append([
                target.replace('tpm_', '').replace('min', ' min'),
                best_model,
                f"{perf['r2']:.3f}",
                f"{perf['rmse']:.1f}",
                f"{perf['mae']:.1f}"
            ])

        table = ax4.table(cellText=summary_data,
                          colLabels=['Target', 'Best Model', 'R²', 'RMSE', 'MAE'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Best Model Performance Summary', pad=20)

        # Add overall statistics
        overall_stats = []
        all_r2_scores = []
        for target in self.target_columns:
            best_model = self.best_models[target]
            all_r2_scores.append(self.model_performance[target][best_model]['r2'])

        avg_r2 = np.mean(all_r2_scores)
        overall_stats.append(['Average R²', f"{avg_r2:.3f}", '', '', ''])

        table2 = ax4.table(cellText=overall_stats,
                           colLabels=['Metric', 'Value', '', '', ''],
                           cellLoc='center',
                           loc='lower center',
                           bbox=[0, -0.3, 1, 0.2])
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'charts/model_comparison_real_data_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

        print("✅ Model comparison visualizations created successfully!")

    def visualize_predictions(self, df):
        """Visualize model predictions vs actual values"""
        print("🎯 Creating prediction visualizations...")

        if not self.models or not self.best_models:
            print("❌ No trained models found. Please train models first.")
            return

        # Prepare data
        X = df[self.feature_columns].fillna(0)

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Model Predictions vs Actual Values - Real Data (Last 500 Points)',
                     fontsize=16, fontweight='bold')

        colors = {'tpm_5min': 'blue', 'tpm_10min': 'green', 'tpm_15min': 'red'}

        # Plot last 500 points for each target
        last_n = min(500, len(df))
        time_indices = range(len(df) - last_n, len(df))

        for i, target in enumerate(self.target_columns):
            print(f"📊 Creating prediction plot for {target}...")

            # Get actual values
            y_actual = df[target].fillna(df['tpm']).iloc[-last_n:]

            # Get predictions from best model
            best_model_name = self.best_models[target]
            model = self.models[target][best_model_name]

            if best_model_name == 'SVR':
                X_scaled = self.scalers[target].transform(X)
                y_pred = model.predict(X_scaled)[-last_n:]
            else:
                y_pred = model.predict(X)[-last_n:]

            # Plot
            ax = axes[i]
            ax.plot(time_indices, y_actual, label='Actual',
                    color='black', alpha=0.7, linewidth=1.5)
            ax.plot(time_indices, y_pred, label=f'Predicted ({best_model_name})',
                    color=colors[target], alpha=0.7, linewidth=1.5)

            # Calculate and display metrics
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_actual, y_pred)

            ax.set_title(f'{target} - R²: {r2:.3f}, RMSE: {rmse:.1f}')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('TPM')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Highlight significant prediction errors
            errors = np.abs(y_actual - y_pred)
            error_threshold = np.percentile(errors, 95)  # Top 5% errors
            high_error_mask = errors > error_threshold

            if np.any(high_error_mask):
                error_indices = np.array(time_indices)[high_error_mask]
                error_values = y_actual[high_error_mask]
                ax.scatter(error_indices, error_values,
                           color='red', s=50, alpha=0.8, marker='x',
                           label='High Error Points')
                ax.legend()

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'charts/predictions_vs_actual_real_data_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Prediction visualizations created successfully!")

    def predict_tpm_ahead(self, recent_data, minutes_ahead=[5, 10, 15]):
        """Predict TPM for specific minutes ahead"""
        print(f"🔮 Predicting TPM for {minutes_ahead} minutes ahead...")

        if not self.models or not self.best_models:
            print("❌ No trained models found. Please train models first.")
            return {}

        predictions = {}

        # Get the last row for prediction
        last_row = recent_data.iloc[-1:][self.feature_columns].fillna(0)

        for minutes in minutes_ahead:
            target = f'tpm_{minutes}min'

            if target in self.best_models:
                best_model_name = self.best_models[target]
                model = self.models[target][best_model_name]
                print(f"🔮 Used model {best_model_name}")

                if best_model_name == 'SVR':
                    last_row_scaled = self.scalers[target].transform(last_row)
                    prediction = model.predict(last_row_scaled)[0]
                else:
                    prediction = model.predict(last_row)[0]

                predictions[target] = prediction

        return predictions

    def predict_high_tpm_periods(self, df, hours_ahead=6, high_tpm_threshold=None):
        """Predict when TPM will be unusually high in the next few hours"""
        print(f"🔥 Predicting high TPM periods for next {hours_ahead} hours...")

        if not self.models or not self.best_models:
            print("❌ No trained models found. Please train models first.")
            return []

        if high_tpm_threshold is None:
            high_tpm_threshold = df['tpm'].quantile(0.8)  # Top 20% as "high"

        # Generate future timestamps
        last_timestamp = df['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(minutes=1),
            periods=hours_ahead * 60,
            freq='1min'
        )

        # Create future dataframe with features
        future_data = []
        for timestamp in future_timestamps:
            row = {
                'timestamp': timestamp,
                'hour': timestamp.hour,
                'minute': timestamp.minute,
                'day_of_week': timestamp.weekday(),
                'is_weekend': int(timestamp.weekday() >= 5),
                'push_notification_active': 0,
                'minutes_since_push': 999999,
            }

            # Add cyclic features
            row['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            row['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            row['minute_sin'] = np.sin(2 * np.pi * timestamp.minute / 60)
            row['minute_cos'] = np.cos(2 * np.pi * timestamp.minute / 60)
            row['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
            row['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)

            # Use recent data for lag and rolling features (simplified)
            recent_tpm = df['tpm'].iloc[-60:].mean()  # Average of last hour
            row['tpm_lag_1'] = recent_tpm
            row['tpm_lag_5'] = recent_tpm
            row['tpm_lag_10'] = recent_tpm
            row['tpm_lag_15'] = recent_tpm
            row['tpm_rolling_mean_5'] = recent_tpm
            row['tpm_rolling_mean_15'] = recent_tpm
            row['tpm_rolling_mean_30'] = recent_tpm
            row['tpm_rolling_std_5'] = df['tpm'].iloc[-60:].std()
            row['tpm_rolling_std_15'] = df['tpm'].iloc[-60:].std()

            future_data.append(row)

        future_df = pd.DataFrame(future_data)

        # Make predictions using 5-minute model (most immediate)
        target = 'tpm_5min'
        if target not in self.best_models:
            print(f"❌ No model found for {target}")
            return []

        best_model_name = self.best_models[target]
        model = self.models[target][best_model_name]

        X_future = future_df[self.feature_columns].fillna(0)

        if best_model_name == 'SVR':
            X_future_scaled = self.scalers[target].transform(X_future)
            predictions = model.predict(X_future_scaled)
        else:
            predictions = model.predict(X_future)

        # Find high TPM periods
        high_periods = []
        for i, (timestamp, pred_tpm) in enumerate(zip(future_timestamps, predictions)):
            if pred_tpm > high_tpm_threshold:
                confidence = pred_tpm / high_tpm_threshold
                high_periods.append({
                    'timestamp': timestamp,
                    'predicted_tpm': pred_tpm,
                    'confidence': confidence
                })

        # Sort by confidence (highest first)
        high_periods.sort(key=lambda x: x['confidence'], reverse=True)

        print(f"🎯 Found {len(high_periods)} high TPM periods")
        return high_periods

    def demonstrate_predictions(self, df, hours_ahead=6):
        """Demonstrate making predictions without retraining.
        This extracts the inline demo logic from run_complete_analysis so it can be called independently.
        """
        # Demonstrate predictions
        print("\n🎯 Demonstration: Predicting TPM for next 5, 10, 15 minutes...")

        # Dùng thresholds từ training; nếu chưa có thì fallback tính tạm từ df truyền vào
        thresholds = self.tpm_thresholds
        if thresholds is None:
            try:
                thresholds = self.compute_tpm_thresholds_from_df(df)
                print(f"⚠️ Using ad-hoc thresholds computed from provided df: {thresholds}")
            except Exception as e:
                print(f"⚠️ Could not compute thresholds from provided df: {e}")
                thresholds = None

        # Use last 60 minutes of data
        recent_data = df.tail(60)
        predictions = self.predict_tpm_ahead(recent_data)

        current_tpm = recent_data['tpm'].iloc[-1]
        current_class = self.classify_tpm_value(current_tpm, thresholds)
        print(f"Current TPM: {current_tpm:.2f} ({current_class})")

        for target, pred in predictions.items():
            minutes = target.split('_')[1].replace('min', '')
            pred_class = self.classify_tpm_value(pred, thresholds)
            print(f"Predicted TPM in {minutes} minutes: {pred:.2f} ({pred_class})")

        # Predict high TPM periods (giữ nguyên)
        high_periods = self.predict_high_tpm_periods(df, hours_ahead=hours_ahead)

        if high_periods:
            print(f"\n🔥 Next high TPM periods (next {hours_ahead} hours):")
            for period in high_periods[:5]:  # Show first 5
                ph_class = self.classify_tpm_value(period['predicted_tpm'], thresholds)
                print(f"  {period['timestamp'].strftime('%H:%M')} - "
                      f"Predicted TPM: {period['predicted_tpm']:.0f} "
                      f"({ph_class}, Confidence: {period['confidence']:.1f}x)")
        else:
            print(f"\n✅ No high TPM periods predicted in the next {hours_ahead} hours")

        return {"predictions": predictions, "high_periods": high_periods}

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting complete traffic prediction analysis with REAL DATA...")

        # Load and preprocess data
        df = self.load_and_preprocess_data()

        # Train models
        self.train_models(df)

        # Save models
        self.save_models()

        # ===== COMPREHENSIVE MODEL EVALUATION =====
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE MODEL EVALUATION (REAL DATA)")
        print("=" * 60)

        self.visualize_model_performance_on_training_data(df)
        # self.plot_learning_curves(df)

        # Create visualizations
        self.visualize_model_comparison()
        self.visualize_predictions(df)

        # Demonstrate predictions
        self.demonstrate_predictions(df, hours_ahead=6)

        print("\n🎉 Analysis complete with REAL NEW RELIC DATA!")

    def predict_from_inputs(self,
                            when: datetime,
                            current_tpm: float,
                            previous_tpms: list = None,
                            response_time: float = None,
                            push_notification_active: int = 0,
                            minutes_since_push: int = 999999,
                            minutes_ahead=(5, 10, 15),
                            extra_features: dict = None):
        """
        Predict TPM for specific future horizons given a point-in-time context.
        Returns:
        - dict: { 'predictions': {...}, 'labels': {...}, 'thresholds': {... or None} }
        """
        if not self.models or not self.best_models:
            raise ValueError("Models are not loaded/trained.")

        # Chuẩn bị đặc trưng thời gian
        hour = when.hour
        minute = when.minute
        day_of_week = when.weekday()
        is_weekend = int(day_of_week >= 5)

        # Circular encodings
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        minute_sin = np.sin(2 * np.pi * minute / 60)
        minute_cos = np.cos(2 * np.pi * minute / 60)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        # Base feature dict (chỉ set những cột thực sự có trong feature_columns)
        row = {}
        base_vals = {
            'tpm': current_tpm,
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'minute_sin': minute_sin,
            'minute_cos': minute_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'push_notification_active': int(push_notification_active),
        }

        # Giới hạn minutes_since_push về khoảng hợp lý (nhất quán training)
        # Nếu > 15 phút thì push_effect_strength ~ 0
        ms_push = max(0, int(minutes_since_push))
        ms_push = min(ms_push, 10_000)  # tránh quá xa phân phối training
        base_vals['minutes_since_push'] = ms_push

        # Gắn các base feature nếu có trong danh sách features của model
        for k, v in base_vals.items():
            if k in self.feature_columns:
                row[k] = v

        if response_time is not None and 'response_time' in self.feature_columns:
            row['response_time'] = float(response_time)

        # Tính các lag/rolling từ previous_tpms + current_tpm để sát logic training
        # previous_tpms[0] = 1 phút trước → tạo chuỗi thời gian theo thứ tự cũ→mới
        hist = []
        if previous_tpms and len(previous_tpms) > 0:
            # previous_tpms: [1min_ago, 2min_ago, ...] → đảo để thành thời gian tăng dần
            hist = list(reversed(previous_tpms))
        hist = hist + [current_tpm]  # thêm điểm hiện tại

        # Map các lag nếu có cột tpm_lag_k
        # Với k phút trước: lấy từ previous_tpms[k-1] nếu có, else fallback current_tpm
        max_lag = max([int(c.split('_')[-1]) for c in self.feature_columns if c.startswith('tpm_lag_')], default=0)
        if max_lag > 0:
            for k in range(1, max_lag + 1):
                col = f'tpm_lag_{k}'
                if col in self.feature_columns:
                    val = previous_tpms[k - 1] if previous_tpms and len(previous_tpms) >= k else current_tpm
                    row[col] = float(val)

        # Tính rolling mean/std theo các cửa sổ có trong feature_columns từ chuỗi hist
        # Giống training: min_periods=1
        def rolling_mean(values, window):
            if len(values) == 0:
                return float(current_tpm)
            w = min(window, len(values))
            return float(np.mean(values[-w:]))

        def rolling_std(values, window):
            if len(values) <= 1:
                return 0.0
            w = min(window, len(values))
            return float(np.std(values[-w:], ddof=0))

        rolling_windows = {
            'tpm_rolling_mean_3': 3,
            'tpm_rolling_mean_5': 5,
            'tpm_rolling_mean_15': 15,
            'tpm_rolling_mean_30': 30,
            'tpm_rolling_std_3': 3,
            'tpm_rolling_std_5': 5,
            'tpm_rolling_std_15': 15,
            'tpm_rolling_std_30': 30,
            # Hỗ trợ các alias nếu training dùng 'tpm_ma_*' hoặc 'tpm_std_*'
            'tpm_ma_3': 3, 'tpm_ma_5': 5, 'tpm_ma_15': 15, 'tpm_ma_30': 30,
            'tpm_std_3': 3, 'tpm_std_5': 5, 'tpm_std_15': 15, 'tpm_std_30': 30,
        }
        for col, win in rolling_windows.items():
            if col in self.feature_columns:
                if 'mean' in col or '_ma_' in col:
                    row[col] = rolling_mean(hist, win)
                elif 'std' in col:
                    row[col] = rolling_std(hist, win)

        # push_effect_strength nếu có trong feature_columns
        if 'push_effect_strength' in self.feature_columns:
            row['push_effect_strength'] = float(np.exp(-ms_push / 10.0)) if ms_push <= 15 else 0.0

        # Thêm extra features nếu có và nằm trong feature_columns
        if extra_features:
            for k, v in extra_features.items():
                if k in self.feature_columns:
                    row[k] = v

        # Tạo DataFrame theo đúng thứ tự feature_columns, điền 0 nếu thiếu
        feature_vector = pd.DataFrame([{col: row.get(col, 0) for col in self.feature_columns}])

        # Dự báo
        predictions = {}
        for minutes in minutes_ahead:
            target = f'tpm_{minutes}min'
            if target not in self.best_models:
                continue
            best_model_name = self.best_models[target]
            model = self.models[target][best_model_name]
            if best_model_name == 'SVR' and target in self.scalers:
                X_scaled = self.scalers[target].transform(feature_vector)
                pred = model.predict(X_scaled)[0]
            else:
                pred = model.predict(feature_vector)[0]
            predictions[target] = float(max(0.0, pred))  # đảm bảo không âm

        # Phân loại theo thresholds
        thresholds = self.tpm_thresholds
        labels = {k: self.classify_tpm_value(v, thresholds) for k, v in predictions.items()}

        return {
            'predictions': predictions,
            'labels': labels,
            'thresholds': thresholds
        }


def main():
    """Main function to run the traffic prediction analysis"""
    predictor = AdvancedTrafficPredictor()
    predictor.run_complete_analysis()


if __name__ == "__main__":
    main()