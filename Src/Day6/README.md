# Day6 - Summary của những hàm trong Day5

## Tổng quan

Day6 là một tổng hợp và tối ưu hóa của tất cả các chức năng chính từ Day5, được tổ chức thành 4 module độc lập nhưng có thể tích hợp với nhau.

## 4 Chức năng chính

### 1. 📡 New Relic Data Collector (`newrelic_data_collector.py`)
**Mục đích**: Thu thập data từ New Relic từ thứ 2 đến chủ nhật theo GMT+7 với độ chi tiết tối đa

**Tính năng chính**:
- Thu thập data theo tuần đầy đủ (thứ 2 - chủ nhật)
- Tự động điều chỉnh granularity dựa trên độ tuổi của data
- Hỗ trợ timezone GMT+7 (Asia/Ho_Chi_Minh)
- Tạo push notification features tự động
- Xử lý rate limiting và error handling

**Cách sử dụng**:
```python
from newrelic_data_collector import collect_full_week_newrelic_data

# Thu thập 4 tuần data
data = collect_full_week_newrelic_data(
    weeks_back=4,
    filename="my_data.csv"
)
```

### 2. 📊 Data Visualizer (`visualizer.py`)
**Mục đích**: Tạo các biểu đồ visualization cho TPM data

**Tính năng chính**:
- TPM Area Chart với statistics
- TPM Heatmap theo giờ và ngày trong tuần
- Comparison charts (day of week, hour of day)
- Comprehensive visualization suite
- Tự động lưu charts với timestamp

**Cách sử dụng**:
```python
from visualizer import create_comprehensive_visualization_suite

# Tạo tất cả charts
figures = create_comprehensive_visualization_suite(
    data=your_data,
    output_dir="charts"
)
```

### 3. 🤖 Time Series Trainer (`time_series_trainer.py`)
**Mục đích**: Training model theo time series với enhanced push notification features

**Tính năng chính**:
- Advanced feature engineering cho time series
- Enhanced push notification features (daytime vs nighttime effects)
- Multiple optimized models (RandomForest, GradientBoosting, Ridge)
- Time-based cross validation
- Feature selection để tránh overfitting
- Model saving/loading với metadata

**Cách sử dụng**:
```python
from time_series_trainer import TimeSeriesTrainer

trainer = TimeSeriesTrainer()
results = trainer.train_time_series_model(
    data=your_data,
    target_col='tpm',
    test_size=0.25,
    max_features=20
)

# Lưu model
model_path, metadata_path = trainer.save_model()
```

### 4. 🎯 Demo Predictor (`demo_predictor.py`)
**Mục đích**: Demo predictions từ model đã training

**Tính năng chính**:
- Load trained models tự động
- Predict từ input parameters
- Predict từ TPM chronological list với push events
- Enhanced push effects cho predictions
- Multiple demo scenarios
- Support cho cả datetime và dict format push events

**Cách sử dụng**:
```python
from demo_predictor import DemoPredictor

predictor = DemoPredictor()
predictor.load_trained_model()

# Predict từ inputs
results = predictor.predict_from_inputs(
    when=datetime.now(),
    current_tpm=500.0,
    previous_tpms=[480, 460, 450],
    push_notification_active=1,
    minutes_since_push=0
)

# Predict từ TPM list
results = predictor.predict_from_tpm_list(
    current_time=datetime.now(),
    tpm_chronological_list=[400, 420, 450, 480, 500],
    push_events=[datetime.now() - timedelta(minutes=5)]
)
```

## 🚀 Main Pipeline (`main.py`)

Main pipeline tích hợp tất cả 4 chức năng thành một workflow hoàn chỉnh:

```python
from main import Day6MLPipeline

pipeline = Day6MLPipeline()

# Chạy toàn bộ pipeline
success = pipeline.run_full_pipeline(
    weeks_back=4,
    use_sample_data=True  # False để dùng New Relic thật
)
```

### Pipeline Steps:
1. **Step 1**: Thu thập data từ New Relic hoặc tạo sample data
2. **Step 2**: Tạo visualization charts
3. **Step 3**: Training time series model
4. **Step 4**: Demo predictions từ model

## 📂 Cấu trúc thư mục

```
Src/Day6/
├── main.py                      # Main pipeline
├── newrelic_data_collector.py   # Data collection
├── visualizer.py                # Data visualization  
├── time_series_trainer.py       # Model training
├── demo_predictor.py            # Prediction demo
├── README.md                    # Documentation
├── day6_charts/                 # Generated charts
├── day6_models/                 # Trained models
└── day6_collected_data.csv      # Collected data
```

## 🎯 Cải tiến so với Day5

### 1. **Modular Architecture**
- Tách thành 4 modules độc lập
- Có thể sử dụng riêng lẻ hoặc tích hợp
- Clean interfaces và error handling

### 2. **Enhanced Push Notification Logic**
- Daytime effect (7h-21h): Strong effect với exponential decay trong 15 phút
- Nighttime (1h-6h): Không có effect
- Other hours: Minimal effect
- Horizon-specific push effects cho predictions

### 3. **Improved Data Collection**
- Full week collection (Monday-Sunday)
- GMT+7 timezone support
- Adaptive granularity strategy
- Better error handling và fallback

### 4. **Advanced Visualization**
- Multiple chart types
- Comprehensive statistics
- Automatic saving với timestamps
- Better styling và formatting

### 5. **Robust Model Training**
- Time-based cross validation
- Feature selection
- Multiple optimized models
- Model persistence với metadata
- Training results visualization

### 6. **Flexible Prediction Interface**
- Multiple input methods
- Enhanced push effects
- Classification labels
- Detailed results với metadata

## 🔧 Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests python-dotenv pytz joblib
```

## 🌟 Environment Variables

Để sử dụng New Relic data collection, cần set environment variables:

```bash
export NEWRELIC_API_KEY="your_api_key"
export NEWRELIC_APP_ID="your_app_id"
```

Hoặc tạo file `.env`:
```
NEWRELIC_API_KEY=your_api_key
NEWRELIC_APP_ID=your_app_id
```

## 🚀 Quick Start

1. **Chạy full pipeline**:
```bash
cd Src/Day6
python main.py
```

2. **Sử dụng từng component riêng**:
```python
# Data collection
from newrelic_data_collector import collect_full_week_newrelic_data
data = collect_full_week_newrelic_data(weeks_back=2)

# Visualization
from visualizer import create_comprehensive_visualization_suite
figures = create_comprehensive_visualization_suite(data=data)

# Training
from time_series_trainer import TimeSeriesTrainer
trainer = TimeSeriesTrainer()
results = trainer.train_time_series_model(data=data)

# Prediction
from demo_predictor import DemoPredictor
predictor = DemoPredictor()
predictor.load_trained_model()
predictor.demo_predictions()
```

## 📊 Test Results

Pipeline test thành công với:
- ✅ Data collection: 672 records (4 weeks sample data)
- ✅ Visualization: 4 charts generated
- ✅ Model training: 3 models trained, best R² = 0.999
- ✅ Demo predictions: Multiple scenarios tested successfully

## 🎉 Kết luận

Day6 đã thành công tổng hợp và cải tiến tất cả các chức năng từ Day5 thành một hệ thống modular, robust và dễ sử dụng. Tất cả 4 yêu cầu đã được hoàn thành:

1. ✅ **Hàm collect data từ New Relic**: Full week collection với GMT+7 timezone
2. ✅ **Hàm visualizer**: Comprehensive charts với multiple types
3. ✅ **Hàm training model**: Advanced time series model với enhanced features
4. ✅ **Hàm demo**: Flexible prediction interface với multiple input methods

Hệ thống có thể được sử dụng độc lập từng component hoặc chạy full pipeline, phù hợp cho cả development và production use cases.