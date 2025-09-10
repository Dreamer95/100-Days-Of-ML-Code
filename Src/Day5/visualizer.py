from Src.Day5.newrelic_traffic_collector import create_traffic_visualization_suite_enhanced

def main():
    print("\n🎨 Creating traffic visualizations...")
    create_traffic_visualization_suite_enhanced(
        start_date='2025-05-01',
        end_date='2025-09-30'
    )

if __name__ == "__main__":
    main()