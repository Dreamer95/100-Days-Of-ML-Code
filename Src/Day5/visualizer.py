from Src.Day5.newrelic_traffic_collector import create_traffic_visualization_suite_enhanced

def main():
    print("\n🎨 Creating traffic visualizations...")
    create_traffic_visualization_suite_enhanced(
        start_date='2025-07-24',
        end_date='2025-08-03'
    )

if __name__ == "__main__":
    main()