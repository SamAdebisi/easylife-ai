#!/bin/bash

# Load testing script for EasyLife AI services

set -e

echo "🚀 Starting EasyLife AI Load Testing..."

# Check if services are running
check_service() {
    local service_name=$1
    local port=$2
    local url="http://localhost:$port/health"

    if curl -s "$url" > /dev/null; then
        echo "✅ $service_name is running on port $port"
        return 0
    else
        echo "❌ $service_name is not running on port $port"
        return 1
    fi
}

# Check all services
echo "🔍 Checking service health..."
services_ok=true

check_service "NLP Service" 8001 || services_ok=false
check_service "CV Service" 8002 || services_ok=false
check_service "TS Forecasting" 8003 || services_ok=false
check_service "Recsys Service" 8004 || services_ok=false

if [ "$services_ok" = false ]; then
    echo "❌ Some services are not running. Please start all services first."
    echo "💡 Run: make up && python -m uvicorn nlp_service.app.main:app --host 0.0.0.0 --port 8001"
    exit 1
fi

# Install locust if not present
if ! command -v locust &> /dev/null; then
    echo "📦 Installing Locust..."
    pip install -r requirements.txt
fi

# Create results directory
mkdir -p results

# Run different load test scenarios
echo "🧪 Running load tests..."

# 1. Light load test (10 users, 1 minute)
echo "📊 Running light load test (10 users, 1 minute)..."
locust -f locustfile.py \
    --headless \
    --users 10 \
    --spawn-rate 2 \
    --run-time 1m \
    --html results/light_load_test.html \
    --csv results/light_load_test \
    --logfile results/light_load_test.log

# 2. Medium load test (50 users, 2 minutes)
echo "📊 Running medium load test (50 users, 2 minutes)..."
locust -f locustfile.py \
    --headless \
    --users 50 \
    --spawn-rate 5 \
    --run-time 2m \
    --html results/medium_load_test.html \
    --csv results/medium_load_test \
    --logfile results/medium_load_test.log

# 3. Heavy load test (100 users, 3 minutes)
echo "📊 Running heavy load test (100 users, 3 minutes)..."
locust -f locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 3m \
    --html results/heavy_load_test.html \
    --csv results/heavy_load_test \
    --logfile results/heavy_load_test.log

# 4. End-to-end scenario test
echo "🔄 Running end-to-end scenario test..."
locust -f locustfile.py \
    --headless \
    --users 20 \
    --spawn-rate 2 \
    --run-time 2m \
    --html results/e2e_test.html \
    --csv results/e2e_test \
    --logfile results/e2e_test.log

echo "✅ Load testing completed!"
echo "📊 Results saved in results/ directory:"
echo "   - HTML reports: results/*.html"
echo "   - CSV data: results/*.csv"
echo "   - Logs: results/*.log"

# Generate performance summary
echo "📈 Generating performance summary..."
python3 << 'EOF'
import pandas as pd
import glob
import os

def analyze_results():
    csv_files = glob.glob("results/*_stats.csv")

    print("\n📊 Performance Summary:")
    print("=" * 50)

    for csv_file in csv_files:
        test_name = os.path.basename(csv_file).replace('_stats.csv', '')
        df = pd.read_csv(csv_file)

        if not df.empty:
            avg_response_time = df['Average Response Time'].iloc[-1]
            rps = df['Current RPS'].iloc[-1]
            failure_rate = df['Failure Count'].iloc[-1] / df['Request Count'].iloc[-1] * 100

            print(f"\n{test_name.upper()}:")
            print(f"  Average Response Time: {avg_response_time:.2f}ms")
            print(f"  Requests per Second: {rps:.2f}")
            print(f"  Failure Rate: {failure_rate:.2f}%")

if __name__ == "__main__":
    try:
        analyze_results()
    except Exception as e:
        print(f"Error analyzing results: {e}")
EOF

echo "\n🎯 Load testing complete! Check the HTML reports for detailed analysis."
