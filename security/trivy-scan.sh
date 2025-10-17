#!/bin/bash

# Security scanning script using Trivy

set -e

echo "🔒 Running security scans for EasyLife AI..."

# Install Trivy if not present
if ! command -v trivy &> /dev/null; then
    echo "📦 Installing Trivy..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install trivy
    else
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi
fi

# Create security reports directory
mkdir -p security/reports

# Scan Docker images
echo "🐳 Scanning Docker images..."
services=("nlp-service" "cv-service" "ts-forecasting" "recsys-service")

for service in "${services[@]}"; do
    echo "Scanning $service..."
    trivy image --format json --output "security/reports/${service}_vulnerabilities.json" "easylife-ai/${service}:latest" || true
    trivy image --format table --output "security/reports/${service}_vulnerabilities.txt" "easylife-ai/${service}:latest" || true
done

# Scan Kubernetes manifests
echo "☸️ Scanning Kubernetes manifests..."
trivy config --format json --output "security/reports/k8s_vulnerabilities.json" k8s/ || true
trivy config --format table --output "security/reports/k8s_vulnerabilities.txt" k8s/ || true

# Scan Python dependencies
echo "🐍 Scanning Python dependencies..."
for service in "${services[@]}"; do
    if [ -f "${service}/requirements.txt" ]; then
        echo "Scanning $service dependencies..."
        trivy fs --format json --output "security/reports/${service}_dependencies.json" "${service}/" || true
        trivy fs --format table --output "security/reports/${service}_dependencies.txt" "${service}/" || true
    fi
done

# Generate security summary
echo "📊 Generating security summary..."
python3 << 'EOF'
import json
import os
import glob

def analyze_security_reports():
    print("\n🔒 Security Scan Summary:")
    print("=" * 50)

    # Analyze image vulnerabilities
    image_reports = glob.glob("security/reports/*_vulnerabilities.json")
    for report in image_reports:
        service = os.path.basename(report).replace('_vulnerabilities.json', '')
        try:
            with open(report, 'r') as f:
                data = json.load(f)
                if 'Results' in data:
                    total_vulns = len(data['Results'])
                    critical = sum(1 for r in data['Results'] if r.get('Severity') == 'CRITICAL')
                    high = sum(1 for r in data['Results'] if r.get('Severity') == 'HIGH')
                    medium = sum(1 for r in data['Results'] if r.get('Severity') == 'MEDIUM')
                    low = sum(1 for r in data['Results'] if r.get('Severity') == 'LOW')

                    print(f"\n{service.upper()}:")
                    print(f"  Total Vulnerabilities: {total_vulns}")
                    print(f"  Critical: {critical}")
                    print(f"  High: {high}")
                    print(f"  Medium: {medium}")
                    print(f"  Low: {low}")
        except Exception as e:
            print(f"Error analyzing {service}: {e}")

if __name__ == "__main__":
    analyze_security_reports()
EOF

echo "✅ Security scanning completed!"
echo "📊 Reports saved in security/reports/ directory"
echo "🔍 Review the reports and address any critical/high severity issues"
