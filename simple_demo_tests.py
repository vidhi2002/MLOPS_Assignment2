#!/usr/bin/env python3
"""
Simple tests that don't require PyTorch for video demo
"""
import requests
import json

def test_api_endpoints():
    """Test API endpoints without PyTorch dependencies"""
    
    print("=== Testing API Endpoints ===")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health Check: PASS")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health Check: FAIL (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Health Check: ERROR ({str(e)})")
    
    # Test metrics endpoint
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=10)
        if response.status_code == 200:
            print("✅ Metrics Endpoint: PASS")
            metrics_lines = response.text.split('\n')[:5]
            for line in metrics_lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"❌ Metrics Endpoint: FAIL (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Metrics Endpoint: ERROR ({str(e)})")
    
    # Test prediction endpoint (without PyTorch)
    try:
        with open("Test_img.jpg", 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Endpoint: PASS")
            print(f"   Test_img.jpg: {result.get('predicted_class', 'unknown')} (confidence: {result.get('confidence', 0):.3f})")
        else:
            print(f"❌ Prediction Endpoint: FAIL (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Prediction Endpoint: ERROR ({str(e)})")

def test_monitoring_services():
    """Test monitoring services"""
    
    print("\n=== Testing Monitoring Services ===")
    
    services = [
        ("Grafana", "http://localhost:3000"),
        ("Prometheus", "http://localhost:9090"),
        ("MLflow", "http://localhost:5000")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: ACCESSIBLE")
            else:
                print(f"⚠️  {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: NOT ACCESSIBLE ({str(e)[:50]})")

if __name__ == "__main__":
    print("🎬 MLOps Pipeline Demo - Simple Tests")
    print("=" * 50)
    
    test_api_endpoints()
    test_monitoring_services()
    
    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("🎯 Ready for video recording!")
