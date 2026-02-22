#!/usr/bin/env python3
"""
Test prediction endpoints
"""
import requests
import json

def test_prediction(image_file):
    """Test prediction with given image file"""
    url = "http://localhost:8000/predict"
    
    try:
        with open(image_file, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {image_file}: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
            return True
        else:
            print(f"❌ {image_file}: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {image_file}: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing predictions...")
    test_prediction("Test_img.jpg")
    test_prediction("Test_img1.jpg")
