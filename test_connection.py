#!/usr/bin/env python3
"""
Test script to verify the ML model connection
"""
import requests
import json

def test_ml_api():
    """Test the ML API directly"""
    print("🧪 Testing ML API connection...")
    
    # Test data matching the training data structure
    test_data = {
        "sensors": {
            "temperature": 85.5,
            "vibration": 2.3,
            "pressure": 150.0,
            "rpm": 2000
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ ML API is working!")
            print(f"   Prediction: {result.get('prediction')}")
            print(f"   Probability: {result.get('probability', 'N/A')}")
            return True
        else:
            print(f"❌ ML API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to ML API. Make sure it's running on port 8000")
        return False
    except Exception as e:
        print(f"❌ ML API test failed: {e}")
        return False

def test_backend_api():
    """Test the backend API"""
    print("\n🧪 Testing Backend API connection...")
    
    test_data = {
        "temperature": 85.5,
        "vibration": 2.3,
        "pressure": 150.0,
        "rpm": 2000
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/predictions",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 201:
            result = response.json()
            print("✅ Backend API is working!")
            print(f"   Result: {result.get('result')}")
            return True
        else:
            print(f"❌ Backend API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Backend API. Make sure it's running on port 5000")
        return False
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Predictive Maintenance System Connection\n")
    
    ml_ok = test_ml_api()
    backend_ok = test_backend_api()
    
    print(f"\n📊 Test Results:")
    print(f"   ML API: {'✅ Working' if ml_ok else '❌ Failed'}")
    print(f"   Backend API: {'✅ Working' if backend_ok else '❌ Failed'}")
    
    if ml_ok and backend_ok:
        print("\n🎉 All systems are working! Your frontend should connect successfully.")
    else:
        print("\n⚠️  Some issues found. Please check the services and try again.")
