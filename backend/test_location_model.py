import pandas as pd
import numpy as np
from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
import requests
import json

def test_model():
    # Load test data
    data_loader = GTDDataLoader()
    preprocessor = GTDPreprocessor()
    
    # Test cases for different regions
    test_regions = [
        {"name": "Middle East", "code": 5},
        {"name": "South Asia", "code": 6},
        {"name": "Western Europe", "code": 8},
        {"name": "North America", "code": 1}
    ]
    
    print("Testing Location Predictions\n")
    print("=" * 50)
    
    for region in test_regions:
        # Create complete feature set matching preprocessor.feature_names
        test_features = {
            'attacktype1': 1,
            'weaptype1': 2,
            'targtype1': 3,
            'region': region['code'],
            'vicinity': 1,
            'suicide': 0,
            'multiple': 0,
            'individual': 1,
            'nperps': 4,
            'nperpcap': 0,
            'property': 1,
            'ishostkid': 0,
            'INT_LOG': 1,
            'INT_IDEO': 1,
            'INT_ANY': 1
        }
        
        print(f"\nTesting Region: {region['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                "http://localhost:8000/predict-location",
                json={"features": test_features}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("Predictions:")
                for pred in result['predictions']:
                    print(f"- {pred['region']}: {pred['probability']:.3f} ({pred['risk_level']} risk)")
                print(f"Confidence Score: {result['confidence_score']:.3f}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            
        print("-" * 30)

if __name__ == "__main__":
    test_model() 