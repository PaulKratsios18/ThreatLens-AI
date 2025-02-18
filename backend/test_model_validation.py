import pandas as pd
import numpy as np
from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
import requests

def validate_model():
    # Test extreme cases
    test_cases = [
        {
            "name": "High Risk Middle East",
            "features": {
                'attacktype1': 5,  # Maximum attack type
                'weaptype1': 5,    # Maximum weapon type
                'targtype1': 5,    # Maximum target type
                'region': 9,       # Middle East & North Africa
                'vicinity': 1,
                'suicide': 1,      # Suicide attack
                'multiple': 1,     # Multiple attacks
                'individual': 0,
                'nperps': 10,      # High number of perpetrators
                'nperpcap': 0,
                'property': 1,
                'ishostkid': 1,    # Hostage situation
                'INT_LOG': 1,
                'INT_IDEO': 1,
                'INT_ANY': 1
            }
        },
        {
            "name": "Low Risk North America",
            "features": {
                'attacktype1': 1,
                'weaptype1': 1,
                'targtype1': 1,
                'region': 0,       # North America
                'vicinity': 0,
                'suicide': 0,
                'multiple': 0,
                'individual': 1,
                'nperps': 1,
                'nperpcap': 1,
                'property': 0,
                'ishostkid': 0,
                'INT_LOG': 0,
                'INT_IDEO': 0,
                'INT_ANY': 0
            }
        }
    ]

    print("\nModel Validation Test")
    print("=" * 50)

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                "http://localhost:8000/predict-location",
                json={"features": case['features']}
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

if __name__ == "__main__":
    validate_model() 