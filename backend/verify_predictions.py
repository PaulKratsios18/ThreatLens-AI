import pandas as pd
import numpy as np
from data_processing.data_loader import GTDDataLoader
import requests
import folium
from datetime import datetime, timedelta

def verify_predictions():
    # Load recent data for verification
    data_loader = GTDDataLoader()
    recent_data = data_loader.load_data("../data/globalterrorismdb_0522dist.csv")
    recent_data = recent_data[recent_data['iyear'] >= 2020]
    
    # Create test cases from actual recent events
    test_cases = []
    for _, event in recent_data.head(5).iterrows():
        test_case = {
            'region': int(event['region']),
            'month': event['imonth'],
            'day_of_week': pd.Timestamp(event['date']).dayofweek,
            'attacktype1': event['attacktype1'],
            'weaptype1': event['weaptype1'],
            'targtype1': event['targtype1'],
            'nperps': event['nperps'],
            'nperpcap': event['nperpcap'],
            'location_cluster': 0,  # Will be updated by feature engineering
            'group_activity_region': 0,  # Will be updated
            'group_target_preference': 0  # Will be updated
        }
        test_cases.append((test_case, event))

    # Test each case
    print("\nVerifying Predictions Against Recent Events:")
    for test_case, actual_event in test_cases:
        response = requests.post(
            "http://localhost:8000/predict-location",
            json={"features": test_case}
        )
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"\nTest Case - {actual_event['city']}, {actual_event['country_txt']}")
            print(f"Actual Region: {actual_event['region_txt']}")
            print("\nPredicted Regions:")
            for pred in predictions['predictions']:
                print(f"- {pred['region']}: {pred['probability']:.3f} ({pred['risk_level']} risk)")
            
            # Calculate prediction accuracy
            actual_region = actual_event['region_txt']
            top_prediction = predictions['predictions'][0]['region']
            print(f"\nPrediction Match: {'✓' if actual_region == top_prediction else '✗'}")

if __name__ == "__main__":
    verify_predictions() 