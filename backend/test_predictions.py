def test_location_predictions():
    test_features = {
        'region': 5,  # Middle East
        'month': 6,
        'day_of_week': 3,
        'attacktype1': 1,
        'weaptype1': 2,
        'targtype1': 3,
        'nperps': 4,
        'nperpcap': 0,
        'location_cluster': 2,
        'group_activity_region': 15,
        'group_target_preference': 8
    }
    
    response = requests.post(
        "http://localhost:8000/predict-location",
        json={"features": test_features}
    )
    
    print("\nLocation Predictions:")
    for pred in response.json()['predictions']:
        print(f"{pred['region']}: {pred['probability']:.3f} ({pred['risk_level']} risk)") 