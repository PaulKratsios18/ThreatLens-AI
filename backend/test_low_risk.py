import requests

def test_prediction(features, case_name):
    url = "http://localhost:8000/predict"
    response = requests.post(url, json={"features": features})
    print(f"\n{case_name}:")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

# Test Case 1: Absolute Minimum Risk
min_risk = {feature: 0 for feature in [
    'attacktype1', 'weaptype1', 'targtype1',
    'region', 'vicinity', 'suicide', 'multiple',
    'individual', 'nperps', 'nperpcap', 'property',
    'ishostkid', 'INT_LOG', 'INT_IDEO', 'INT_ANY'
]}

# Test Case 2: Maximum Risk
max_risk = {feature: 5 for feature in [
    'attacktype1', 'weaptype1', 'targtype1',
    'region', 'vicinity', 'suicide', 'multiple',
    'individual', 'nperps', 'nperpcap', 'property',
    'ishostkid', 'INT_LOG', 'INT_IDEO', 'INT_ANY'
]}

# Run tests
test_prediction(min_risk, "MINIMUM RISK CASE")
test_prediction(max_risk, "MAXIMUM RISK CASE")