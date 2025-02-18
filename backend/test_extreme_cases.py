import requests

def test_prediction(features, case_name):
    url = "http://localhost:8000/predict"
    response = requests.post(url, json={"features": features})
    print(f"\n{case_name}:")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")
    
# Test Case 1: Absolute Minimum Risk
min_risk = {
    'attacktype1': 0,
    'weaptype1': 0,
    'targtype1': 0,
    'region': 0,
    'vicinity': 0,
    'suicide': 0,
    'multiple': 0,
    'individual': 0,
    'nperps': 0,
    'nperpcap': 0,
    'property': 0,
    'ishostkid': 0,
    'INT_LOG': 0,
    'INT_IDEO': 0,
    'INT_ANY': 0
}

# Test Case 2: Maximum Risk
max_risk = {
    'attacktype1': 5,
    'weaptype1': 5,
    'targtype1': 5,
    'region': 5,
    'vicinity': 1,
    'suicide': 1,
    'multiple': 1,
    'individual': 1,
    'nperps': 10,
    'nperpcap': 5,
    'property': 1,
    'ishostkid': 1,
    'INT_LOG': 1,
    'INT_IDEO': 1,
    'INT_ANY': 1
}

# Test Case 3: Medium Risk
med_risk = {feature: value/2 for feature, value in max_risk.items()}

# Run tests
test_prediction(min_risk, "MINIMUM RISK CASE")
test_prediction(max_risk, "MAXIMUM RISK CASE")
test_prediction(med_risk, "MEDIUM RISK CASE") 