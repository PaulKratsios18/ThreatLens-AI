import requests

def test_explanation(features, case_name):
    explain_url = "http://localhost:8000/explain"
    
    # Get explanation
    exp_response = requests.post(explain_url, json={"features": features})
    
    # Check if response is error
    if exp_response.status_code != 200:
        print(f"\nError in {case_name}:")
        print(f"Status code: {exp_response.status_code}")
        print(f"Error message: {exp_response.json()['detail']}")
        return
        
    result = exp_response.json()
    print(f"\n{case_name}:")
    print(f"Prediction: {result['success_probability']:.3f}")
    print(f"Risk Level: {result['risk_level']}")
    print("\nFeature Importance:")
    
    # Sort explanations by absolute importance
    sorted_explanations = sorted(result['explanations'], 
                                key=lambda x: abs(x[1]),
                                reverse=True)
    
    for feature, importance in sorted_explanations:
        print(f"{feature}: {importance:+.3f}")

# Use the same test cases from extreme_cases.py
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

# Add after line 65
med_risk = {feature: value/2 for feature, value in max_risk.items()}

# Run explanation tests
test_explanation(min_risk, "MINIMUM RISK EXPLANATION")
test_explanation(max_risk, "MAXIMUM RISK EXPLANATION")
test_explanation(med_risk, "MEDIUM RISK EXPLANATION") 