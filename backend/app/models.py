import numpy as np

class PredictionModel:
    def predict(self, features):
        # Simulated model predictions for testing
        regions = [
            "North America", "South America", "Western Europe", 
            "Eastern Europe", "Middle East", "North Africa",
            "Sub-Saharan Africa", "Central Asia", "South Asia", 
            "East Asia", "Southeast Asia", "Oceania"
        ]
        
        predictions = []
        for region in regions:
            pred = {
                'region': region,
                'expected_attacks': np.random.randint(50, 200),
                'confidence_score': np.random.uniform(0.3, 0.9),
                'attack_types': {
                    'Cyber Attack': np.random.uniform(0.2, 0.4),
                    'Physical Attack': np.random.uniform(0.3, 0.5),
                    'Infrastructure Attack': np.random.uniform(0.2, 0.4)
                }
            }
            predictions.append(pred)
        
        return predictions

model = PredictionModel() 