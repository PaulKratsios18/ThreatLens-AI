from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    
    def train(self, X, y):
        """Train the risk prediction model"""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Make risk predictions"""
        return self.model.predict_proba(X) 