import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class GTDPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = [
            'attacktype1', 'weaptype1', 'targtype1',
            'country', 'vicinity', 'nperps', 'nperpcap',
            'property', 'ishostkid', 'nkill', 'nwound',
            'group_activity_region', 'group_target_preference',
            'month', 'day_of_week', 'casualty_rate',
            'multiple_targets', 'INT_LOG', 'INT_IDEO', 'INT_ANY'
        ]
        self.region_names = {
            0: "North America",
            1: "Central America & Caribbean",
            2: "South America",
            3: "East Asia",
            4: "Southeast Asia",
            5: "South Asia",
            6: "Central Asia",
            7: "Western Europe",
            8: "Eastern Europe",
            9: "Middle East & North Africa",
            10: "Sub-Saharan Africa",
            11: "Russia & the Newly Independent States",
            12: "Australasia & Oceania"
        }
        
    def get_region_name(self, region_id: int) -> str:
        """Get region name from region ID"""
        return self.region_names.get(region_id, "Unknown Region")
        
    def fit_transform(self, X):
        """Fit and transform the features"""
        X = X.copy()
        
        # Handle missing and infinite values
        for feature in self.feature_names:
            if feature in X.columns:
                # Replace infinities with NaN then fill with 0
                X[feature] = X[feature].replace([np.inf, -np.inf], np.nan)
                if X[feature].dtype == 'object':
                    X[feature] = X[feature].fillna('Unknown')
                else:
                    X[feature] = X[feature].fillna(0)
        
        # Encode categorical variables
        for feature in self.feature_names:
            if feature in X.columns and X[feature].dtype == 'object':
                self.label_encoders[feature] = LabelEncoder()
                X[feature] = self.label_encoders[feature].fit_transform(X[feature])
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        return X_scaled
    
    def transform(self, X):
        """Transform features using fitted encoders and scaler"""
        X = X.copy()
        
        # Handle missing and infinite values
        for feature in self.feature_names:
            if feature in X.columns:
                X[feature] = X[feature].replace([np.inf, -np.inf], np.nan)
                if X[feature].dtype == 'object':
                    X[feature] = X[feature].fillna('Unknown')
                else:
                    X[feature] = X[feature].fillna(0)
        
        # Transform with fitted encoders
        for feature, encoder in self.label_encoders.items():
            if feature in X.columns:
                # Handle unseen categories
                X[feature] = X[feature].map(lambda x: 'Unknown' if x not in encoder.classes_ else x)
                X[feature] = encoder.transform(X[feature])
        
        # Scale numerical features
        return self.scaler.transform(X[self.feature_names])

    def preprocess(self, data: pd.DataFrame):
        """Preprocess GTD data for neural network"""
        # Create target variable (success)
        data['success'] = data['success'].astype(int)
        
        # Handle missing values
        for feature in self.feature_names:
            if data[feature].dtype == 'object':
                data[feature] = data[feature].fillna('Unknown')
            else:
                data[feature] = data[feature].fillna(-1)
        
        # Encode categorical variables
        for feature in self.feature_names:
            if data[feature].dtype == 'object':
                self.label_encoders[feature] = LabelEncoder()
                data[feature] = self.label_encoders[feature].fit_transform(data[feature])
        
        # Scale numerical features
        X = self.scaler.fit_transform(data[self.feature_names])
        y = data['success']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
