import pandas as pd
import numpy as np

class TemporalFeatureEngineer:
    def __init__(self):
        self.attack_types = {}
        self.weapon_types = {}
        
    def engineer_features(self, df):
        df = df.copy()
        
        # Create temporal features
        df['month'] = df['imonth'].fillna(1).astype(int)
        df['quarter'] = pd.qcut(df['month'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Calculate attack frequencies
        df['attack_frequency'] = df.groupby(['iyear', 'region'])['eventid'].transform('count')
        df['weapon_frequency'] = df.groupby(['iyear', 'region', 'weaptype1'])['eventid'].transform('count')
        df['target_frequency'] = df.groupby(['iyear', 'region', 'targtype1'])['eventid'].transform('count')
        
        # Calculate rolling statistics
        df['rolling_attacks'] = df.groupby('region')['attack_frequency'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['attack_trend'] = df.groupby('region')['attack_frequency'].transform(lambda x: x.pct_change())
        
        return df 