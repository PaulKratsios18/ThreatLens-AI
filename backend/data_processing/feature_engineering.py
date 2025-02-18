import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.cluster import KMeans

class GTDFeatureEngineer:
    def __init__(self):
        self.base_features = [
            'region', 'country',
            'attacktype1', 'weaptype1', 'targtype1',
            'gname', 'nperps', 'nperpcap'
        ]
        
    def engineer_features(self, df):
        """Engineer features for attack prediction"""
        df = df.copy()
        
        # Time-based features
        df.loc[:, 'iyear'] = df['iyear'].astype(int)
        df.loc[:, 'imonth'] = df['imonth'].fillna(1).astype(int)
        df.loc[:, 'iday'] = df['iday'].fillna(1).astype(int)
        
        # Create date features
        date_strings = (
            df['iyear'].astype(str) + '-' +
            df['imonth'].astype(str).str.zfill(2) + '-' +
            df['iday'].astype(str).str.zfill(2)
        )
        df.loc[:, 'date'] = pd.to_datetime(date_strings, errors='coerce')
        df.loc[:, 'month'] = df['imonth']
        df.loc[:, 'day_of_week'] = df['date'].dt.dayofweek.fillna(0).astype(int)
        
        # Group-based features
        df.loc[:, 'group_activity_region'] = self._calculate_group_region_activity(df)
        df.loc[:, 'group_target_preference'] = self._calculate_group_targets(df)
        
        # Attack severity features - handle division by zero
        total_casualties = df['nkill'].fillna(0) + df['nwound'].fillna(0)
        perpetrators = df['nperps'].fillna(1).replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        df.loc[:, 'casualty_rate'] = (total_casualties / perpetrators).clip(0, 1000)  # Clip to reasonable range
        
        df.loc[:, 'multiple_targets'] = (df['targtype1'] != df['targtype1'].shift()).astype(int)
        
        return df
        
    def _calculate_group_region_activity(self, df):
        """Calculate how active each group is in different regions"""
        group_region_counts = df.groupby(['gname', 'region']).size().unstack(fill_value=0)
        group_region_prefs = group_region_counts.idxmax(axis=1)
        return df['gname'].map(group_region_prefs).fillna(-1)
    
    def _calculate_group_targets(self, df):
        """Calculate preferred target types for each group"""
        group_target_counts = df.groupby(['gname', 'targtype1']).size().unstack(fill_value=0)
        group_target_prefs = group_target_counts.idxmax(axis=1)
        return df['gname'].map(group_target_prefs).fillna(-1)
