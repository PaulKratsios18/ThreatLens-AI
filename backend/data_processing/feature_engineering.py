import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.cluster import KMeans
from .socioeconomic_data import SocioeconomicDataLoader

class GTDFeatureEngineer:
    def __init__(self):
        self.base_features = [
            'region', 'country',
            'attacktype1', 'weaptype1', 'targtype1',
            'gname', 'nperps', 'nperpcap'
        ]
        self.socioeconomic_loader = SocioeconomicDataLoader()
        
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
        
        # Add socioeconomic features
        df = self._add_socioeconomic_features(df)
        
        return df
    
    def _add_socioeconomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add socioeconomic features to the dataset"""
        # Load socioeconomic data
        try:
            socio_data = self.socioeconomic_loader.load_socioeconomic_data()
            
            # Add country-level socioeconomic features
            for _, row in df.iterrows():
                country = row['country']
                year = row['iyear']
                
                # Get country data
                country_data = self.socioeconomic_loader.get_country_data(country, year)
                
                # Add features to the row
                for feature, value in country_data.items():
                    if feature not in ['country', 'year']:
                        df.loc[_, f'socio_{feature}'] = value
            
            # Add region-level socioeconomic features
            for _, row in df.iterrows():
                region = row['region']
                year = row['iyear']
                
                # Get region averages
                region_data = self.socioeconomic_loader.get_region_averages(region, year)
                
                # Add features to the row
                for feature, value in region_data.items():
                    if feature not in ['country', 'year']:
                        df.loc[_, f'region_socio_{feature}'] = value
            
            # Calculate socioeconomic risk factors
            df['socioeconomic_risk'] = self._calculate_socioeconomic_risk(df)
            
        except Exception as e:
            print(f"Error adding socioeconomic features: {e}")
            # Add placeholder columns with NaN values
            for feature in ['gdp_per_capita', 'unemployment_rate', 'gini_index', 'population',
                          'urban_population_percent', 'primary_school_enrollment', 'life_expectancy']:
                df[f'socio_{feature}'] = np.nan
                df[f'region_socio_{feature}'] = np.nan
            df['socioeconomic_risk'] = np.nan
        
        return df
    
    def _calculate_socioeconomic_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a composite socioeconomic risk score"""
        # Define weights for different indicators
        weights = {
            'gdp_per_capita': -0.3,  # Negative weight as higher GDP reduces risk
            'unemployment_rate': 0.4,
            'gini_index': 0.3,
            'urban_population_percent': 0.1,
            'primary_school_enrollment': -0.2,
            'life_expectancy': -0.1
        }
        
        # Calculate weighted sum
        risk_score = pd.Series(0, index=df.index)
        for feature, weight in weights.items():
            if f'socio_{feature}' in df.columns:
                # Normalize the feature
                normalized_feature = (df[f'socio_{feature}'] - df[f'socio_{feature}'].mean()) / df[f'socio_{feature}'].std()
                risk_score += normalized_feature * weight
        
        # Normalize the final risk score to 0-1 range
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
        
        return risk_score
    
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
