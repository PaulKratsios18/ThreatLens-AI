import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class GTDDataLoader:
    def __init__(self):
        self.data = None
        
    def load_data(self, path: str) -> pd.DataFrame:
        """Load and preprocess GTD data"""
        self.data = pd.read_csv(path)
        self._clean_data()
        return self.data
    
    def _clean_data(self):
        """Clean and preprocess GTD data"""
        # Convert date columns
        date_cols = ['iyear', 'imonth', 'iday']
        self.data['date'] = pd.to_datetime(
            self.data[date_cols].assign(
                iday=self.data.iday.fillna(1)
            )
        )
        
        # Clean numeric columns
        numeric_cols = ['nkill', 'nwound']
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        
        # Clean categorical columns
        self.data['gname'] = self.data['gname'].fillna('Unknown')
        
    def get_attacks_by_region(self) -> pd.DataFrame:
        """Get attack counts by region"""
        return self.data.groupby('region_txt')['eventid'].count()
    
    def get_attack_coordinates(self) -> List[Dict]:
        """Get coordinates of all attacks for mapping"""
        return self.data[['latitude', 'longitude', 'city', 'country_txt', 'nkill']].dropna().to_dict('records')

class DataLoader:
    def __init__(self):
        self.gtd_data = None
        self.socioeconomic_data = None
    
    def load_gtd_data(self, path: str) -> pd.DataFrame:
        """Load Global Terrorism Database data"""
        self.gtd_data = pd.read_csv(path)
        return self.gtd_data
    
    def load_socioeconomic_data(self, path: str) -> pd.DataFrame:
        """Load World Bank socioeconomic indicators"""
        self.socioeconomic_data = pd.read_csv(path)
        return self.socioeconomic_data 

class GTDAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze attack patterns over time"""
        monthly = self.data.groupby(['iyear', 'imonth'])['eventid'].count()
        return {
            'total_attacks': len(self.data),
            'monthly_trends': monthly.to_dict(),
            'peak_month': monthly.idxmax()
        }
    
    def analyze_attack_types(self) -> Dict:
        """Analyze most common attack types"""
        return self.data['attacktype1_txt'].value_counts().to_dict()
    
    def analyze_casualties(self) -> Dict:
        """Analyze casualty statistics"""
        return {
            'total_killed': self.data['nkill'].sum(),
            'total_wounded': self.data['nwound'].sum(),
            'avg_casualties_per_attack': self.data['nkill'].mean()
        } 