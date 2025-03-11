import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List, Optional
from datetime import datetime

class GTDDataLoader:
    # Class-level cache to persist across instances
    _data_cache = {}
    
    def __init__(self):
        self.data = None
        
    def load_data(self, path: str, use_cache: bool = True) -> pd.DataFrame:
        """Load and preprocess GTD data with caching support"""
        # Use cached data if available
        cache_key = f"data_{os.path.basename(path)}"
        
        if use_cache and cache_key in GTDDataLoader._data_cache:
            print(f"Using cached data for {os.path.basename(path)}")
            self.data = GTDDataLoader._data_cache[cache_key]
            return self.data
            
        # Look for a pickle file first (much faster loading)
        pickle_path = path.replace('.csv', '.pkl')
        if use_cache and os.path.exists(pickle_path):
            try:
                print(f"Loading from pickle file: {pickle_path}")
                with open(pickle_path, 'rb') as f:
                    self.data = pickle.load(f)
                # Store in class cache
                GTDDataLoader._data_cache[cache_key] = self.data
                return self.data
            except Exception as e:
                print(f"Error loading pickle: {str(e)}")
                # Continue to CSV loading if pickle fails
        
        # Specify dtypes for problematic columns
        dtype_dict = {
            'iyear': 'Int64',
            'imonth': 'Int64',
            'iday': 'Int64',
            'region': 'Int64',
            'vicinity': 'Int64',
            'success': 'Int64',
            'suicide': 'Int64',
            'multiple': 'Int64',
            'nperps': 'float64',
            'nperpcap': 'float64'
        }
        
        try:
            print(f"Loading CSV file: {path}")
            # For large files, only load necessary columns to reduce memory
            essential_columns = None
            if os.path.getsize(path) > 100 * 1024 * 1024:  # If file > 100MB
                essential_columns = [
                    'eventid', 'iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 
                    'region', 'country', 'provstate', 'city', 'latitude', 'longitude', 
                    'attacktype1', 'attacktype1_txt', 'nkill', 'nwound', 'gname'
                ]
                print(f"Large file detected, loading only essential columns")
            
            # Load with specified dtypes and low_memory=False
            self.data = pd.read_csv(
                path, 
                dtype=dtype_dict, 
                header=0, 
                low_memory=False,
                usecols=essential_columns
            )
            
            # Clean data after loading
            self._clean_data()
            
            # Save to cache
            GTDDataLoader._data_cache[cache_key] = self.data
            
            # Save to pickle for faster future loading
            if use_cache:
                try:
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(self.data, f, protocol=4)  # Protocol 4 for better compatibility
                    print(f"Saved processed data to {pickle_path}")
                except Exception as e:
                    print(f"Error saving pickle: {str(e)}")
            
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty DataFrame if error occurs
            return pd.DataFrame()
    
    def _clean_data(self):
        """Clean and normalize GTD data"""
        if self.data is None or self.data.empty:
            return
            
        # Handle date columns first
        self.data['iyear'] = pd.to_numeric(self.data['iyear'], errors='coerce')
        self.data['imonth'] = pd.to_numeric(self.data['imonth'], errors='coerce').fillna(1)
        self.data['iday'] = pd.to_numeric(self.data['iday'], errors='coerce').fillna(1)
        
        # Create date string with validation
        self.data['date'] = pd.to_datetime({
            'year': self.data['iyear'],
            'month': self.data['imonth'].clip(1, 12),  # Ensure valid months
            'day': self.data['iday'].clip(1, 31)  # Ensure valid days
        }, errors='coerce')
        
        # Handle missing values
        numeric_cols = ['nkill', 'nwound', 'nperps', 'nperpcap']
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
        
        # Clean categorical columns with sensible defaults
        categorical_cols = ['gname', 'country_txt', 'region_txt', 'attacktype1_txt', 
                          'targtype1_txt', 'weaptype1_txt']
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna('Unknown')
        
        # Clean location data
        self.data['latitude'] = pd.to_numeric(self.data['latitude'], errors='coerce')
        self.data['longitude'] = pd.to_numeric(self.data['longitude'], errors='coerce')
        
        # Convert boolean columns
        bool_cols = ['success', 'suicide', 'multiple']
        for col in bool_cols:
            self.data[col] = self.data[col].fillna(0).astype(int)
        
        # Remove entries with critical missing data
        self.data = self.data.dropna(subset=['date', 'country_txt', 'region_txt'])
        
        # Standardize numeric values
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
            
        # Clean property damage values
        self.data['property'] = self.data['property'].fillna(0).astype(int)
        
        # Handle extended text fields
        text_cols = ['summary', 'motive']
        for col in text_cols:
            self.data[col] = self.data[col].fillna('').str.strip()
        
    def get_attacks_by_region(self) -> pd.DataFrame:
        """Get attack frequency by region"""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        return self.data.groupby('region_txt').size().reset_index(name='count')
    
    def get_attack_coordinates(self) -> List[Dict]:
        """Get coordinates of attacks for mapping"""
        if self.data is None or self.data.empty:
            return []
            
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