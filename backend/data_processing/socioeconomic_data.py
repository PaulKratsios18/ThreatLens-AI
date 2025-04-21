import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import requests
from pathlib import Path
import os
from .country_utils import standardize_country_name, get_socioeconomic_country_name

class SocioeconomicDataLoader:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'socioeconomic'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.data_dir / 'socioeconomic_cache.pkl'
        self.cached_data = None
        
    def _download_world_bank_data(self, indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Download data from World Bank API"""
        url = f"http://api.worldbank.org/v2/country/all/indicator/{indicator}?date={start_year}:{end_year}&format=json"
        response = requests.get(url)
        data = response.json()
        
        # Process the data
        records = []
        for entry in data[1]:
            country_name = entry['country']['value']
            # Standardize country name when saving to cache
            records.append({
                'country': country_name,
                'standardized_country': standardize_country_name(country_name),
                'year': int(entry['date']),
                'value': entry['value']
            })
        
        return pd.DataFrame(records)
    
    def _download_un_data(self, indicator: str) -> pd.DataFrame:
        """Download data from UN Data API"""
        # This is a placeholder - you'll need to implement the actual UN API call
        # For now, we'll return an empty DataFrame
        return pd.DataFrame()
    
    def load_socioeconomic_data(self, force_download: bool = False) -> pd.DataFrame:
        """Load socioeconomic data from cache or download if needed"""
        if not force_download and self.cached_data is not None:
            return self.cached_data
            
        if not force_download and self.cache_file.exists():
            self.cached_data = pd.read_pickle(self.cache_file)
            return self.cached_data
            
        # Define indicators to download
        indicators = {
            'NY.GDP.PCAP.CD': 'gdp_per_capita',
            'SL.UEM.TOTL.ZS': 'unemployment_rate',
            'SI.POV.GINI': 'gini_index',
            'SP.POP.TOTL': 'population',
            'SP.URB.TOTL.IN.ZS': 'urban_population_percent',
            'SE.PRM.ENRR': 'primary_school_enrollment',
            'SP.DYN.LE00.IN': 'life_expectancy'
        }
        
        # Download data for each indicator
        dfs = []
        for indicator, col_name in indicators.items():
            try:
                df = self._download_world_bank_data(indicator, 1970, 2022)
                df = df.rename(columns={'value': col_name})
                dfs.append(df)
            except Exception as e:
                print(f"Error downloading {indicator}: {e}")
        
        # Merge all dataframes
        if not dfs:
            print("Warning: No socioeconomic data could be downloaded. Using dummy data.")
            # Create a dummy DataFrame with basic data
            dummy_data = self._create_dummy_data()
            self.cached_data = dummy_data
            return dummy_data
            
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=['country', 'standardized_country', 'year'], how='outer')
        
        # Save to cache
        merged_df.to_pickle(self.cache_file)
        self.cached_data = merged_df
        
        return merged_df
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy data for testing when real data can't be loaded"""
        countries = ["United States", "Canada", "United Kingdom", "France", "Germany", 
                     "Russia", "China", "Japan", "India", "Brazil", "South Africa", 
                     "Australia", "Mexico", "Egypt", "Nigeria", "Saudi Arabia", "Iraq", 
                     "Iran", "Israel", "Syria"]
        
        years = list(range(1970, 2023))
        
        records = []
        for country in countries:
            for year in years:
                std_country = standardize_country_name(country)
                records.append({
                    'country': country,
                    'standardized_country': std_country,
                    'year': year,
                    'gdp_per_capita': np.random.uniform(1000, 60000),
                    'unemployment_rate': np.random.uniform(2, 15),
                    'gini_index': np.random.uniform(25, 60),
                    'population': np.random.uniform(1000000, 300000000),
                    'urban_population_percent': np.random.uniform(30, 90),
                    'primary_school_enrollment': np.random.uniform(70, 100),
                    'life_expectancy': np.random.uniform(60, 85)
                })
        
        return pd.DataFrame(records)
    
    def get_country_data(self, country: str, year: int) -> Dict[str, float]:
        """Get socioeconomic data for a specific country and year"""
        if self.cached_data is None:
            self.load_socioeconomic_data()
        
        # First standardize the input country name    
        try:
            standardized_country = standardize_country_name(country)
            
            # Convert standard name to socioeconomic data name if needed
            socio_country = get_socioeconomic_country_name(standardized_country)
                
            # Try looking up by standardized country first
            data = self.cached_data[
                (self.cached_data['standardized_country'] == standardized_country) & 
                (self.cached_data['year'] == year)
            ]
            
            # If no data found, try by original country name
            if data.empty:
                data = self.cached_data[
                    (self.cached_data['country'] == socio_country) & 
                    (self.cached_data['year'] == year)
                ]
            
            # If still empty, try to get the closest year's data
            if data.empty:
                data = self.cached_data[
                    ((self.cached_data['standardized_country'] == standardized_country) | 
                     (self.cached_data['country'] == socio_country)) & 
                    (self.cached_data['year'] <= year)
                ].sort_values('year', ascending=False)
                
                if not data.empty:
                    data = data.iloc[0]
            elif len(data) > 1:
                # If multiple rows are found, take the first one
                data = data.iloc[0]
                
            if len(data) == 0:
                # Return empty data silently instead of logging a warning
                return {}
                
            result = data.to_dict()
            if isinstance(result, dict) and 'series' in result:
                return result
            return result
                
        except Exception as e:
            # Silently return empty data instead of logging every error
            # Only log unexpected errors, not common 'standardized_country' key errors
            if 'standardized_country' not in str(e):
                print(f"Error getting socioeconomic data for {country}: {e}")
            return {}
    
    def get_region_averages(self, country_or_region: str, year: int) -> Dict[str, float]:
        """Get average socioeconomic indicators for a region"""
        if self.cached_data is None:
            self.load_socioeconomic_data()
            
        try:
            # Check if the input is a country or a region
            standardized_input = standardize_country_name(country_or_region)
            
            # If it's a country, get its region
            region = self._get_region_for_country(standardized_input)
            if not region:
                region = country_or_region  # Use as-is if not found
                
            # Map region to countries
            region_countries = self._get_region_countries(region)
            
            # If no countries in the region, return empty data silently
            if not region_countries:
                return {}
                
            # Get standardized country names for the region
            standardized_countries = [standardize_country_name(c) for c in region_countries]
            
            # Filter data by standardized country and year
            data = self.cached_data[
                (self.cached_data['standardized_country'].isin(standardized_countries)) & 
                (self.cached_data['year'] == year)
            ]
            
            # If no data found, look up by original country names
            if data.empty:
                data = self.cached_data[
                    (self.cached_data['country'].isin(region_countries)) & 
                    (self.cached_data['year'] == year)
                ]
            
            # If still empty, try to get the closest year's data
            if data.empty:
                latest_years = {}
                for country in standardized_countries:
                    country_data = self.cached_data[
                        ((self.cached_data['standardized_country'] == country) | 
                         (self.cached_data['country'].isin(region_countries))) & 
                        (self.cached_data['year'] <= year)
                    ].sort_values('year', ascending=False)
                    
                    if not country_data.empty:
                        latest_years[country] = country_data.iloc[0]['year']
                
                if latest_years:
                    # Find the most common latest year
                    most_common_year = max(set(latest_years.values()), key=list(latest_years.values()).count)
                    
                    # Get data for that year
                    data = self.cached_data[
                        ((self.cached_data['standardized_country'].isin(standardized_countries)) | 
                         (self.cached_data['country'].isin(region_countries))) & 
                        (self.cached_data['year'] == most_common_year)
                    ]
            
            if data.empty:
                # Return empty data silently
                return {}
                
            # Calculate mean for numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            result = data[numeric_cols].mean().to_dict()
            
            # Add region info
            result['region'] = region
            
            return result
                
        except Exception as e:
            # Silently return empty data instead of logging every error
            if 'standardized_country' not in str(e):
                print(f"Error getting region averages for {country_or_region}: {e}")
            return {}
    
    def _get_region_for_country(self, country: str) -> str:
        """Get the region for a country"""
        # Simplified region mapping - you should expand this
        country_to_region = {
            'USA': 'North America',
            'Canada': 'North America',
            'Mexico': 'North America',
            'Brazil': 'South America',
            'Colombia': 'South America',
            'Argentina': 'South America',
            'France': 'Western Europe',
            'Germany': 'Western Europe',
            'United Kingdom': 'Western Europe',
            'Russia': 'Eastern Europe',
            'China': 'East Asia',
            'Japan': 'East Asia',
            'India': 'South Asia',
            'Iraq': 'Middle East & North Africa',
            'Iran': 'Middle East & North Africa',
            'Egypt': 'Middle East & North Africa',
            'Nigeria': 'Sub-Saharan Africa',
            'Somalia': 'Sub-Saharan Africa',
            'South Africa': 'Sub-Saharan Africa',
            'Australia': 'Oceania'
        }
        
        return country_to_region.get(country, '')
    
    def _get_region_countries(self, region: str) -> List[str]:
        """Map region names to country lists"""
        # This is a simplified mapping - you should expand this
        region_mapping = {
            'Middle East & North Africa': ['Iraq', 'Syria', 'Iran', 'Saudi Arabia', 'Egypt', 'Libya', 'Yemen', 'Israel', 'United Arab Emirates', 'Jordan', 'Lebanon', 'Qatar', 'Kuwait'],
            'Sub-Saharan Africa': ['Nigeria', 'Somalia', 'South Africa', 'Burkina Faso', 'Mali', 'Niger', 'Kenya', 'DR Congo', 'Ethiopia', 'Tanzania', 'Uganda'],
            'South Asia': ['Afghanistan', 'Pakistan', 'India', 'Bangladesh', 'Sri Lanka', 'Nepal'],
            'Western Europe': ['France', 'United Kingdom', 'Germany', 'Italy', 'Spain', 'Netherlands', 'Belgium', 'Switzerland', 'Austria', 'Ireland', 'Iceland'],
            'Eastern Europe': ['Russia', 'Ukraine', 'Poland', 'Romania', 'Hungary', 'Czech Republic', 'Bulgaria', 'Serbia', 'Greece', 'Turkey'],
            'North America': ['USA', 'Canada', 'Mexico'],
            'Central America & Caribbean': ['Guatemala', 'El Salvador', 'Honduras', 'Nicaragua', 'Costa Rica', 'Panama', 'Cuba', 'Jamaica', 'Haiti', 'Dominican Republic'],
            'South America': ['Brazil', 'Colombia', 'Argentina', 'Venezuela', 'Peru', 'Chile', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay'],
            'East Asia': ['China', 'Japan', 'South Korea', 'North Korea', 'Taiwan', 'Mongolia'],
            'Southeast Asia': ['Indonesia', 'Philippines', 'Vietnam', 'Thailand', 'Myanmar', 'Malaysia', 'Singapore', 'Cambodia', 'Laos'],
            'Oceania': ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji']
        }
        return region_mapping.get(region, []) 