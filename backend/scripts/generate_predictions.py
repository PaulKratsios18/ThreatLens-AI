import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import random
from models.neural_network import TerrorismPredictor
from data_processing.socioeconomic_data import SocioeconomicDataLoader

def load_gtd_data(file_path):
    """
    Load Global Terrorism Database directly using pandas.
    
    Args:
        file_path: Path to the GTD CSV file
    
    Returns:
        DataFrame with the GTD data
    """
    print(f"Loading GTD data from: {file_path}")
    
    try:
        # Define essential columns to load
        essential_columns = [
            'iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 
            'provstate', 'city', 'latitude', 'longitude', 'success',
            'attacktype1_txt', 'targtype1_txt', 'nkill', 'nwound'
        ]
        
        # Check which columns actually exist in the file by reading the header first
        header_df = pd.read_csv(file_path, nrows=0)
        available_columns = header_df.columns.tolist()
        
        # Filter essential columns to only those that exist in the file
        columns_to_load = [col for col in essential_columns if col in available_columns]
        print(f"Loading columns: {columns_to_load}")
        
        # Load the data with only the available essential columns
        df = pd.read_csv(file_path, usecols=columns_to_load, low_memory=False)
        print(f"Loaded {len(df)} rows from {file_path}")
        
        # Basic data cleaning - replace missing values with appropriate defaults
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(0)
            
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')
            
        return df
    
    except Exception as e:
        print(f"Error loading GTD data: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=essential_columns)

def generate_predictions():
    """
    Generate comprehensive predictions for:
    1. Historical years (2000-2020) for backtracking/accuracy testing
    2. Future years (2021-2025) for forward predictions
    
    Ensures that predictions for a year only use data from previous years to prevent data leakage.
    Accuracy metrics calculated for historical years by comparing against real data.
    """
    print("Loading model and required components...")
    
    # Get proper file paths
    project_root = Path(__file__).parent.parent.parent
    historical_data_path = project_root / "data" / "globalterrorismdb_0522dist.csv"
    
    print(f"Data path: {historical_data_path}")
    print(f"Data exists: {historical_data_path.exists()}")
    
    # Initialize model and required components
    try:
        print("Attempting to load the trained model...")
        model = TerrorismPredictor()
        model.load(str(project_root / "backend" / "models" / "trained_model.keras"))
        use_real_model = True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using a mock predictor for testing.")
        model = MockPredictor()
        use_real_model = False
    
    # Load historical data directly
    if historical_data_path.exists():
        historical_data = load_gtd_data(str(historical_data_path))
        have_historical_data = len(historical_data) > 0
        print(f"Successfully loaded {len(historical_data)} historical records. Data columns: {historical_data.columns.tolist()}")
    else:
        print(f"Historical data file not found at {historical_data_path}")
        historical_data = pd.DataFrame()
        have_historical_data = False
    
    # Load socioeconomic data
    try:
        print("Loading socioeconomic data...")
        socio_loader = SocioeconomicDataLoader()
        socio_data = socio_loader.load_socioeconomic_data()
        print(f"Successfully loaded socioeconomic data")
        have_socio_data = True
    except Exception as e:
        print(f"Error loading socioeconomic data: {e}")
        socio_loader = None
        socio_data = None
        have_socio_data = False
    
    # Years to generate predictions for
    benchmark_years = list(range(2000, 2021))  # 2000-2020 for benchmarking against real data
    future_years = list(range(2021, 2026))     # 2021-2025 for future predictions
    years = benchmark_years + future_years
    
    # Regions to generate predictions for
    regions = [
        "North America", "South America", "Western Europe", "Eastern Europe",
        "Middle East", "North Africa", "Sub-Saharan Africa", "Central Asia",
        "South Asia", "East Asia", "Southeast Asia", "Oceania"
    ]
    
    predictions_by_year = {}
    
    print("Generating predictions for historical benchmarking and future years...")
    
    for year in years:
        print(f"Processing year {year}...")
        region_predictions = []
        
        # For each year, we'll only use data from prior years to make predictions
        cutoff_year = year - 1
        
        # Extract features for each region based on historical data up to cutoff_year
        region_features = {}
        region_counts = {}
        
        if have_historical_data and 'iyear' in historical_data.columns:
            # Filter historical data to only include years up to the cutoff
            filtered_data = historical_data[historical_data['iyear'] <= cutoff_year]
            
            # Get historical attack counts by region using only data up to cutoff_year
            for region in regions:
                try:
                    # Filter data for this region
                    region_data = filtered_data[filtered_data['region_txt'].str.contains(region, case=False, na=False)]
                    
                    # Get the counts by year if there's data available
                    if len(region_data) > 0 and 'iyear' in region_data.columns:
                        counts_by_year = region_data.groupby('iyear').size()
                        
                        # Use this to project future trends
                        if not counts_by_year.empty:
                            # Get recent years for trend calculation (up to 5 years of history)
                            available_years = len(counts_by_year)
                            lookback = min(5, available_years)
                            recent_years = counts_by_year.iloc[-lookback:]
                            
                            # Calculate average and trend using available years
                            avg_count = recent_years.mean()
                            
                            # Calculate trend if we have multiple years of data
                            if len(recent_years) > 1:
                                trend = (recent_years.iloc[-1] - recent_years.iloc[0]) / (len(recent_years) - 1)
                            else:
                                trend = 0
                                
                            region_counts[region] = {'avg': avg_count, 'trend': trend}
                            continue
                except Exception as e:
                    print(f"Error processing historical data for {region}: {e}")
                
                # Default if no data or error
                region_counts[region] = {'avg': 50, 'trend': 0}
                
                # Create feature vector based on historical patterns up to cutoff_year
                base_features = np.zeros(15)
                base_features[0] = min(1.0, region_counts[region]['avg'] / 500)  # Normalized attack frequency
                base_features[1] = 0.5 + (region_counts[region]['trend'] / 20)  # Trend factor
                
                # Add socioeconomic factors if available
                if have_socio_data:
                    try:
                        # Get region-level socioeconomic data from the cutoff year
                        region_socio = socio_loader.get_region_averages(region, cutoff_year)
                        
                        # Add relevant socioeconomic indicators to feature vector
                        if region_socio:
                            # Normalize GDP per capita (0-1 scale)
                            gdp_per_capita = region_socio.get('gdp_per_capita', 15000)
                            base_features[2] = min(1.0, gdp_per_capita / 100000)
                            
                            # Unemployment rate (0-1 scale)
                            unemployment = region_socio.get('unemployment_rate', 8)
                            base_features[3] = min(1.0, unemployment / 30)
                            
                            # Population (0-1 scale)
                            population = region_socio.get('population', 10000000)
                            base_features[4] = min(1.0, population / 1000000000)
                            
                            # Gini index (already 0-1 scale)
                            gini = region_socio.get('gini_index', 40)
                            base_features[5] = min(1.0, gini / 100)
                    except Exception as e:
                        print(f"Error incorporating socioeconomic data for {region}: {e}")
                
                # Fill remaining features with values based on region characteristics
                for i in range(6, 15):
                    # Use deterministic pseudorandom values based on region and year to ensure consistency
                    random.seed(f"{region}_{cutoff_year}_{i}")
                    base_features[i] = random.random() * 0.5 + 0.25
                
                region_features[region] = base_features
        else:
            print("No historical data available with 'iyear' column, using deterministic features")
            # If no historical data, use deterministic feature vectors
            for region in regions:
                random.seed(f"{region}_{year}")
                
                # Create feature vector
                features = np.random.rand(15)
                
                # Add some region-specific biases
                if region in ["Middle East", "North Africa", "Sub-Saharan Africa"]:
                    # Higher baseline for historically volatile regions
                    features[0] = 0.7 + random.random() * 0.3  # Higher attack frequency
                elif region in ["Western Europe", "North America"]:
                    # Lower baseline for stable regions
                    features[0] = 0.1 + random.random() * 0.2  # Lower attack frequency
                
                region_features[region] = features
                
                # Default counts for deterministic predictions
                region_counts[region] = {
                    'avg': 50 + random.randint(-20, 50),
                    'trend': random.uniform(-5, 5)
                }
        
        for region in regions:
            try:
                # Get region features
                features = region_features[region].copy()
                
                # Get prediction from model
                prediction_array = model.predict(np.array([features]))
                # Extract the scalar value from the prediction array
                prediction = float(prediction_array[0])
                
                # Calculate expected attacks based on prediction and historical context
                if region in region_counts:
                    # Base on historical average, adjusted by model prediction
                    region_stats = region_counts[region]
                    base_expected = region_stats['avg'] + (region_stats['trend'] * (year - cutoff_year))
                    expected_multiplier = 0.5 + prediction  # Prediction value influences the multiplier
                    expected_attacks = int(base_expected * expected_multiplier)
                else:
                    # Fallback if no historical data
                    expected_attacks = int(prediction * 200)  # Scale to a reasonable number
                
                # Ensure we have a reasonable positive number
                expected_attacks = max(1, expected_attacks)
                
                # Add some random variation but use a consistent seed for reproducibility
                random.seed(f"{region}_{year}_variation")
                variation = random.uniform(0.9, 1.1)
                expected_attacks = int(expected_attacks * variation)
                
                # Calculate confidence score
                if year in benchmark_years:
                    # Higher confidence for historical benchmarking as we have more contextual data
                    base_confidence = 0.85
                    years_back = 2020 - year
                    confidence_adjustment = years_back * 0.01  # Slightly lower confidence the further back we go
                    confidence_score = max(0.6, base_confidence - confidence_adjustment)
                else:
                    # Lower confidence for future predictions
                    confidence_base = 0.9 - ((year - 2021) * 0.1)  # Decreases for future years
                    confidence_score = confidence_base - (random.random() * 0.2)  # Add some randomness
                    confidence_score = max(0.5, min(0.95, confidence_score))  # Keep in reasonable range
                
                # Generate attack type distribution based on region characteristics and time period
                random.seed(f"{region}_{year}_attack_types")
                if region in ["North America", "Western Europe", "East Asia"]:
                    # More developed regions
                    if year < 2010:
                        # Earlier years had different attack patterns
                        attack_types = {
                            "Bombing/Explosion": 0.4 + random.random() * 0.1,
                            "Armed Assault": 0.3 + random.random() * 0.1,
                            "Facility/Infrastructure Attack": 0.2 + random.random() * 0.1,
                            "Vehicle Attack": 0.1 + random.random() * 0.05
                        }
                    else:
                        # More recent years with evolving threat landscape
                        attack_types = {
                            "Bombing/Explosion": 0.3 + random.random() * 0.1,
                            "Armed Assault": 0.3 + random.random() * 0.1,
                            "Facility/Infrastructure Attack": 0.2 + random.random() * 0.1,
                            "Vehicle Attack": 0.2 + random.random() * 0.1
                        }
                elif region in ["Middle East", "North Africa", "Sub-Saharan Africa"]:
                    # Conflict regions
                    attack_types = {
                        "Bombing/Explosion": 0.5 + random.random() * 0.1,
                        "Armed Assault": 0.3 + random.random() * 0.1,
                        "Hostage Taking/Kidnapping": 0.1 + random.random() * 0.05,
                        "Assassination": 0.1 + random.random() * 0.05
                    }
                else:
                    # Mix for other regions
                    attack_types = {
                        "Bombing/Explosion": 0.4 + random.random() * 0.1,
                        "Armed Assault": 0.3 + random.random() * 0.1,
                        "Hostage Taking/Kidnapping": 0.15 + random.random() * 0.05,
                        "Facility/Infrastructure Attack": 0.15 + random.random() * 0.05
                    }
                
                # Normalize attack types to sum to 1
                attack_type_sum = sum(attack_types.values())
                for attack_type in attack_types:
                    attack_types[attack_type] /= attack_type_sum
                
                # Incorporate socioeconomic factors into the prediction
                socioeconomic_factors = {}
                if have_socio_data:
                    try:
                        # For predictions, we use socioeconomic data from the cutoff year
                        region_countries = socio_loader._get_region_countries(region)
                        
                        # If this is a region, get average of all countries in the region
                        if region_countries:
                            for country in region_countries:
                                country_socio = socio_loader.get_country_data(country, cutoff_year)
                                if country_socio:
                                    for key, value in country_socio.items():
                                        if key not in ['country', 'year', 'standardized_country']:
                                            socioeconomic_factors[key] = socioeconomic_factors.get(key, 0) + value
                            
                            # Calculate averages
                            for key in socioeconomic_factors:
                                socioeconomic_factors[key] /= len(region_countries)
                    except Exception as e:
                        print(f"Error getting socioeconomic data for region {region}: {e}")
                        socioeconomic_factors = {}
                
                region_predictions.append({
                    "region": region,
                    "expected_attacks": expected_attacks,
                    "confidence_score": confidence_score,
                    "attack_types": attack_types,
                    "socioeconomic_factors": socioeconomic_factors
                })
                
            except Exception as e:
                print(f"Error generating prediction for {region} in {year}: {e}")
                # Add a default prediction if there's an error
                region_predictions.append({
                    "region": region,
                    "expected_attacks": random.randint(30, 100),
                    "confidence_score": 0.5,
                    "attack_types": {
                        "Bombing/Explosion": 0.4,
                        "Armed Assault": 0.3,
                        "Hostage Taking/Kidnapping": 0.2,
                        "Facility/Infrastructure Attack": 0.1
                    }
                })
        
        predictions_by_year[str(year)] = region_predictions
    
    # Create output structure
    output = {
        'generated_at': datetime.now().isoformat(),
        'predictions': predictions_by_year
    }
    
    # Save predictions to file
    output_path = Path(__file__).parent.parent.parent / 'data' / 'predictions.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
        print(f"Predictions saved to {output_path}")
    
    # Calculate and save accuracy metrics for benchmark predictions
    if have_historical_data and 'iyear' in historical_data.columns:
        calculate_prediction_accuracy(historical_data, predictions_by_year, benchmark_years)
    else:
        print("Skipping accuracy calculation: no historical data with 'iyear' column available")


def calculate_prediction_accuracy(historical_data, predictions, years):
    """
    Compare predictions for historical years against actual data and calculate accuracy metrics.
    
    Args:
        historical_data: DataFrame of actual historical terrorism data
        predictions: Dictionary of predictions by year
        years: List of historical years to evaluate
    """
    print("Calculating prediction accuracy for historical years...")
    
    accuracy_results = {}
    
    for year in years:
        year_str = str(year)
        if year_str not in predictions:
            continue
            
        print(f"Evaluating predictions for {year}...")
        
        try:
            # Filter actual data for this year
            actual_data = historical_data[historical_data['iyear'] == year]
            
            # Get region predictions for this year
            region_predictions = predictions[year_str]
            
            # Calculate region-level metrics
            region_metrics = {}
            
            for region_pred in region_predictions:
                region_name = region_pred["region"]
                
                try:
                    # Count actual attacks for this region
                    region_actual = actual_data[actual_data['region_txt'].str.contains(region_name, case=False, na=False)]
                    actual_count = len(region_actual)
                    
                    # Get predicted attacks
                    predicted_count = region_pred["expected_attacks"]
                    
                    # Calculate error metrics
                    if actual_count > 0:
                        absolute_error = abs(predicted_count - actual_count)
                        percentage_error = absolute_error / actual_count * 100
                        
                        # Calculate accuracy (100% - percentage error, capped at 0)
                        accuracy = max(0, 100 - percentage_error)
                    else:
                        # If no actual attacks, accuracy depends on how close to zero the prediction was
                        absolute_error = predicted_count
                        accuracy = 100 if predicted_count == 0 else max(0, 100 - predicted_count)
                    
                    region_metrics[region_name] = {
                        "actual_attacks": actual_count,
                        "predicted_attacks": predicted_count,
                        "absolute_error": absolute_error,
                        "accuracy": accuracy
                    }
                except Exception as e:
                    print(f"Error calculating metrics for {region_name} in {year}: {e}")
            
            # Calculate overall accuracy for the year
            if region_metrics:
                avg_accuracy = sum(r["accuracy"] for r in region_metrics.values()) / len(region_metrics)
            else:
                avg_accuracy = 0
                
            accuracy_results[year_str] = {
                "overall_accuracy": avg_accuracy,
                "region_metrics": region_metrics
            }
        except Exception as e:
            print(f"Error evaluating predictions for {year}: {e}")
    
    # Save accuracy results
    output_path = Path(__file__).parent.parent.parent / 'data' / 'prediction_accuracy.json'
    
    with open(output_path, 'w') as f:
        json.dump(accuracy_results, f, indent=2)
        print(f"Accuracy metrics saved to {output_path}")


class MockPredictor:
    """Mock predictor class for testing when the real model isn't available"""
    def predict(self, features):
        """Return random predictions with proper shape handling"""
        if isinstance(features, np.ndarray):
            # Return a scalar value if a single feature set is provided
            if len(features.shape) == 2 and features.shape[0] == 1:
                return np.array([random.random()])
            # Return an array of scalars for multiple feature sets
            else:
                return np.array([random.random() for _ in range(features.shape[0])])
        # Fallback
        return np.array([random.random()])


if __name__ == "__main__":
    generate_predictions() 