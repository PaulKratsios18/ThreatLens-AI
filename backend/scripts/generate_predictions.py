import os
import sys
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Add the parent directory to sys.path to import from data_processing
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir)
print(f"Added to Python path: {parent_dir}")

# Import local modules
from data_processing.data_loader import GTDDataLoader

# Define simplified versions of the classes we need
class Preprocessor:
    """Simplified preprocessor for the prediction script."""
    
    def process(self, data):
        """Process the data for prediction."""
        print("Preprocessing data...")
        return data

class FeatureEngineer:
    """Simplified feature engineer for the prediction script."""
    
    def engineer_features(self, data):
        """Engineer features for prediction."""
        print("Engineering features...")
        return data

class Predictor:
    """Simplified predictor for the prediction script."""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, features):
        """Make a prediction using the model."""
        return self.model.predict(features)

class MockPredictor:
    """Mock predictor for the prediction script."""
    
    def predict(self, features):
        """Return a random prediction factor."""
        print("Using MockPredictor - returning random prediction factor")
        return random.uniform(0.8, 1.2)

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

def train_and_predict_for_year(training_data, region_data, target_year, force_use_mock=False):
    """
    Train a model on historical data and make a prediction for the target year.
    
    Args:
        training_data: DataFrame with data prior to target_year
        region_data: DataFrame with data for the specific region
        target_year: Year to predict
        force_use_mock: Whether to use mock predictor
        
    Returns:
        prediction_factor: Factor to multiply historical average by
    """
    if force_use_mock:
        return random.uniform(0.8, 1.2)
    
    try:
        # Preprocess training data
        preprocessor = Preprocessor()
        feature_engineer = FeatureEngineer()
        
        processed_data = preprocessor.process(training_data)
        features = feature_engineer.engineer_features(processed_data)
        
        # Get numeric features only
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Skip if no numeric features
        if len(numeric_features.columns) == 0:
            print(f"  No numeric features available for year {target_year}")
            return 1.0
        
        # Create a target - more recent incidents have higher risk
        years_column = numeric_features['iyear']
        # Normalize years to [0, 1] range for risk assessment
        years_norm = (years_column - years_column.min()) / (years_column.max() - years_column.min() + 1e-10)
        # Fill missing values
        numeric_features = numeric_features.fillna(0)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(numeric_features, years_norm)
        
        # Filter features for the specific region
        region_features = features[features['region_txt'] == region_data['region_txt'].iloc[0]]
        
        # Get most recent data as a baseline for prediction
        recent_years = [y for y in range(target_year - 5, target_year)]
        region_recent = region_features[region_features['iyear'].isin(recent_years)]
        
        if len(region_recent) == 0:
            region_recent = region_features
        
        # Get numeric features for prediction
        predict_features = region_recent.select_dtypes(include=[np.number])
        predict_features = predict_features.fillna(0)
        
        # Make prediction
        if len(predict_features) > 0:
            preds = model.predict(predict_features)
            # Scale predictions to reasonable factor range (0.7 - 1.3)
            factors = 0.7 + (preds * 0.6)
            # Return mean factor
            factor = float(np.mean(factors))
            factor = min(1.5, max(0.7, factor))
            
            print(f"  Model prediction factor for {target_year}: {factor:.4f}")
            return factor
        else:
            print(f"  No prediction features for {target_year}, using default factor")
            return 1.0
    except Exception as e:
        print(f"  Error training model for year {target_year}: {e}")
        return 1.0

def train_and_predict_for_country(training_data, country_data, target_year, force_use_mock=False):
    """
    Train a model on historical country data and make a prediction for the target year.
    
    Args:
        training_data: DataFrame with data prior to target_year
        country_data: DataFrame with data for the specific country
        target_year: Year to predict
        force_use_mock: Whether to use mock predictor
        
    Returns:
        prediction_factor: Factor to multiply historical average by
    """
    if force_use_mock:
        return random.uniform(0.8, 1.2)
    
    try:
        # Preprocess training data
        preprocessor = Preprocessor()
        feature_engineer = FeatureEngineer()
        
        processed_data = preprocessor.process(training_data)
        features = feature_engineer.engineer_features(processed_data)
        
        # Get numeric features only
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Skip if no numeric features
        if len(numeric_features.columns) == 0:
            print(f"  No numeric features available for country in year {target_year}")
            return 1.0
        
        # Create a target - more recent incidents have higher risk
        years_column = numeric_features['iyear']
        # Normalize years to [0, 1] range for risk assessment
        years_norm = (years_column - years_column.min()) / (years_column.max() - years_column.min() + 1e-10)
        # Fill missing values
        numeric_features = numeric_features.fillna(0)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(numeric_features, years_norm)
        
        # Filter features for the specific country
        country_features = features[features['country_txt'] == country_data['country_txt'].iloc[0]]
        
        # Get most recent data as a baseline for prediction
        recent_years = [y for y in range(target_year - 5, target_year)]
        country_recent = country_features[country_features['iyear'].isin(recent_years)]
        
        if len(country_recent) == 0:
            country_recent = country_features
        
        # Get numeric features for prediction
        predict_features = country_recent.select_dtypes(include=[np.number])
        predict_features = predict_features.fillna(0)
        
        # Make prediction
        if len(predict_features) > 0:
            preds = model.predict(predict_features)
            # Scale predictions to reasonable factor range (0.7 - 1.3)
            factors = 0.7 + (preds * 0.6)
            # Return mean factor
            factor = float(np.mean(factors))
            factor = min(1.5, max(0.7, factor))
            
            print(f"  Country prediction factor for {target_year}: {factor:.4f}")
            return factor
        else:
            print(f"  No prediction features for country in {target_year}, using default factor")
            return 1.0
    except Exception as e:
        print(f"  Error training model for country in year {target_year}: {e}")
        return 1.0

def calculate_benchmark_statistics(predictions, output_path):
    """
    Calculate benchmark statistics comparing predictions to actual data.
    
    Args:
        predictions: Dictionary of all predictions
        output_path: Path to save benchmark statistics
    """
    print("\nCalculating benchmark statistics...")
    
    benchmark = {
        "overall": {},
        "by_year": {},
        "by_region": {}
    }
    
    # Collect all prediction-actual pairs
    all_pairs = []
    for region in predictions["regions"]:
        for pred in region["predictions"]:
            if "actual_attacks" in pred:
                all_pairs.append({
                    "year": pred["year"],
                    "region": region["region"],
                    "expected": pred["expected_attacks"],
                    "actual": pred["actual_attacks"],
                    "accuracy": pred.get("accuracy", 0)
                })
    
    # Calculate overall metrics
    if all_pairs:
        avg_accuracy = sum(p["accuracy"] for p in all_pairs if "accuracy" in p) / len([p for p in all_pairs if "accuracy" in p])
        total_expected = sum(p["expected"] for p in all_pairs)
        total_actual = sum(p["actual"] for p in all_pairs)
        
        benchmark["overall"] = {
            "average_accuracy_percent": round(avg_accuracy, 1),
            "total_predicted_attacks": round(total_expected),
            "total_actual_attacks": total_actual,
            "prediction_ratio": round(total_expected / total_actual, 2) if total_actual > 0 else None
        }
    
    # Calculate metrics by year
    years = sorted(set(p["year"] for p in all_pairs))
    for year in years:
        year_pairs = [p for p in all_pairs if p["year"] == year]
        if year_pairs:
            year_accuracy = sum(p["accuracy"] for p in year_pairs if "accuracy" in p) / len([p for p in year_pairs if "accuracy" in p])
            year_expected = sum(p["expected"] for p in year_pairs)
            year_actual = sum(p["actual"] for p in year_pairs)
            
            benchmark["by_year"][str(year)] = {
                "average_accuracy_percent": round(year_accuracy, 1),
                "total_predicted_attacks": round(year_expected),
                "total_actual_attacks": year_actual,
                "prediction_ratio": round(year_expected / year_actual, 2) if year_actual > 0 else None
            }
    
    # Calculate metrics by region
    regions = sorted(set(p["region"] for p in all_pairs))
    for region in regions:
        region_pairs = [p for p in all_pairs if p["region"] == region]
        if region_pairs:
            region_accuracy = sum(p["accuracy"] for p in region_pairs if "accuracy" in p) / len([p for p in region_pairs if "accuracy" in p])
            region_expected = sum(p["expected"] for p in region_pairs)
            region_actual = sum(p["actual"] for p in region_pairs)
            
            benchmark["by_region"][region] = {
                "average_accuracy_percent": round(region_accuracy, 1),
                "total_predicted_attacks": round(region_expected),
                "total_actual_attacks": region_actual,
                "prediction_ratio": round(region_expected / region_actual, 2) if region_actual > 0 else None
            }
    
    # Save benchmark to file
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"Benchmark statistics saved to {output_path}")

def generate_predictions(
    output_path='../data/predictions.json',
    force_use_mock=False
):
    """
    Generate predictions for terrorist attacks for years 2000-2025.
    For each year, train a model using only data preceding that year.
    Compare predictions with actual data for 2000-2021 for benchmarking.
    
    Args:
        output_path: Path to save the predictions JSON file
        force_use_mock: Force the use of the mock predictor instead of a real model
    
    Returns:
        Boolean indicating success or failure
    """
    print("Generating predictions for years 2000-2025...")
    
    # Define global risk thresholds
    global_thresholds = {
        "low": 1.0,     # Very Low -> Low threshold
        "medium": 15.0,  # Low -> Medium threshold
        "high": 40.0    # Medium -> High threshold
    }
    
    # Path to the GTD data file
    data_file_path = Path(__file__).parent.parent.parent / 'data' / 'globalterrorismdb_0522dist.csv'
    
    # Check if data file exists
    if not os.path.exists(data_file_path):
        # Try to find the data file in other common locations
        potential_paths = [
            Path(__file__).parent.parent / 'data' / 'globalterrorismdb_0522dist.csv',
            Path.cwd() / 'data' / 'globalterrorismdb_0522dist.csv',
            Path(__file__).parent.parent / 'database' / 'data' / 'globalterrorismdb_0522dist.csv'
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                data_file_path = path
                break
        else:
            print(f"Error: Data file not found at {data_file_path} or common alternatives")
            return False
    
    print(f"Using data file at: {data_file_path}")
    
    # Try to load risk mappings
    risk_mappings_path = Path(__file__).parent.parent.parent / 'data' / 'risk_mappings.json'
    region_risk_multipliers = {}
    country_risk_mappings = {}
    
    try:
        if os.path.exists(risk_mappings_path):
            print(f"Loading risk mappings from {risk_mappings_path}")
            with open(risk_mappings_path, 'r') as f:
                risk_mappings = json.load(f)
                region_risk_multipliers = risk_mappings.get('regions', {})
                country_risk_mappings = risk_mappings.get('countries', {})
            
            print(f"Loaded risk mappings with {len(region_risk_multipliers)} regions and {len(country_risk_mappings)} countries")
        else:
            print(f"Warning: Risk mappings file not found at {risk_mappings_path}")
            print("Using default risk thresholds")
    except Exception as e:
        print(f"Error loading risk mappings: {e}")
        print("Using default risk thresholds")
    
    # Load all data
    data_loader = GTDDataLoader()
    all_data = data_loader.load_data(path=str(data_file_path))
    
    if all_data is None or all_data.empty:
        print("Error: Failed to load data")
        return False
    
    # Check for required columns in the loaded data
    required_columns = ['iyear', 'country_txt', 'region_txt']
    missing_columns = [col for col in required_columns if col not in all_data.columns]
    if missing_columns:
        print(f"Error: Missing required columns in data: {missing_columns}")
        return False
    
    # Determine the range of years in the data
    min_year = all_data['iyear'].min()
    max_year = all_data['iyear'].max()
    print(f"Data spans years {min_year} to {max_year}")
    
    # Define forecast years (2000-2025)
    forecast_years = list(range(2000, 2026))
    
    # Filter out forecast years with insufficient data
    valid_forecast_years = []
    for year in forecast_years:
        # Need at least 3 years of historical data to train a model
        if year - min_year >= 3:
            valid_forecast_years.append(year)
        else:
            print(f"Skipping year {year} due to insufficient historical data")
    
    # Initialize results structure
    all_predictions = {
        "generated_at": datetime.now().isoformat(),
        "model_type": "RandomForestRegressor",
        "years_predicted": valid_forecast_years,
        "baseline_years": list(range(min_year, max_year + 1)),
        "regions": []
    }
    
    # Collect benchmark statistics for years where we have actual data
    benchmark_years = [year for year in valid_forecast_years if year <= max_year]
    all_benchmark_results = {
        "generated_at": datetime.now().isoformat(),
        "benchmark_years": benchmark_years,
        "overall_metrics": {},
        "yearly_metrics": {}
    }
    
    # Get unique regions
    unique_regions = all_data['region_txt'].unique()
    
    for region in unique_regions:
        print(f"\nProcessing region: {region}")
        region_data = all_data[all_data['region_txt'] == region]
        
        # Get countries in this region
        countries = region_data['country_txt'].unique()
        
        # Initialize region predictions
        region_predictions = {
            "region": region,
            "countries": [],
            "predictions": []
        }
        
        # Calculate historical average attacks per year for this region
        yearly_attacks = region_data.groupby('iyear').size()
        historical_avg = yearly_attacks.mean() if len(yearly_attacks) > 0 else 0
        region_predictions["historical_avg_attacks"] = float(historical_avg)
        
        # Initialize region year predictions dictionary to track aggregated country predictions
        region_year_predictions = {year: {"expected_attacks": 0, "actual_attacks": 0} for year in valid_forecast_years}
        
        # Process countries
        for country in countries:
            country_data = region_data[region_data['country_txt'] == country]
            
            # Calculate historical average attacks per year for this country
            country_yearly_attacks = country_data.groupby('iyear').size()
            country_historical_avg = country_yearly_attacks.mean() if len(country_yearly_attacks) > 0 else 0
            
            # Skip countries with negligible historical activity
            if country_historical_avg < 0.1 and len(countries) > 10:
                continue
            
            # Initialize country predictions
            country_predictions = {
                "country": country,
                "historical_avg_attacks": round(float(country_historical_avg), 1),
                "predictions": []
            }
            
            # Generate predictions for each year for this country
            for year in valid_forecast_years:
                print(f"  Generating prediction for {country} in year {year}")
                
                # For each prediction year, use only data from years before it
                training_data = all_data[all_data['iyear'] < year]
                
                # Skip years with insufficient training data
                if len(training_data) < 100:
                    print(f"  Insufficient training data for {country} in year {year}, using MockPredictor")
                    prediction_factor = random.uniform(0.9, 1.1)  # Random factor close to 1
                else:
                    # Train model on historical data for this country
                    prediction_factor = train_and_predict_for_country(training_data, country_data, year, force_use_mock)
                
                # Calculate actual attacks for this year if available (for benchmarking)
                actual_data = country_data[country_data['iyear'] == year]
                actual_attacks = len(actual_data) if not actual_data.empty else None
                
                # Adjust confidence based on distance from present year
                current_year = datetime.now().year
                if year <= max_year:
                    # For historical years with actual data
                    year_diff = current_year - year
                    confidence = max(0.7, 0.9 - (year_diff * 0.01))  # Slightly decreasing for older years
                else:
                    # Future predictions have decreasing confidence
                    year_diff = year - max_year
                    confidence = max(0.5, 0.9 - (year_diff * 0.1))
                
                # Calculate expected attacks for country
                # Use only data up to prediction year for calculating historical average
                historical_data_for_year = country_data[country_data['iyear'] < year]
                
                # Use recent years for average, but only those before the prediction year
                historical_lookback = min(5, year - min_year)
                recent_years = [y for y in range(year - historical_lookback, year) if y >= min_year]
                
                if recent_years:
                    recent_data = country_data[country_data['iyear'].isin(recent_years)]
                    recent_avg = len(recent_data) / len(recent_years) if recent_years else country_historical_avg
                else:
                    # If no recent years data is available, use all historical data up to prediction year
                    if not historical_data_for_year.empty:
                        yearly_attacks_for_year = historical_data_for_year.groupby('iyear').size()
                        recent_avg = yearly_attacks_for_year.mean() if len(yearly_attacks_for_year) > 0 else 0
                    else:
                        recent_avg = 0
                
                country_expected_attacks = recent_avg * prediction_factor
                
                # Determine country-specific risk level based on its expected attacks
                country_risk_multiplier = country_risk_mappings.get(country, 1.0)
                if isinstance(country_risk_multiplier, (list, tuple, dict, str)):
                    # Handle case where mapping returned a non-numeric value
                    country_risk_multiplier = 1.0
                
                # Use the global thresholds modified by country-specific multiplier
                country_low_threshold = global_thresholds["low"] * country_risk_multiplier
                country_medium_threshold = global_thresholds["medium"] * country_risk_multiplier
                country_high_threshold = global_thresholds["high"] * country_risk_multiplier
                
                if country_expected_attacks > country_high_threshold:
                    country_risk_level = "High"
                elif country_expected_attacks > country_medium_threshold:
                    country_risk_level = "Medium"
                elif country_expected_attacks > country_low_threshold:
                    country_risk_level = "Low"
                else:
                    country_risk_level = "Very Low"
                
                # Create country year prediction
                country_year_prediction = {
                    "year": year,
                    "expected_attacks": round(float(country_expected_attacks), 1),
                    "confidence": round(float(confidence), 2),
                    "risk_level": country_risk_level,
                    "prediction_factor": round(float(prediction_factor), 3)
                }
                
                # Add actual attacks for benchmarking if available
                if actual_attacks is not None and actual_attacks > 0:
                    country_year_prediction["actual_attacks"] = actual_attacks
                    
                    # Calculate accuracy metrics if actual data is available
                    country_error = abs(country_expected_attacks - actual_attacks)
                    country_relative_error = country_error / actual_attacks
                    country_accuracy = max(0, 1 - country_relative_error)
                    country_year_prediction["accuracy"] = round(float(country_accuracy * 100), 1)
                    
                    # Add to region year totals for actual attacks
                    region_year_predictions[year]["actual_attacks"] += actual_attacks
                
                # Add to region year totals for expected attacks
                region_year_predictions[year]["expected_attacks"] += country_expected_attacks
                
                # Add to country predictions
                country_predictions["predictions"].append(country_year_prediction)
            
            # Add country to region predictions
            region_predictions["countries"].append(country_predictions)
        
        # Now generate region-level predictions based on aggregated country data
        for year in valid_forecast_years:
            # Get the aggregated data for this year
            expected_attacks = region_year_predictions[year]["expected_attacks"]
            actual_attacks = region_year_predictions[year]["actual_attacks"] if year <= max_year else None
            
            # Adjust confidence based on distance from present year
            current_year = datetime.now().year
            if year <= max_year:
                # For historical years with actual data
                year_diff = current_year - year
                confidence = max(0.7, 0.9 - (year_diff * 0.01))  # Slightly decreasing for older years
            else:
                # Future predictions have decreasing confidence
                year_diff = year - max_year
                confidence = max(0.5, 0.9 - (year_diff * 0.1))
            
            # Apply region-specific risk multiplier if available
            risk_multiplier = region_risk_multipliers.get(region, 1.0)
            
            # Determine risk level using thresholds
            low_threshold = global_thresholds["low"] * risk_multiplier
            medium_threshold = global_thresholds["medium"] * risk_multiplier
            high_threshold = global_thresholds["high"] * risk_multiplier
            
            if expected_attacks > high_threshold:
                risk_level = "High"
            elif expected_attacks > medium_threshold:
                risk_level = "Medium"
            elif expected_attacks > low_threshold:
                risk_level = "Low"
            else:
                risk_level = "Very Low"
            
            # Create year prediction entry
            year_prediction = {
                "year": year,
                "expected_attacks": round(float(expected_attacks), 1),
                "confidence": round(float(confidence), 2),
                "risk_level": risk_level
            }
            
            # Add actual attacks for benchmarking if available
            if actual_attacks is not None and actual_attacks > 0:
                year_prediction["actual_attacks"] = actual_attacks
                
                # Calculate accuracy metrics if actual data is available
                error = abs(expected_attacks - actual_attacks)
                relative_error = error / actual_attacks
                accuracy = max(0, 1 - relative_error)
                year_prediction["accuracy"] = round(float(accuracy * 100), 1)
            
            # Add to region predictions
            region_predictions["predictions"].append(year_prediction)
        
        # Add region to all predictions
        all_predictions["regions"].append(region_predictions)
    
    # Process benchmark metrics
    for year in benchmark_years:
        year_benchmark = {
            "year": year,
            "total_actual": 0,
            "total_expected": 0,
            "accuracy_scores": [],
            "region_accuracies": {}
        }
        
        for region_data in all_predictions["regions"]:
            # Find prediction for this year
            for pred in region_data["predictions"]:
                if pred["year"] == year and "actual_attacks" in pred:
                    actual = pred["actual_attacks"]
                    expected = pred["expected_attacks"]
                    
                    year_benchmark["total_actual"] += actual
                    year_benchmark["total_expected"] += expected
                    
                    if "accuracy" in pred:
                        year_benchmark["accuracy_scores"].append(pred["accuracy"])
                        year_benchmark["region_accuracies"][region_data["region"]] = pred["accuracy"]
        
        # Calculate overall metrics for the year
        if year_benchmark["accuracy_scores"]:
            year_benchmark["average_accuracy"] = round(sum(year_benchmark["accuracy_scores"]) / len(year_benchmark["accuracy_scores"]), 1)
        else:
            year_benchmark["average_accuracy"] = 0
            
        # Calculate error at global level
        if year_benchmark["total_actual"] > 0:
            error = abs(year_benchmark["total_expected"] - year_benchmark["total_actual"])
            relative_error = error / year_benchmark["total_actual"]
            year_benchmark["global_accuracy"] = round((1 - relative_error) * 100, 1)
        else:
            year_benchmark["global_accuracy"] = 0
            
        all_benchmark_results["yearly_metrics"][str(year)] = year_benchmark
    
    # Calculate overall benchmark metrics across all years
    if benchmark_years:
        overall_accuracies = [metrics["global_accuracy"] for _, metrics in all_benchmark_results["yearly_metrics"].items()]
        all_benchmark_results["overall_metrics"]["average_accuracy"] = round(sum(overall_accuracies) / len(overall_accuracies), 1)
        
    # Save predictions to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
        print(f"Predictions saved to {output_path}")
    
    # Save benchmark results to file if we have benchmark data
    if benchmark_years:
        benchmark_output_path = output_path.replace('.json', '_benchmark.json')
        with open(benchmark_output_path, 'w') as f:
            json.dump(all_benchmark_results, f, indent=2)
        
        print(f"Benchmark results saved to {benchmark_output_path}")
    
    return True

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

class RealPredictor:
    """Adapter class to use the various model types with the prediction script."""
    
    def __init__(self, model):
        self.model = model
        # Check the model type for better handling
        self.model_type = type(model).__name__
        print(f"RealPredictor initialized with {self.model_type} model")
    
    def predict(self, features):
        """
        Make a prediction for the given features.
        
        Args:
            features: DataFrame with features for prediction
        
        Returns:
            Prediction factor for adjusting the historical average
        """
        try:
            print(f"Making prediction with {self.model_type} model on {len(features)} samples")
            
            # Ensure we're working with a DataFrame
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            
            # Filter out datetime columns and non-numeric columns that would cause issues
            numeric_features = features.select_dtypes(include=['number'])
            
            # Check if we have enough features
            if len(numeric_features.columns) == 0:
                print("No numeric features available for prediction")
                return 1.2
            
            # Fill NaN values with 0
            numeric_features = numeric_features.fillna(0)
            
            # If our model already has a predict method, use it directly
            if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                try:
                    # Use the model's predict method directly
                    # For RandomForestClassifier, use predict_proba and get the positive class probability
                    if self.model_type == 'RandomForestClassifier':
                        try:
                            # Try to get probabilities
                            probs = self.model.predict_proba(numeric_features)
                            # Use the probability of the positive class (usually the last column)
                            factor = 0.8 + (np.mean(probs[:, -1]) * 0.7)
                        except Exception as e:
                            print(f"Error using predict_proba: {e}")
                            # Fall back to simple prediction
                            preds = self.model.predict(numeric_features)
                            factor = 1.2  # Default factor
                    else:
                        # For other model types, use regular predict
                        factor = self.model.predict(numeric_features)
                    
                    # Make sure we return a float, not an array
                    if hasattr(factor, '__len__') and len(factor) > 0:
                        factor = float(np.mean(factor))
                    
                    # Ensure the result is in a reasonable range
                    factor = max(0.8, min(1.5, factor))
                    print(f"Final prediction factor: {factor:.4f}")
                    return factor
                except Exception as e:
                    print(f"Error in direct model prediction: {e}")
                    return 1.2  # Return a default factor
            else:
                # Should not get here since we already checked for predict method
                print(f"Model does not have a predict method")
                return 1.2
                
        except Exception as e:
            print(f"Error in RealPredictor.predict: {e}")
            import traceback
            traceback.print_exc()
            return 1.2  # Return a reasonable default prediction factor

class RFPredictor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, features):
        # Convert to dataframe if it's not already
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        
        # Get numeric features only
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(0)
        
        # Predict and scale to our factor range (0.8 - 1.5)
        try:
            preds = self.model.predict(numeric_features)
            # Scale predictions to range 0.8 - 1.5
            factors = 0.8 + (preds * 0.7)
            # Return mean factor
            mean_factor = float(np.mean(factors))
            mean_factor = min(1.5, max(0.8, mean_factor))
            print(f"RF model prediction: {mean_factor:.4f}")
            return mean_factor
        except Exception as e:
            print(f"Error in RF prediction: {e}")
            return 1.2  # Default if prediction fails

if __name__ == "__main__":
    # Generate predictions for years 2000-2025
    generate_predictions(force_use_mock=False)