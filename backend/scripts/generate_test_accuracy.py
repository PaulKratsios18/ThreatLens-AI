"""
Script to generate sample prediction accuracy data for testing the feedback optimizer.
This creates a file with historical benchmark data comparing predicted vs actual terrorism incidents by region.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import os
import random
from datetime import datetime
import numpy as np

def generate_test_accuracy_data():
    """
    Generate sample prediction accuracy data for benchmark years (2000-2020).
    This creates realistic but synthetic data for testing the feedback optimization system.
    """
    print("Generating test prediction accuracy data...")
    
    # Years to generate benchmark data for
    benchmark_years = list(range(2000, 2021))  # 2000-2020
    
    # Regions to generate prediction accuracy for
    regions = [
        "North America", "South America", "Western Europe", "Eastern Europe",
        "Middle East", "North Africa", "Sub-Saharan Africa", "Central Asia",
        "South Asia", "East Asia", "Southeast Asia", "Oceania"
    ]
    
    # Region-specific bias factors to simulate systematic prediction errors
    # Values > 1 mean we consistently under-predict for this region
    # Values < 1 mean we consistently over-predict for this region
    region_bias = {
        "North America": 0.85,  # We tend to over-predict attacks in North America
        "South America": 1.2,   # We tend to under-predict attacks in South America
        "Western Europe": 0.9,
        "Eastern Europe": 1.05,
        "Middle East": 1.3,     # Significant under-prediction in Middle East
        "North Africa": 1.15,
        "Sub-Saharan Africa": 1.25,
        "Central Asia": 0.95,
        "South Asia": 1.15,
        "East Asia": 0.8,       # Significant over-prediction in East Asia
        "Southeast Asia": 1.1,
        "Oceania": 0.7          # Significant over-prediction in Oceania
    }
    
    # Generate accuracy data for each year
    accuracy_results = {}
    
    for year in benchmark_years:
        print(f"Generating accuracy data for {year}...")
        
        # Calculate region-level metrics for this year
        region_metrics = {}
        
        for region in regions:
            # Generate synthetic "actual" attack counts for this region and year
            # Based on realistic patterns by region
            base_count = get_base_count_for_region(region, year)
            
            # Add yearly fluctuation
            yearly_fluctuation = np.random.normal(0, base_count * 0.1)  # 10% standard deviation
            actual_count = max(0, int(base_count + yearly_fluctuation))
            
            # Generate "predicted" count by applying the region's bias factor
            # and adding random prediction error
            bias_factor = region_bias[region]
            random_error = np.random.normal(0, actual_count * 0.15)  # 15% standard deviation
            
            # If bias factor < 1, we over-predict
            # If bias factor > 1, we under-predict
            predicted_count = max(0, int((actual_count / bias_factor) + random_error))
            
            # Calculate metrics
            if actual_count > 0:
                absolute_error = abs(predicted_count - actual_count)
                percentage_error = (absolute_error / actual_count) * 100
                accuracy = max(0, 100 - percentage_error)
            else:
                # If no actual attacks, accuracy depends on how close to zero the prediction was
                absolute_error = predicted_count
                accuracy = 100 if predicted_count == 0 else max(0, 100 - predicted_count)
            
            # Store metrics for this region
            region_metrics[region] = {
                "actual_attacks": actual_count,
                "predicted_attacks": predicted_count,
                "absolute_error": absolute_error,
                "accuracy": accuracy
            }
        
        # Calculate overall accuracy for the year
        if region_metrics:
            avg_accuracy = sum(r["accuracy"] for r in region_metrics.values()) / len(region_metrics)
        else:
            avg_accuracy = 0
            
        # Store results for this year
        accuracy_results[str(year)] = {
            "overall_accuracy": avg_accuracy,
            "region_metrics": region_metrics
        }
    
    # Save accuracy results
    output_path = Path(__file__).parent.parent.parent / 'data' / 'prediction_accuracy.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(accuracy_results, f, indent=2)
        print(f"Test accuracy metrics saved to {output_path}")


def get_base_count_for_region(region, year):
    """
    Return a realistic base count of terrorism incidents for a region in a specific year.
    Based on general historical patterns.
    """
    # Base counts by region (approximate annual incidents around 2010)
    base_counts = {
        "North America": 15,
        "South America": 80,
        "Western Europe": 40,
        "Eastern Europe": 60,
        "Middle East": 800,
        "North Africa": 120,
        "Sub-Saharan Africa": 700,
        "Central Asia": 250,
        "South Asia": 900,
        "East Asia": 20,
        "Southeast Asia": 300,
        "Oceania": 5
    }
    
    # Apply time-based trends
    base = base_counts.get(region, 50)
    
    # Pre-2001 had generally lower levels
    if year < 2001:
        base = base * 0.6
    # 2001-2006 saw increases in certain regions
    elif year <= 2006:
        if region in ["Middle East", "South Asia", "Central Asia"]:
            base = base * 1.2
    # 2007-2014 saw significant increases in conflict zones
    elif year <= 2014:
        if region in ["Middle East", "South Asia", "Sub-Saharan Africa"]:
            base = base * 1.5
        elif region in ["North Africa", "Central Asia"]:
            base = base * 1.3
    # 2015-2020 saw declines in some areas but increases in others
    else:
        if region in ["Middle East"]:
            base = base * 0.9
        elif region in ["Sub-Saharan Africa"]:
            base = base * 1.3
    
    # Add some randomization
    random.seed(f"{region}_{year}_base")
    randomization = random.uniform(0.85, 1.15)
    
    return int(base * randomization)


if __name__ == "__main__":
    generate_test_accuracy_data() 