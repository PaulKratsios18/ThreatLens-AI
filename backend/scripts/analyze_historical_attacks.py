import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

def analyze_historical_patterns():
    """
    Analyze historical terrorism data to determine data-driven risk thresholds.
    Calculates attack statistics by country and region, and exports the results
    for use in risk prediction algorithms.
    """
    print("Starting historical data analysis...")
    
    # Path to historical terrorism data
    project_root = Path(__file__).parent.parent.parent
    historical_data_path = project_root / "data" / "globalterrorismdb_0522dist.csv"
    
    if not historical_data_path.exists():
        print(f"Error: Historical data file not found at {historical_data_path}")
        return False
    
    print(f"Loading historical data from {historical_data_path}...")
    
    try:
        # Define essential columns to load
        essential_columns = [
            'iyear', 'imonth', 'country_txt', 'region_txt', 
            'attacktype1_txt', 'nkill', 'nwound'
        ]
        
        # Load only essential columns to conserve memory
        df = pd.read_csv(historical_data_path, usecols=essential_columns, low_memory=False)
        print(f"Loaded {len(df)} historical incident records")
        
        # Filter out rows with missing critical data
        df = df.dropna(subset=['iyear', 'country_txt', 'region_txt'])
        
        # Group by country and year to get attack frequencies
        country_year_counts = df.groupby(['country_txt', 'iyear']).size().reset_index(name='attacks')
        
        # Group by region and year
        region_year_counts = df.groupby(['region_txt', 'iyear']).size().reset_index(name='attacks')
        
        # Calculate overall statistics
        all_counts = country_year_counts['attacks'].values
        percentiles = [25, 50, 75, 90, 95, 99]
        overall_percentiles = {
            f"p{p}": np.percentile(all_counts, p) for p in percentiles
        }
        
        print(f"Overall attack frequency percentiles:")
        for p, value in overall_percentiles.items():
            print(f"  {p}: {value:.1f}")
        
        # Calculate region-specific statistics
        region_stats = {}
        for region in df['region_txt'].unique():
            region_data = country_year_counts[
                country_year_counts['country_txt'].isin(
                    df[df['region_txt'] == region]['country_txt'].unique()
                )
            ]
            
            if len(region_data) > 0:
                region_counts = region_data['attacks'].values
                region_stats[region] = {
                    "avg_attacks": float(np.mean(region_counts)),
                    "median_attacks": float(np.median(region_counts)),
                    "max_attacks": float(np.max(region_counts)),
                    "p75": float(np.percentile(region_counts, 75) if len(region_counts) > 0 else 0),
                    "p90": float(np.percentile(region_counts, 90) if len(region_counts) > 0 else 0),
                    "total_incidents": int(np.sum(region_counts)),
                    "countries": len(df[df['region_txt'] == region]['country_txt'].unique()),
                    "years_with_data": len(region_data['iyear'].unique())
                }
                
                # Calculate appropriate risk thresholds based on region-specific distribution
                region_multiplier = region_stats[region]["avg_attacks"] / overall_percentiles["p50"]
                region_multiplier = min(max(region_multiplier, 0.5), 2.0)  # Bound the multiplier
                
                region_stats[region]["risk_multiplier"] = float(region_multiplier)
                region_stats[region]["thresholds"] = {
                    "low": float(max(1, overall_percentiles["p25"] * region_multiplier)),
                    "medium": float(max(5, overall_percentiles["p75"] * region_multiplier)),
                    "high": float(max(20, overall_percentiles["p90"] * region_multiplier))
                }
                
        # Calculate country-specific statistics
        country_stats = {}
        for country in df['country_txt'].unique():
            country_data = country_year_counts[country_year_counts['country_txt'] == country]
            
            if len(country_data) > 0:
                country_counts = country_data['attacks'].values
                country_stats[country] = {
                    "avg_attacks": float(np.mean(country_counts)),
                    "median_attacks": float(np.median(country_counts)),
                    "max_attacks": float(np.max(country_counts)),
                    "p75": float(np.percentile(country_counts, 75) if len(country_counts) > 0 else 0),
                    "p90": float(np.percentile(country_counts, 90) if len(country_counts) > 0 else 0),
                    "total_incidents": int(np.sum(country_counts)),
                    "years_with_data": len(country_data['iyear'].unique()),
                    "region": str(df[df['country_txt'] == country]['region_txt'].iloc[0])
                }
                
                # Calculate country risk level based on historical data
                region = country_stats[country]["region"]
                avg_attacks = country_stats[country]["avg_attacks"]
                
                if region in region_stats:
                    # Determine risk level based on region-specific thresholds
                    region_thresholds = region_stats[region]["thresholds"]
                    
                    if avg_attacks > region_thresholds["high"]:
                        risk_level = "High"
                    elif avg_attacks > region_thresholds["medium"]:
                        risk_level = "Medium"
                    elif avg_attacks > region_thresholds["low"]:
                        risk_level = "Medium"  # Reduce "Low" skew
                    else:
                        risk_level = "Low"
                        
                    country_stats[country]["risk_level"] = risk_level
                    
                    # Calculate risk threshold multiplier
                    country_multiplier = avg_attacks / region_stats[region]["avg_attacks"]
                    country_multiplier = min(max(country_multiplier, 0.5), 2.0)  # Bound the multiplier
                    country_stats[country]["risk_multiplier"] = float(country_multiplier)
                
        # Create output structure
        results = {
            "generated_at": datetime.now().isoformat(),
            "overall_percentiles": overall_percentiles,
            "region_stats": region_stats,
            "country_stats": country_stats,
            "recommended_thresholds": {
                "global": {
                    "low": float(overall_percentiles["p25"]),
                    "medium": float(overall_percentiles["p75"]),
                    "high": float(overall_percentiles["p90"])
                }
            }
        }
        
        # Export region risk multipliers for use in prediction code
        region_risk_multipliers = {
            region: stats["risk_multiplier"] 
            for region, stats in region_stats.items()
        }
        
        country_risk_mappings = {
            country: stats["risk_level"]
            for country, stats in country_stats.items()
            if "risk_level" in stats
        }
        
        # Save the analysis results
        output_path = project_root / "data" / "risk_analysis.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        # Save simplified risk mappings for direct use in prediction code
        mapping_path = project_root / "data" / "risk_mappings.json"
        with open(mapping_path, "w") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "region_risk_multipliers": region_risk_multipliers,
                "country_risk_mappings": country_risk_mappings,
                "global_thresholds": results["recommended_thresholds"]["global"]
            }, f, indent=2)
            
        print(f"Analysis complete. Results saved to:")
        print(f"  - {output_path}")
        print(f"  - {mapping_path}")
        
        return True
            
    except Exception as e:
        print(f"Error analyzing historical data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_historical_patterns() 