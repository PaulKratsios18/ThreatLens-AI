"""
Feedback Optimizer for ThreatLens-AI

This module implements a feedback mechanism that uses historical prediction accuracy
to calibrate and improve model predictions. It applies adaptive scaling, regional bias
correction, and confidence-based calibration.
"""

import json
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeedbackOptimizer:
    """
    Uses historical prediction accuracy metrics to calibrate and improve 
    model predictions through adaptive scaling and regional bias correction.
    """
    
    def __init__(self):
        self.accuracy_data = {}
        self.region_bias_factors = {}
        self.region_scaling_factors = {}
        self.confidence_calibration = {}
        self.initialized = False
    
    def load_accuracy_data(self, accuracy_file_path: Optional[Path] = None) -> bool:
        """
        Load prediction accuracy data from JSON file.
        
        Args:
            accuracy_file_path: Path to the prediction accuracy JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if accuracy_file_path is None:
                # Default path
                accuracy_file_path = Path(__file__).parent.parent.parent / 'data' / 'prediction_accuracy.json'
            
            if not accuracy_file_path.exists():
                logger.warning(f"Accuracy data file not found at {accuracy_file_path}")
                return False
                
            with open(accuracy_file_path, 'r') as f:
                self.accuracy_data = json.load(f)
                
            logger.info(f"Loaded accuracy data for {len(self.accuracy_data)} years")
            
            # Calculate bias and scaling factors for each region
            self._calculate_correction_factors()
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading accuracy data: {e}")
            return False
    
    def _calculate_correction_factors(self) -> None:
        """
        Calculate region-specific bias and scaling factors based on historical accuracy.
        """
        # Initialize dictionaries to store region data
        region_errors = {}
        region_error_ratios = {}
        region_accuracies = {}
        
        # Process all years in the accuracy data
        for year, year_data in self.accuracy_data.items():
            region_metrics = year_data.get("region_metrics", {})
            
            for region, metrics in region_metrics.items():
                actual = metrics.get("actual_attacks", 0)
                predicted = metrics.get("predicted_attacks", 0)
                
                if region not in region_errors:
                    region_errors[region] = []
                    region_error_ratios[region] = []
                    region_accuracies[region] = []
                
                # Store absolute error
                error = metrics.get("absolute_error", 0)
                region_errors[region].append(error)
                
                # Store error ratio (prediction / actual) - for scaling factor
                if actual > 0:
                    ratio = predicted / actual
                    region_error_ratios[region].append(ratio)
                
                # Store accuracy
                accuracy = metrics.get("accuracy", 0)
                region_accuracies[region].append(accuracy)
        
        # Calculate bias factors (systematic under or over prediction)
        for region, errors in region_errors.items():
            if errors:
                # Average error
                avg_error = np.mean(errors)
                # Average accuracy
                avg_accuracy = np.mean(region_accuracies[region]) if region_accuracies[region] else 0
                # Average error ratio
                avg_ratio = np.mean(region_error_ratios[region]) if region_error_ratios[region] else 1.0
                
                # Bias factor is inverse of ratio - if we consistently predict 2x the actual,
                # we should multiply predictions by 0.5
                bias_factor = 1.0 / avg_ratio if avg_ratio > 0 else 1.0
                
                # Constrain bias factor to reasonable range
                bias_factor = max(0.1, min(3.0, bias_factor))
                
                # Calculate confidence calibration based on accuracy
                confidence_factor = avg_accuracy / 100.0 if avg_accuracy > 0 else 0.5
                
                self.region_bias_factors[region] = bias_factor
                self.region_scaling_factors[region] = avg_ratio
                self.confidence_calibration[region] = confidence_factor
                
                logger.info(f"Region {region}: bias_factor={bias_factor:.2f}, scaling={avg_ratio:.2f}, confidence={confidence_factor:.2f}")
        
        # Add defaults for regions not in historical data
        for missing_region in ["North America", "South America", "Western Europe", "Eastern Europe",
                              "Middle East", "North Africa", "Sub-Saharan Africa", "Central Asia",
                              "South Asia", "East Asia", "Southeast Asia", "Oceania"]:
            if missing_region not in self.region_bias_factors:
                self.region_bias_factors[missing_region] = 1.0
                self.region_scaling_factors[missing_region] = 1.0
                self.confidence_calibration[missing_region] = 0.7
    
    def optimize_predictions(self, predictions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply correction factors to optimize predictions based on historical accuracy.
        
        Args:
            predictions: Dictionary of predictions by year
            
        Returns:
            Dictionary of optimized predictions
        """
        if not self.initialized:
            logger.warning("FeedbackOptimizer not initialized with accuracy data. Using uncorrected predictions.")
            return predictions
        
        optimized_predictions = {}
        
        for year, regions_data in predictions.items():
            optimized_regions = []
            
            for region_data in regions_data:
                region_name = region_data.get("region", "Unknown")
                
                # Get correction factors for this region
                bias_factor = self.region_bias_factors.get(region_name, 1.0)
                confidence_factor = self.confidence_calibration.get(region_name, 0.7)
                
                # Create a copy of the region data
                optimized_region = region_data.copy()
                
                # Apply bias correction to expected attacks
                original_attacks = region_data.get("expected_attacks", 0)
                
                # Apply bias correction, ensuring we never predict negative attacks
                corrected_attacks = max(1, int(original_attacks * bias_factor))
                optimized_region["expected_attacks"] = corrected_attacks
                
                # Adjust confidence score based on historical accuracy
                original_confidence = region_data.get("confidence_score", 0.7)
                calibrated_confidence = original_confidence * confidence_factor
                
                # Ensure confidence stays in reasonable range
                calibrated_confidence = max(0.1, min(0.95, calibrated_confidence))
                optimized_region["confidence_score"] = calibrated_confidence
                
                # Indicate that this prediction was optimized
                optimized_region["optimized"] = True
                optimized_region["original_expected_attacks"] = original_attacks
                optimized_region["original_confidence"] = original_confidence
                
                optimized_regions.append(optimized_region)
            
            optimized_predictions[year] = optimized_regions
        
        return optimized_predictions
    
    def optimize_country_predictions(self, country_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply correction factors to optimize country-level predictions.
        
        Args:
            country_predictions: List of country prediction dictionaries
            
        Returns:
            List of optimized country predictions
        """
        if not self.initialized:
            logger.warning("FeedbackOptimizer not initialized with accuracy data. Using uncorrected predictions.")
            return country_predictions
        
        optimized_predictions = []
        
        for country_pred in country_predictions:
            region_name = country_pred.get("region_name", "Unknown")
            
            # Get correction factors for this region
            bias_factor = self.region_bias_factors.get(region_name, 1.0)
            confidence_factor = self.confidence_calibration.get(region_name, 0.7)
            
            # Create a copy of the country prediction
            optimized_pred = country_pred.copy()
            
            # Apply bias correction to expected attacks
            original_attacks = country_pred.get("expected_attacks", 0)
            
            # Apply bias correction, ensuring we never predict negative attacks
            corrected_attacks = max(1, int(original_attacks * bias_factor))
            optimized_pred["expected_attacks"] = corrected_attacks
            
            # Adjust confidence score based on historical accuracy
            original_confidence = country_pred.get("confidence_score", 0.7)
            calibrated_confidence = original_confidence * confidence_factor
            
            # Ensure confidence stays in reasonable range
            calibrated_confidence = max(0.1, min(0.95, calibrated_confidence))
            optimized_pred["confidence_score"] = calibrated_confidence
            
            # Adjust attack types proportionally
            if "attack_types" in country_pred:
                original_attack_types = country_pred["attack_types"]
                optimized_attack_types = {}
                
                # Calculate the scaling factor to maintain the proper distribution
                total_original_attacks = sum(original_attack_types.values())
                attack_scaling = corrected_attacks / total_original_attacks if total_original_attacks > 0 else 1.0
                
                # Scale each attack type
                for attack_type, count in original_attack_types.items():
                    optimized_attack_types[attack_type] = max(0, int(count * attack_scaling))
                
                # Ensure we have the correct total
                total_optimized = sum(optimized_attack_types.values())
                if total_optimized != corrected_attacks:
                    # Distribute the difference
                    diff = corrected_attacks - total_optimized
                    # Find the attack type with the highest count
                    max_attack = max(optimized_attack_types.items(), key=lambda x: x[1])
                    optimized_attack_types[max_attack[0]] += diff
                
                optimized_pred["attack_types"] = optimized_attack_types
            
            # Add metadata about the optimization
            optimized_pred["optimized"] = True
            optimized_pred["optimization_factor"] = bias_factor
            
            optimized_predictions.append(optimized_pred)
        
        # Re-sort by expected attacks
        optimized_predictions.sort(key=lambda x: x["expected_attacks"], reverse=True)
        
        # Update ranks
        for i, pred in enumerate(optimized_predictions):
            pred["rank"] = i + 1
        
        return optimized_predictions
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the optimization factors.
        
        Returns:
            Dictionary with optimization statistics
        """
        if not self.initialized:
            return {"initialized": False, "message": "Optimizer not initialized"}
        
        stats = {
            "initialized": True,
            "num_years_analyzed": len(self.accuracy_data),
            "regions_analyzed": len(self.region_bias_factors),
            "region_factors": {}
        }
        
        for region, bias in self.region_bias_factors.items():
            stats["region_factors"][region] = {
                "bias_factor": bias,
                "scaling_factor": self.region_scaling_factors.get(region, 1.0),
                "confidence_calibration": self.confidence_calibration.get(region, 0.7)
            }
        
        return stats 