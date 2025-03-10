from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import random

from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
from data_processing.feature_engineering import GTDFeatureEngineer
from models.neural_network import TerrorismPredictor
from models.xai_explainer import AttackExplainer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_loader = GTDDataLoader()
preprocessor = GTDPreprocessor()
feature_engineer = GTDFeatureEngineer()

# Load and process a small sample of data for the explainer
print("Loading sample data for explainer...")
data = data_loader.load_data("../data/globalterrorismdb_0522dist.csv")
data_with_features = feature_engineer.engineer_features(data)
X_train, _, y_train, _ = preprocessor.preprocess(data_with_features)

# Initialize model and explainer
model = TerrorismPredictor()
model.load("models/trained_model.keras")
explainer = AttackExplainer(model, preprocessor.feature_names, X_train[:1000])  # Use first 1000 samples

class PredictionRequest(BaseModel):
    features: Dict[str, float]

@app.post("/predict")
async def predict_attack(request: PredictionRequest):
    try:
        features = np.array(list(request.features.values())).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        return {
            "success_probability": float(prediction),
            "risk_level": "High" if prediction > 0.8 else "Medium" if prediction > 0.4 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
async def get_feature_importance():
    return {
        "feature_importance": model.get_feature_importance(),
        "feature_names": preprocessor.feature_names
    }

@app.post("/explain")
async def explain_prediction(request: PredictionRequest):
    try:
        features = np.array(list(request.features.values())).reshape(1, -1)
        explanation = explainer.explain_prediction(features[0])
        prediction = float(model.predict(features)[0])
        
        # Get key factors analysis
        key_factors = explainer.analyze_key_factors(explanation)
        
        return {
            "success_probability": prediction,
            "risk_level": "High" if prediction > 0.8 else "Medium" if prediction > 0.4 else "Low",
            "explanations": explanation,
            "key_factors": key_factors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-attacks")
async def predict_future_attacks(request: PredictionRequest):
    try:
        features = np.array(list(request.features.values())).reshape(1, -1)
        
        predictions = {
            "location_probabilities": location_predictions,
            "time_frame": time_predictions,
            "attack_type_probabilities": attack_type_predictions,
            "risk_zones": risk_zones,
            "confidence_score": confidence
        }
        
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-location")
async def predict_attack_location(request: PredictionRequest):
    try:
        features = np.array(list(request.features.values())).reshape(1, -1)
        
        # Get location cluster predictions
        location_probs = model.predict(features)[0]
        
        # Get top 3 most likely regions
        top_regions = []
        for cluster_id, prob in enumerate(location_probs):
            region_name = preprocessor.get_region_name(cluster_id)
            if prob > 0.1:  # Only include significant probabilities
                top_regions.append({
                    "region": region_name,
                    "probability": float(prob),
                    "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
                })
        
        return {
            "predictions": sorted(top_regions, key=lambda x: x['probability'], reverse=True),
            "timestamp": datetime.now().isoformat(),
            "confidence_score": float(max(location_probs))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/static-predictions")
async def get_static_predictions():
    # Return comprehensive mock data for countries based on GTI
    return {
        "predictions": {
            "2023": [
                # Middle East & North Africa countries
                {
                    "country": "Iraq",
                    "region": "Middle East & North Africa",
                    "gti_score": 7.6,
                    "expected_attacks": 69,
                    "confidence_score": 0.85,
                    "risk_level": "High",
                    "rank": 7,
                    "change_from_previous": -0.5,
                    "attack_types": {
                        "Bombing/Explosion": 30,
                        "Armed Assault": 20,
                        "Assassination": 10,
                        "Hostage Taking": 5,
                        "Other": 4
                    },
                    "primary_groups": ["Islamic State", "Unknown"]
                },
                {
                    "country": "Syria",
                    "region": "Middle East & North Africa",
                    "gti_score": 8.1,
                    "expected_attacks": 105,
                    "confidence_score": 0.88,
                    "risk_level": "High",
                    "rank": 3,
                    "change_from_previous": 0.2,
                    "attack_types": {
                        "Bombing/Explosion": 45,
                        "Armed Assault": 35,
                        "Hostage Taking": 15,
                        "Assassination": 10,
                        "Other": 0
                    },
                    "primary_groups": ["Islamic State", "Al-Qaeda", "Unknown"]
                },
                # Sahel region countries (highest impact per GTI)
                {
                    "country": "Burkina Faso",
                    "region": "Sub-Saharan Africa",
                    "gti_score": 8.6,
                    "expected_attacks": 120,
                    "confidence_score": 0.9,
                    "risk_level": "High",
                    "rank": 1,
                    "change_from_previous": 0.8,
                    "attack_types": {
                        "Armed Assault": 65,
                        "Bombing/Explosion": 30,
                        "Hostage Taking": 15,
                        "Assassination": 5,
                        "Other": 5
                    },
                    "primary_groups": ["JNIM", "Islamic State Sahel Province", "Unknown"]
                },
                {
                    "country": "Mali",
                    "region": "Sub-Saharan Africa",
                    "gti_score": 8.5,
                    "expected_attacks": 115,
                    "confidence_score": 0.87,
                    "risk_level": "High",
                    "rank": 2,
                    "change_from_previous": 0.6,
                    "attack_types": {
                        "Armed Assault": 60,
                        "Bombing/Explosion": 35,
                        "Hostage Taking": 10,
                        "Assassination": 5,
                        "Other": 5
                    },
                    "primary_groups": ["JNIM", "Islamic State Sahel Province"]
                },
                {
                    "country": "Niger",
                    "region": "Sub-Saharan Africa",
                    "gti_score": 8.0,
                    "expected_attacks": 90,
                    "confidence_score": 0.85,
                    "risk_level": "High",
                    "rank": 4,
                    "change_from_previous": 0.9,
                    "attack_types": {
                        "Armed Assault": 50,
                        "Bombing/Explosion": 25,
                        "Hostage Taking": 10,
                        "Assassination": 5,
                        "Other": 0
                    },
                    "primary_groups": ["Islamic State Sahel Province", "JNIM", "Boko Haram"]
                },
                # South Asia
                {
                    "country": "Afghanistan",
                    "region": "South Asia",
                    "gti_score": 7.8,
                    "expected_attacks": 85,
                    "confidence_score": 0.83,
                    "risk_level": "High",
                    "rank": 5,
                    "change_from_previous": -0.7,
                    "attack_types": {
                        "Bombing/Explosion": 45,
                        "Armed Assault": 25,
                        "Assassination": 10,
                        "Hostage Taking": 5,
                        "Other": 0
                    },
                    "primary_groups": ["Islamic State Khorasan", "Tehrik-e-Taliban Pakistan (TTP)"]
                },
                {
                    "country": "Pakistan",
                    "region": "South Asia",
                    "gti_score": 7.7,
                    "expected_attacks": 75,
                    "confidence_score": 0.8,
                    "risk_level": "High",
                    "rank": 6,
                    "change_from_previous": 0.5,
                    "attack_types": {
                        "Bombing/Explosion": 35,
                        "Armed Assault": 20,
                        "Assassination": 10,
                        "Hostage Taking": 5,
                        "Other": 5
                    },
                    "primary_groups": ["Tehrik-e-Taliban Pakistan (TTP)", "Balochistan Liberation Army"]
                },
                # Other notable countries
                {
                    "country": "Nigeria",
                    "region": "Sub-Saharan Africa",
                    "gti_score": 7.5,
                    "expected_attacks": 65,
                    "confidence_score": 0.78,
                    "risk_level": "High",
                    "rank": 8,
                    "change_from_previous": -0.3,
                    "attack_types": {
                        "Armed Assault": 30,
                        "Bombing/Explosion": 20,
                        "Hostage Taking": 10,
                        "Assassination": 5,
                        "Other": 0
                    },
                    "primary_groups": ["Boko Haram", "Islamic State West Africa Province (ISWAP)"]
                },
                {
                    "country": "Somalia",
                    "region": "Sub-Saharan Africa",
                    "gti_score": 7.4,
                    "expected_attacks": 60,
                    "confidence_score": 0.76,
                    "risk_level": "High",
                    "rank": 9,
                    "change_from_previous": -0.1,
                    "attack_types": {
                        "Bombing/Explosion": 25,
                        "Armed Assault": 20,
                        "Assassination": 10,
                        "Hostage Taking": 5,
                        "Other": 0
                    },
                    "primary_groups": ["Al-Shabaab"]
                },
                {
                    "country": "Democratic Republic of the Congo",
                    "region": "Sub-Saharan Africa",
                    "gti_score": 7.3,
                    "expected_attacks": 55,
                    "confidence_score": 0.75,
                    "risk_level": "High",
                    "rank": 10,
                    "change_from_previous": 0.4,
                    "attack_types": {
                        "Armed Assault": 30,
                        "Bombing/Explosion": 10,
                        "Hostage Taking": 10,
                        "Assassination": 5,
                        "Other": 0
                    },
                    "primary_groups": ["Islamic State Central Africa Province", "Allied Democratic Forces"]
                },
                # Western countries
                {
                    "country": "France",
                    "region": "Western Europe",
                    "gti_score": 4.2,
                    "expected_attacks": 15,
                    "confidence_score": 0.65,
                    "risk_level": "Medium",
                    "rank": 32,
                    "change_from_previous": 0.2,
                    "attack_types": {
                        "Armed Assault": 7,
                        "Bombing/Explosion": 5,
                        "Vehicular Attack": 3,
                        "Other": 0
                    },
                    "primary_groups": ["Islamic State-inspired", "Political Extremism"]
                },
                {
                    "country": "United States",
                    "region": "North America",
                    "gti_score": 4.0,
                    "expected_attacks": 12,
                    "confidence_score": 0.6,
                    "risk_level": "Medium",
                    "rank": 36,
                    "change_from_previous": -0.3,
                    "attack_types": {
                        "Armed Assault": 6,
                        "Bombing/Explosion": 3,
                        "Vehicular Attack": 1,
                        "Other": 2
                    },
                    "primary_groups": ["Political Extremism", "Lone Wolf"]
                },
                {
                    "country": "United Kingdom",
                    "region": "Western Europe",
                    "gti_score": 3.5,
                    "expected_attacks": 8,
                    "confidence_score": 0.55,
                    "risk_level": "Medium",
                    "rank": 42,
                    "change_from_previous": 0.1,
                    "attack_types": {
                        "Armed Assault": 4,
                        "Bombing/Explosion": 2,
                        "Vehicular Attack": 1,
                        "Other": 1
                    },
                    "primary_groups": ["Islamic State-inspired", "Political Extremism"]
                },
                # Low impact countries
                {
                    "country": "Canada",
                    "region": "North America",
                    "gti_score": 2.1,
                    "expected_attacks": 3,
                    "confidence_score": 0.4,
                    "risk_level": "Low",
                    "rank": 75,
                    "change_from_previous": -0.2,
                    "attack_types": {
                        "Armed Assault": 2,
                        "Vehicular Attack": 1,
                        "Other": 0
                    },
                    "primary_groups": ["Political Extremism", "Lone Wolf"]
                },
                {
                    "country": "Japan",
                    "region": "East Asia",
                    "gti_score": 0.8,
                    "expected_attacks": 1,
                    "confidence_score": 0.3,
                    "risk_level": "Low",
                    "rank": 112,
                    "change_from_previous": 0.0,
                    "attack_types": {
                        "Armed Assault": 1,
                        "Other": 0
                    },
                    "primary_groups": ["Lone Wolf"]
                },
                {
                    "country": "Iceland",
                    "region": "Western Europe",
                    "gti_score": 0.0,
                    "expected_attacks": 0,
                    "confidence_score": 0.1,
                    "risk_level": "Low",
                    "rank": 163,
                    "change_from_previous": 0.0,
                    "attack_types": {
                        "Other": 0
                    },
                    "primary_groups": []
                }
            ]
        }
    }

@app.get("/historical-data")
async def get_historical_data(year: int = 2020):
    """
    Get historical terrorism data for a specific year from the Global Terrorism Database.
    """
    try:
        # Read from the actual GTD dataset
        gtd_file_path = Path(__file__).parent.parent.parent / 'data' / 'globalterrorismdb_0522dist.csv'
        
        # Check if file exists
        if not gtd_file_path.exists():
            raise FileNotFoundError(f"GTD dataset file not found at {gtd_file_path}")
        
        # Read only essential columns to reduce memory usage
        essential_columns = [
            'eventid', 'iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 
            'city', 'latitude', 'longitude', 'attacktype1_txt',
            'targtype1_txt', 'weaptype1_txt', 'nkill', 'nwound', 'gname'
        ]
        
        # Read the data for the specified year only
        df = pd.read_csv(gtd_file_path, usecols=essential_columns, low_memory=False)
        df = df[df['iyear'] == year].copy()
        
        # Create standardized response
        incidents = []
        
        for idx, row in df.iterrows():
            # Handle missing values
            latitude = row['latitude'] if not pd.isna(row['latitude']) else 0
            longitude = row['longitude'] if not pd.isna(row['longitude']) else 0
            num_killed = row['nkill'] if not pd.isna(row['nkill']) else 0
            num_wounded = row['nwound'] if not pd.isna(row['nwound']) else 0
            
            incidents.append({
                "id": int(row['eventid']),
                "year": int(row['iyear']),
                "month": int(row['imonth']) if not pd.isna(row['imonth']) else 0,
                "day": int(row['iday']) if not pd.isna(row['iday']) else 0,
                "region": row['region_txt'],
                "country": row['country_txt'],
                "city": row['city'] if not pd.isna(row['city']) else "",
                "latitude": float(latitude),
                "longitude": float(longitude),
                "attack_type": row['attacktype1_txt'] if not pd.isna(row['attacktype1_txt']) else "Unknown",
                "weapon_type": row['weaptype1_txt'] if not pd.isna(row['weaptype1_txt']) else "Unknown",
                "target_type": row['targtype1_txt'] if not pd.isna(row['targtype1_txt']) else "Unknown",
                "num_killed": int(num_killed),
                "num_wounded": int(num_wounded),
                "group_name": row['gname'] if not pd.isna(row['gname']) else "Unknown"
            })
        
        return {"incidents": incidents}
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error getting historical data: {str(e)}")
        # Return an empty response with error details
        return {
            "error": str(e),
            "incidents": []
        }
