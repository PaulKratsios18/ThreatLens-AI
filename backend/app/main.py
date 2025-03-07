from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Dict
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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Be more specific than "*"
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
    try:
        predictions_path = Path(__file__).parent.parent / 'data' / 'predictions.json'
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical-data")
async def get_historical_data(year: int = 2020):
    try:
        # For this example, we'll generate mock data
        # In a real application, you would load this from your database or files
        
        # Regions match those in your frontend
        regions = [
            "North America", "South America", "Western Europe", 
            "Eastern Europe", "Middle East", "North Africa",
            "Sub-Saharan Africa", "Central Asia", "South Asia", 
            "East Asia", "Southeast Asia", "Oceania"
        ]
        
        attack_types = [
            "Bombing/Explosion", "Armed Assault", "Assassination",
            "Hostage Taking", "Infrastructure Attack", "Unarmed Assault"
        ]
        
        weapon_types = [
            "Explosives", "Firearms", "Melee", "Vehicle", "Chemical", "Other"
        ]
        
        target_types = [
            "Civilian", "Government", "Military", "Infrastructure", "Religious", "Educational"
        ]
        
        # Generate some sample data
        historical_data = []
        
        # Different number of incidents based on year to show trends
        num_incidents = {
            2018: 250,
            2019: 280,
            2020: 300,
            2021: 270,
            2022: 290
        }.get(year, 300)
        
        for i in range(num_incidents):
            region = random.choice(regions)
            
            # Set coordinates based on region
            if region == "North America":
                lat, lng = 40 + random.uniform(-15, 15), -100 + random.uniform(-25, 25)
                country = random.choice(["United States", "Canada", "Mexico"])
            elif region == "South America":
                lat, lng = -15 + random.uniform(-15, 15), -60 + random.uniform(-15, 15)
                country = random.choice(["Brazil", "Argentina", "Colombia", "Peru"])
            # ... add coordinates for other regions similarly
            
            # Default if region not matched
            else:
                lat, lng = random.uniform(-50, 50), random.uniform(-180, 180)
                country = "Unknown"
                
            historical_data.append({
                "id": i,
                "year": year,
                "month": random.randint(1, 12),
                "day": random.randint(1, 28),
                "region": region,
                "country": country,
                "city": f"City {i}",
                "latitude": lat,
                "longitude": lng,
                "attack_type": random.choice(attack_types),
                "weapon_type": random.choice(weapon_types),
                "target_type": random.choice(target_types),
                "num_killed": max(0, int(random.expovariate(0.5))),
                "num_wounded": max(0, int(random.expovariate(0.3)))
            })
            
        return historical_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
