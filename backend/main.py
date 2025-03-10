from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
from data_processing.feature_engineering import GTDFeatureEngineer
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def main():
    # Initialize components
    data_loader = GTDDataLoader()
    preprocessor = GTDPreprocessor()
    feature_engineer = GTDFeatureEngineer()
    
    # 1. Load and clean data
    print("Loading and cleaning data...")
    data = data_loader.load_data("../data/globalterrorismdb_0522dist.csv")
    print(f"Loaded {len(data)} records")
    
    # 2. Engineer features
    print("\nEngineering features...")
    data_with_features = feature_engineer.engineer_features(data)
    print(f"Created features. Shape: {data_with_features.shape}")
    
    # 3. Preprocess for ML
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess(data_with_features)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 4. Display sample statistics
    print("\nSample statistics:")
    print(f"Number of attacks by region:")
    print(data_loader.get_attacks_by_region())
    
    # Save processed data (optional)
    pd.DataFrame(X_train).to_csv('processed_data/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('processed_data/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv('processed_data/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('processed_data/y_test.csv', index=False)

# Add your route handlers here
@app.get("/static-predictions")
async def get_static_predictions():
    # Return mock data for now
    return {
        "predictions": {
            "2023": [
                {
                    "region": "Middle East & North Africa",
                    "expected_attacks": 180,
                    "confidence_score": 0.85,
                    "attack_types": {
                        "Bombing/Explosion": 60,
                        "Armed Assault": 40,
                        "Hostage Taking": 10,
                        "Other": 15
                    }
                },
                {
                    "region": "South Asia",
                    "expected_attacks": 150,
                    "confidence_score": 0.78,
                    "attack_types": {
                        "Bombing/Explosion": 55,
                        "Armed Assault": 35,
                        "Assassination": 15,
                        "Other": 10
                    }
                }
            ]
        }
    }

@app.get("/historical-data")
async def get_historical_data(year: int = 2020):
    # Return mock data for now
    return {
        "incidents": [
            {
                "id": 1,
                "year": year,
                "month": 1,
                "day": 15,
                "region": "Middle East & North Africa",
                "country": "Iraq",
                "city": "Baghdad",
                "latitude": 33.3152,
                "longitude": 44.3661,
                "attack_type": "Bombing/Explosion",
                "weapon_type": "Explosives",
                "target_type": "Government",
                "num_killed": 5,
                "num_wounded": 12
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 