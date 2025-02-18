import pandas as pd
import numpy as np
from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import os

def train_model():
    print("Initializing components...")
    data_loader = GTDDataLoader()
    
    print("Loading and preprocessing data...")
    data = data_loader.load_data("../data/globalterrorismdb_0522dist.csv")
    
    # Features that might indicate future attack patterns
    pattern_features = [
        'attacktype1', 'weaptype1', 'targtype1',
        'nperps', 'nkill', 'nwound',
        'suicide', 'success', 'multiple',
        'INT_LOG', 'INT_IDEO', 'INT_ANY'
    ]
    
    # Group data by year and region to get attack frequencies
    yearly_patterns = data.groupby(['iyear', 'region', 'attacktype1']).size().reset_index(name='frequency')
    
    # Create sequences for time series prediction
    sequence_length = 5  # Use 5 years of data to predict next year
    X, y = create_sequences(yearly_patterns, sequence_length)
    
    # Split data chronologically
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model for time series prediction
    model = Sequential([
        LSTM(256, input_shape=(sequence_length, 4), return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(13, activation='softplus')  # Changed to softplus for non-negative predictions
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/attack_prediction_model.h5')
    
    # Generate future predictions
    predict_future_attacks(model, X_test[-1:], 2025)

def create_sequences(data, sequence_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    
    # First convert data to proper numeric format
    numeric_columns = ['iyear', 'region', 'attacktype1', 'frequency']
    for col in numeric_columns:
        data[col] = data[col].astype(float)
    
    # Create sequences for each region
    for region in data['region'].unique():
        region_data = data[data['region'] == region].sort_values('iyear')
        features = region_data[numeric_columns].values
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:(i + sequence_length)])
            # Create one-hot encoded target for region (13 classes) and add frequency
            target = np.zeros(13)
            target[int(features[i + sequence_length, 1])] = features[i + sequence_length, 3]
            y.append(target)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def predict_future_attacks(model, last_sequence, target_year):
    """Predict attack patterns for future years"""
    predictions = []
    current_sequence = last_sequence.copy()
    preprocessor = GTDPreprocessor()
    
    print("\nPredicted Attack Patterns:")
    for year in range(2023, target_year + 1):
        # Get predictions for all regions
        pred = model.predict(current_sequence, verbose=0)[0]
        
        # Normalize predictions to be between 0 and 1000
        normalized_pred = (pred - pred.min()) / (pred.max() - pred.min()) * 1000
        
        # Calculate confidence scores
        confidence_scores = np.exp(pred) / np.sum(np.exp(pred))  # Softmax for probabilities
        
        # Process predictions for each region
        for region in range(13):
            predicted_attacks = int(normalized_pred[region])
            confidence = float(confidence_scores[region])
            
            if predicted_attacks > 50:  # Only show significant predictions
                # Get attack type distribution from historical data
                attack_types = get_attack_type_distribution(region, year-1)
                
                prediction_data = {
                    'year': year,
                    'region': region,
                    'region_name': preprocessor.get_region_name(region),
                    'expected_attacks': predicted_attacks,
                    'confidence_score': confidence,
                    'risk_level': "High" if confidence > 0.6 else "Medium" if confidence > 0.3 else "Low",
                    'attack_types': attack_types
                }
                
                predictions.append(prediction_data)
                print(f"\nYear {year}:")
                print(f"Region: {prediction_data['region_name']}")
                print(f"Expected Attacks: {predicted_attacks}")
                print(f"Confidence Score: {confidence:.3f} ({prediction_data['risk_level']} risk)")
                print("Likely Attack Types:")
                for attack_type, probability in attack_types.items():
                    print(f"- {attack_type}: {probability:.1%}")
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        new_row = current_sequence[0, -2].copy()
        new_row[3] = np.mean(normalized_pred)
        current_sequence[0, -1] = new_row
    
    if not predictions:
        print("\nNo significant attack patterns predicted for the given time period.")
    
    return predictions

def get_attack_type_distribution(region, year):
    """Get historical attack type distribution for a region"""
    data_loader = GTDDataLoader()
    data = data_loader.load_data("../data/globalterrorismdb_0522dist.csv")
    
    # Filter data for the specific region and recent years
    recent_data = data[
        (data['region'] == region) & 
        (data['iyear'] >= year-5) & 
        (data['iyear'] <= year)
    ]
    
    # Get attack type distribution
    attack_counts = recent_data['attacktype1'].value_counts()
    total_attacks = attack_counts.sum()
    
    # Map attack type codes to names
    attack_type_names = {
        1: "Bombing/Explosion",
        2: "Armed Assault",
        3: "Assassination",
        4: "Hostage Taking",
        5: "Infrastructure Attack",
        6: "Unarmed Assault",
        7: "Other"
    }
    
    # Calculate probabilities
    distribution = {
        attack_type_names.get(code, "Other"): count/total_attacks 
        for code, count in attack_counts.items()
    }
    
    return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:3])

if __name__ == "__main__":
    train_model() 