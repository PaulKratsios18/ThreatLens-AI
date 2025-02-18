from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
from data_processing.feature_engineering import GTDFeatureEngineer
import pandas as pd

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

if __name__ == "__main__":
    main() 