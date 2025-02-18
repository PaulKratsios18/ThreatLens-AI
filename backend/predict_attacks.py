from data_processing.data_loader import GTDDataLoader
from data_processing.temporal_feature_engineering import TemporalFeatureEngineer
from train_model import train_model

def main():
    # Train the model
    train_model()
    
    # The model will automatically generate predictions up to 2025
    # Results will be printed to console

if __name__ == "__main__":
    main() 