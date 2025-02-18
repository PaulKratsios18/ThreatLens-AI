import pandas as pd
import numpy as np
from data_processing.data_loader import GTDDataLoader
from data_processing.data_preprocessor import GTDPreprocessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def validate_model_performance():
    # Load components
    data_loader = GTDDataLoader()
    preprocessor = GTDPreprocessor()
    
    try:
        # Load the trained model
        print("Loading model...")
        model = joblib.load('models/location_prediction_model.joblib')
        
        # Load and preprocess test data
        print("Loading test data...")
        test_data = pd.read_csv('processed_data/X_test.csv')
        test_labels = pd.read_csv('processed_data/y_test.csv')
        
        # Make predictions
        predictions = model.predict(test_data)
        
        # Print classification report
        print("\nModel Performance Report:")
        print("-" * 50)
        print(classification_report(test_labels, predictions))
        
        # Plot confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=preprocessor.region_names.values(),
                    yticklabels=preprocessor.region_names.values())
        plt.title('Confusion Matrix')
        plt.ylabel('True Region')
        plt.xlabel('Predicted Region')
        plt.tight_layout()
        plt.savefig('model_validation/confusion_matrix.png')
        
        # Print feature importance if model supports it
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importance:")
            print("-" * 50)
            importance = pd.DataFrame({
                'feature': preprocessor.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(importance)
            
    except FileNotFoundError as e:
        print(f"Error: Could not find model or test data files: {str(e)}")
    except Exception as e:
        print(f"Error during model validation: {str(e)}")

if __name__ == "__main__":
    validate_model_performance() 