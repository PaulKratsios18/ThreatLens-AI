import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.models import model
import json
from datetime import datetime

def generate_predictions():
    years = [2023, 2024, 2025]
    predictions_by_year = {}
    
    for year in years:
        predictions = model.predict({'year': year})
        predictions_by_year[str(year)] = predictions
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'predictions': predictions_by_year
    }
    
    output_path = Path(__file__).parent.parent / 'data' / 'predictions.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
        print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    generate_predictions() 