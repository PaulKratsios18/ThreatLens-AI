import lime
import lime.lime_tabular
import numpy as np

class AttackExplainer:
    def __init__(self, model, feature_names, training_data):
        self.model = model
        self.feature_names = feature_names
        self.feature_descriptions = {
            'attacktype1': 'Attack Type',
            'weaptype1': 'Weapon Type',
            'targtype1': 'Target Type',
            'region': 'Region',
            'vicinity': 'In Vicinity',
            'suicide': 'Suicide Attack',
            'multiple': 'Multiple Attacks',
            'individual': 'Individual Attack',
            'nperps': 'Number of Perpetrators',
            'nperpcap': 'Number Captured',
            'property': 'Property Damage',
            'ishostkid': 'Hostage/Kidnapping',
            'INT_LOG': 'Logistical Success',
            'INT_IDEO': 'Ideological Success',
            'INT_ANY': 'Any Success'
        }
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=['Failure', 'Success'],
            mode='classification'
        )
    
    def explain_prediction(self, instance, num_features=10):
        """Generate LIME explanation for a prediction"""
        # Ensure instance is 2D array
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
            
        # Create prediction function that returns probabilities for both classes
        def predict_fn(x):
            probs = self.model.predict(x.reshape(-1, 15))
            # Convert single probability to two-class probability array
            return np.column_stack([1 - probs, probs])
            
        explanation = self.explainer.explain_instance(
            instance[0], 
            predict_fn,
            num_features=num_features
        )
        # Convert feature names to descriptions
        exp_list = explanation.as_list()
        return [(self.feature_descriptions.get(feat.split(' ')[0], feat), imp) 
                for feat, imp in exp_list]
    
    def analyze_key_factors(self, explanation):
        """Analyze key factors in the prediction"""
        positive_factors = []
        negative_factors = []
        
        for feature, importance in explanation:
            if importance > 0.1:  # Significant positive impact
                positive_factors.append(feature)
            elif importance < -0.1:  # Significant negative impact
                negative_factors.append(feature)
                
        return {
            "increasing_risk": positive_factors,
            "decreasing_risk": negative_factors
        }