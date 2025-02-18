from keras.models import Sequential, Dense, Dropout

class AttackPredictor:
    def __init__(self):
        self.location_model = Sequential([
            Dense(256, activation='relu', input_shape=(20,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_location_clusters, activation='softmax')  # Predict location clusters
        ])
        
        self.time_model = Sequential([
            Dense(128, activation='relu', input_shape=(20,)),
            Dense(64, activation='relu'),
            Dense(12, activation='softmax')  # Predict month probabilities
        ]) 