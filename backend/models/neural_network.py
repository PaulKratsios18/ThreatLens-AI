import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class TerrorismPredictor:
    def __init__(self):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(15,)),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
    def train(self, X_train, y_train, X_val, y_val):
        # Calculate class weights
        n_neg = len(y_train[y_train == 0])
        n_pos = len(y_train[y_train == 1])
        weight_for_0 = (1 / n_neg) * (n_neg + n_pos) / 2.0
        weight_for_1 = (1 / n_pos) * (n_neg + n_pos) / 2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            class_weight=class_weight
        )
        
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save the trained model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path) 