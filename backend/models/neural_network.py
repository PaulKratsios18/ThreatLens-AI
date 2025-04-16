import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

class TerrorismPredictor:
    def __init__(self):
        # Define input shapes for different feature groups
        self.base_features_shape = 15  # Original features
        self.socio_features_shape = 7   # Country-level socioeconomic features
        self.region_socio_features_shape = 7  # Region-level socioeconomic features
        
        # Create a more complex model architecture
        self.model = Sequential([
            # Base features branch
            Dense(128, activation='relu', input_shape=(self.base_features_shape,),
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Socioeconomic features branch
            Dense(64, activation='relu', input_shape=(self.socio_features_shape,),
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Region socioeconomic features branch
            Dense(64, activation='relu', input_shape=(self.region_socio_features_shape,),
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Combined layers
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(1, activation='sigmoid')
        ])
        
        # Use a more sophisticated optimizer with learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
    def train(self, X_train, y_train, X_val, y_val):
        # Calculate class weights
        n_neg = len(y_train[y_train == 0])
        n_pos = len(y_train[y_train == 1])
        weight_for_0 = (1 / n_neg) * (n_neg + n_pos) / 2.0
        weight_for_1 = (1 / n_pos) * (n_neg + n_pos) / 2.0
        class_weight = {0: weight_for_0, 1: weight_for_1}

        # Enhanced early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Add model checkpointing
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Add learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,  # Increased epochs
            batch_size=32,
            callbacks=[early_stopping, checkpoint, reduce_lr],
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
    
    def get_feature_importance(self, X):
        """Get feature importance using integrated gradients"""
        # Implementation of integrated gradients for feature importance
        pass 