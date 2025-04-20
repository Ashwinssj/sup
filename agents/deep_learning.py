import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class DeepLearningForecaster:
    """
    Deep Learning based forecasting for supply chain management.
    This class provides neural network models for more accurate demand forecasting
    and anomaly detection compared to traditional statistical methods.
    """
    
    def __init__(self, lookback=30, horizon=14, hidden_units=64, dropout_rate=0.2):
        """
        Initialize the forecaster with model parameters.
        
        Args:
            lookback (int): Number of past time steps to use as input features
            horizon (int): Number of future time steps to predict
            hidden_units (int): Number of hidden units in LSTM layers
            dropout_rate (float): Dropout rate for regularization
        """
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.history = None
    
    def _create_sequences(self, data, target_col):
        """
        Create input sequences for the LSTM model.
        
        Args:
            data (pd.DataFrame): Input time series data
            target_col (str): Target column to predict
            
        Returns:
            tuple: (X, y) where X is input sequences and y is target values
        """
        X, y = [], []
        data_array = data[target_col].values
        
        for i in range(len(data_array) - self.lookback - self.horizon + 1):
            X.append(data_array[i:(i + self.lookback)])
            y.append(data_array[(i + self.lookback):(i + self.lookback + self.horizon)])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape):
        """
        Build and compile an LSTM model for time series forecasting.
        
        Args:
            input_shape (tuple): Shape of input data (lookback, features)
            
        Returns:
            keras.Model: Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        x = layers.LSTM(self.hidden_units, return_sequences=True)(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.LSTM(self.hidden_units)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.horizon)(x)
        
        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse"
        )
        return model
    
    def train(self, historical_data, target_col="orders", epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the forecasting model on historical data.
        
        Args:
            historical_data (list): List of dictionaries with historical data
            target_col (str): Column to predict (e.g., "orders", "inventory")
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            dict: Training history
        """
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        else:
            df = historical_data.copy()
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        df[target_col] = self.scaler.fit_transform(df[[target_col]])
        
        # Create sequences
        X, y = self._create_sequences(df, target_col)
        X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input
        
        # Build and train model
        self.model = self._build_model((self.lookback, 1))
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=1
        )
        
        return self.history.history
    
    def forecast(self, historical_data, target_col="orders"):
        """
        Generate forecasts using the trained model.
        
        Args:
            historical_data (list): Recent historical data for forecasting
            target_col (str): Column to predict
            
        Returns:
            list: Forecasted values for the next 'horizon' time steps
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        else:
            df = historical_data.copy()
        
        # Normalize data using the same scaler
        df[target_col] = self.scaler.transform(df[[target_col]])
        
        # Use the most recent data points for prediction
        recent_data = df[target_col].values[-self.lookback:]
        recent_data = recent_data.reshape(1, self.lookback, 1)  # Reshape for LSTM input
        
        # Generate prediction
        forecast = self.model.predict(recent_data)
        
        # Inverse transform to get original scale
        forecast_reshaped = forecast.reshape(forecast.shape[1], 1)
        forecast_original = self.scaler.inverse_transform(forecast_reshaped).flatten()
        
        # Generate forecast dates
        last_date = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(self.horizon)]
        
        # Create forecast result
        result = []
        for i, date in enumerate(forecast_dates):
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "forecasted_demand": max(int(forecast_original[i]), 0),
                "lower_bound": max(int(forecast_original[i] * 0.8), 0),  # Simple confidence interval
                "upper_bound": int(forecast_original[i] * 1.2)  # Simple confidence interval
            })
        
        return result

class DeepAnomalyDetector:
    """
    Deep Learning based anomaly detection for supply chain data.
    Uses autoencoders to identify unusual patterns in multivariate time series data.
    """
    
    def __init__(self, lookback=14, latent_dim=8, threshold_percentile=95):
        """
        Initialize the anomaly detector.
        
        Args:
            lookback (int): Number of past time steps to use as context
            latent_dim (int): Dimension of the latent space in autoencoder
            threshold_percentile (int): Percentile to use for anomaly threshold
        """
        self.lookback = lookback
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.scaler = None
        self.reconstruction_errors = None
        self.threshold = None
    
    def _create_sequences(self, data):
        """
        Create input sequences for the autoencoder model.
        
        Args:
            data (np.array): Input multivariate time series data
            
        Returns:
            np.array: Sequence data
        """
        X = []
        for i in range(len(data) - self.lookback + 1):
            X.append(data[i:(i + self.lookback)])
        return np.array(X)
    
    def _build_model(self, input_shape):
        """
        Build an autoencoder model for anomaly detection.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            keras.Model: Compiled autoencoder model
        """
        # Encoder
        inputs = keras.Input(shape=input_shape)
        encoded = layers.LSTM(64, return_sequences=True)(inputs)
        encoded = layers.LSTM(32)(encoded)
        encoded = layers.Dense(self.latent_dim, activation="relu")(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(self.lookback)(encoded)
        decoded = layers.LSTM(32, return_sequences=True)(decoded)
        decoded = layers.LSTM(64, return_sequences=True)(decoded)
        outputs = layers.TimeDistributed(layers.Dense(input_shape[1]))(decoded)
        
        # Autoencoder model
        autoencoder = keras.Model(inputs, outputs)
        autoencoder.compile(optimizer="adam", loss="mse")
        
        return autoencoder
    
    def train(self, historical_data, numerical_cols=None, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the anomaly detection model on historical data.
        
        Args:
            historical_data (list): List of dictionaries with historical data
            numerical_cols (list): List of numerical columns to use for anomaly detection
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            dict: Training history
        """
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        else:
            df = historical_data.copy()
        
        # Select numerical columns if not specified
        if numerical_cols is None:
            numerical_cols = ['orders', 'deliveries', 'inventory', 'costs']
        
        # Ensure all required columns exist
        for col in numerical_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df[numerical_cols]),
            columns=numerical_cols,
            index=df.index
        )
        
        # Create sequences
        sequences = self._create_sequences(df_scaled.values)
        
        # Build and train model
        self.model = self._build_model((self.lookback, len(numerical_cols)))
        history = self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # Calculate reconstruction errors on training data
        reconstructions = self.model.predict(sequences)
        reconstruction_errors = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        self.reconstruction_errors = reconstruction_errors
        
        # Set threshold for anomaly detection
        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        return history.history
    
    def detect_anomalies(self, historical_data, numerical_cols=None, sensitivity=1.0):
        """
        Detect anomalies in supply chain data.
        
        Args:
            historical_data (list): Historical data to analyze
            numerical_cols (list): List of numerical columns to use
            sensitivity (float): Multiplier for threshold (higher = more sensitive)
            
        Returns:
            list: Detected anomalies with dates and details
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(historical_data, list):
            df = pd.DataFrame(historical_data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                dates = df.index
            else:
                dates = [i for i in range(len(df))]
        else:
            df = historical_data.copy()
            dates = df.index if isinstance(df.index, pd.DatetimeIndex) else [i for i in range(len(df))]
        
        # Select numerical columns if not specified
        if numerical_cols is None:
            numerical_cols = ['orders', 'deliveries', 'inventory', 'costs']
        
        # Ensure all required columns exist
        for col in numerical_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Normalize data
        df_scaled = pd.DataFrame(
            self.scaler.transform(df[numerical_cols]),
            columns=numerical_cols,
            index=df.index
        )
        
        # Create sequences
        sequences = self._create_sequences(df_scaled.values)
        
        # Calculate reconstruction errors
        reconstructions = self.model.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        
        # Detect anomalies
        anomalies = []
        adjusted_threshold = self.threshold * sensitivity
        
        for i, error in enumerate(mse):
            if error > adjusted_threshold:
                # Calculate contribution of each feature to the anomaly
                feature_errors = np.mean(np.square(sequences[i] - reconstructions[i]), axis=0)
                anomalous_features = []
                
                for j, feat_err in enumerate(feature_errors):
                    if feat_err > np.mean(feature_errors) + np.std(feature_errors):
                        anomalous_features.append(numerical_cols[j])
                
                # Get the date for this anomaly (accounting for lookback window)
                anomaly_date = dates[i + self.lookback - 1]
                if isinstance(anomaly_date, pd.Timestamp):
                    anomaly_date = anomaly_date.strftime("%Y-%m-%d")
                
                # Get the original values for the anomalous point
                values = {col: float(df[col].iloc[i + self.lookback - 1]) for col in numerical_cols}
                
                # Calculate severity (normalized error score)
                severity = min(1.0, error / (self.threshold * 2))
                
                # Generate potential causes
                causes = self._generate_potential_causes(anomalous_features, values)
                
                anomalies.append({
                    "date": anomaly_date,
                    "anomalous_metrics": anomalous_features,
                    "values": {col: values[col] for col in anomalous_features},
                    "severity": severity,
                    "reconstruction_error": float(error),
                    "potential_causes": causes
                })
        
        return anomalies
    
    def _generate_potential_causes(self, anomalous_features, values):
        """
        Generate potential causes for anomalies based on the affected metrics.
        
        Args:
            anomalous_features (list): List of anomalous features
            values (dict): Values of all features at the anomaly point
            
        Returns:
            list: Potential causes for the anomaly
        """
        causes = []
        
        if 'orders' in anomalous_features:
            if values.get('orders', 0) > values.get('deliveries', 0) * 1.3:
                causes.append("Unexpected demand surge")
            elif values.get('orders', 0) < values.get('deliveries', 0) * 0.7:
                causes.append("Significant drop in customer orders")
        
        if 'deliveries' in anomalous_features:
            if values.get('deliveries', 0) < values.get('orders', 0) * 0.7:
                causes.append("Possible supplier disruption or logistics issues")
            elif values.get('deliveries', 0) > values.get('orders', 0) * 1.3:
                causes.append("Unusual delivery pattern - possible inventory correction")
        
        if 'inventory' in anomalous_features:
            if values.get('inventory', 0) < 1500:
                causes.append("Critical inventory shortage")
            elif values.get('inventory', 0) > 8000:
                causes.append("Excess inventory buildup")
        
        if 'costs' in anomalous_features:
            if values.get('costs', 0) > 12000:
                causes.append("Unusual cost increase - check for price changes or emergency shipping")
            elif values.get('costs', 0) < 5000 and values.get('deliveries', 0) > 100:
                causes.append("Unusually low costs relative to delivery volume")
        
        # If multiple metrics are anomalous together
        if len(anomalous_features) >= 3:
            causes.append("Multiple metrics showing unusual patterns - possible major supply chain disruption")
        
        # If no specific causes identified, provide a generic message
        if not causes:
            causes.append("Unusual pattern detected - requires further investigation")
        
        return causes

# Example usage function
def deep_learning_forecast(historical_data, target_col="orders", forecast_days=14):
    """
    Generate a deep learning based forecast for supply chain metrics.
    
    Args:
        historical_data (list): List of dictionaries with historical data
        target_col (str): Column to forecast (e.g., "orders", "inventory")
        forecast_days (int): Number of days to forecast
        
    Returns:
        list: Forecasted values with dates and confidence intervals
    """
    try:
        # Initialize and train the forecaster
        forecaster = DeepLearningForecaster(lookback=30, horizon=forecast_days)
        forecaster.train(historical_data, target_col=target_col, epochs=30)
        
        # Generate forecast
        forecast = forecaster.forecast(historical_data, target_col=target_col)
        return forecast
    except Exception as e:
        print(f"Error in deep learning forecast: {e}")
        # Fall back to simpler forecasting method
        from agents.advanced_analytics import advanced_demand_forecast
        return advanced_demand_forecast(historical_data, forecast_days=forecast_days)

def deep_anomaly_detection(historical_data, sensitivity=1.0):
    """
    Detect anomalies in supply chain data using deep learning.
    
    Args:
        historical_data (list): List of dictionaries with historical data
        sensitivity (float): Sensitivity multiplier for anomaly detection
        
    Returns:
        list: Detected anomalies with dates and details
    """
    try:
        # Initialize and train the anomaly detector
        detector = DeepAnomalyDetector(lookback=14, threshold_percentile=95)
        detector.train(historical_data, epochs=30)
        
        # Detect anomalies
        anomalies = detector.detect_anomalies(historical_data, sensitivity=sensitivity)
        return anomalies
    except Exception as e:
        print(f"Error in deep anomaly detection: {e}")
        # Fall back to simpler anomaly detection method
        from agents.advanced_analytics import detect_supply_chain_anomalies
        return detect_supply_chain_anomalies(historical_data, sensitivity=0.05)