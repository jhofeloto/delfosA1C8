"""
Gradient Boosting Model for Diabetes Prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import logging

logger = logging.getLogger(__name__)

class DiabetesGradientBoosting:
    def __init__(self, **kwargs):
        self.params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 7,
            'min_samples_split': 5,
            'random_state': 42
        }
        self.params.update(kwargs)
        self.model = GradientBoostingRegressor(**self.params)
        self.is_fitted = False
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not trained")
        predictions = self.model.predict(X)
        return np.clip(predictions, 50, 400)
        
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return {
            'r2': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions)
        }
        
    def save(self, filepath):
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_fitted = True
