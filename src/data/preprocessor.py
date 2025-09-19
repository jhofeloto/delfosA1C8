"""
Data Preprocessing Pipeline
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DiabetesDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_name = 'Resultado'
        
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records")
        return df
        
    def clean_data(self, df):
        df = df.dropna(subset=[self.target_name])
        return df
        
    def engineer_features(self, df):
        df = df.copy()
        if 'tas' in df.columns and 'tad' in df.columns:
            df['presion_arterial_media'] = (df['tas'] + 2*df['tad']) / 3
        if 'perimetro_abdominal' in df.columns and 'talla' in df.columns:
            df['ratio_cintura_altura'] = df['perimetro_abdominal'] / df['talla']
        return df
        
    def ingenieria_features_avanzada(self, df):
        """Extiende engineer_features con features avanzadas para análisis comparativo"""
        df = self.engineer_features(df)
        
        # Categorización de IMC
        if 'imc' in df.columns:
            df['indice_masa_corporal_cat'] = pd.cut(
                df['imc'],
                bins=[0, 18.5, 25, 30, 100],
                labels=[0, 1, 2, 3]
            ).astype('int64')
        
        return df
    
    def split_data(self, df, test_size=0.2):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != self.target_name]
        
        X = df[feature_cols]
        y = df[self.target_name]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)
        
    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
