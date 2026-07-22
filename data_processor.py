# ================================================================
# Data Processing and Augmentation Pipeline
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import joblib

@dataclass
class TabletData:
    """Container for tablet formulation data"""
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    target_names: List[str]

class DataProcessor:
    """Data preprocessing and augmentation pipeline"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False
        
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic formulation data"""
        np.random.seed(self.random_state)
        
        data = []
        for _ in range(n_samples):
            # Random formulation
            api = np.random.uniform(80, 98)
            binder = np.random.uniform(1.4, 6.0)
            pvpp = np.random.uniform(1.0, 6.0)
            mgst = np.random.uniform(0.10, 1.2)
            mcc = np.random.uniform(1.5, 8.0)
            moisture = np.random.uniform(0.5, 5.0)
            pressure = np.random.uniform(150, 250)
            speed = np.random.uniform(15, 30)
            
            # Calculate properties with noise
            density = (0.55 + 0.4 * api/100) * (1 - 0.05 * mgst) * (1 + 0.02 * np.random.randn())
            tensile = (0.5 + 8.0 * binder/6.0) * (1 - 0.1 * mgst) * (1 + 0.03 * np.random.randn())
            efrf = (0.1 + 0.9 * mgst/1.2) * (1 + 0.1 * np.random.randn())
            disintegration = (2.0 + 45.0 * pvpp/6.0) * (1 + 0.05 * np.random.randn())
            dissolution = (10.0 + 80.0 * (1 - api/100)) * (1 + 0.05 * np.random.randn())
            
            data.append({
                'api': api,
                'binder': binder,
                'pvpp': pvpp,
                'mgst': mgst,
                'mcc': mcc,
                'moisture': moisture,
                'pressure': pressure,
                'speed': speed,
                'density': np.clip(density, 0.55, 0.95),
                'tensile': np.clip(tensile, 0.5, 8.5),
                'efrf': np.clip(efrf, 0.0, 1.0),
                'disintegration': np.clip(disintegration, 2.0, 47.0),
                'dissolution': np.clip(dissolution, 10.0, 90.0)
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, data: pd.DataFrame) -> TabletData:
        """Preprocess dataframe into features and targets"""
        feature_cols = ['api', 'binder', 'pvpp', 'mgst', 'mcc', 'moisture', 'pressure', 'speed']
        target_cols = ['density', 'tensile', 'efrf', 'disintegration', 'dissolution']
        
        features = data[feature_cols].values
        targets = data[target_cols].values
        
        return TabletData(
            features=features,
            targets=targets,
            feature_names=feature_cols,
            target_names=target_cols
        )
    
    def normalize_features(self, data: TabletData, fit: bool = True) -> np.ndarray:
        """Normalize feature matrix"""
        if fit or not self.is_fitted:
            return self.feature_scaler.fit_transform(data.features)
        else:
            return self.feature_scaler.transform(data.features)
    
    def normalize_targets(self, data: TabletData, fit: bool = True) -> np.ndarray:
        """Normalize target matrix"""
        if fit or not self.is_fitted:
            return self.target_scaler.fit_transform(data.targets)
        else:
            return self.target_scaler.transform(data.targets)
    
    def split_data(self, 
                   data: TabletData, 
                   test_size: float = 0.2, 
                   val_size: float = 0.1) -> Dict[str, TabletData]:
        """Split data into train, validation, test sets"""
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            data.features, data.targets, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        return {
            'train': TabletData(X_train, y_train, data.feature_names, data.target_names),
            'val': TabletData(X_val, y_val, data.feature_names, data.target_names),
            'test': TabletData(X_test, y_test, data.feature_names, data.target_names)
        }
    
    def augment_data(self, data: TabletData, noise_std: float = 0.01) -> TabletData:
        """Add noise for data augmentation"""
        noise = np.random.normal(0, noise_std, data.features.shape)
        augmented_features = data.features + noise
        return TabletData(
            features=augmented_features,
            targets=data.targets.copy(),
            feature_names=data.feature_names,
            target_names=data.target_names
        )
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers"""
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }, filepath)
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers"""
        scalers = joblib.load(filepath)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        self.is_fitted = True
