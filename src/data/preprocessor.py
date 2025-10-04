"""
Data preprocessing module for KOI/TOI exoplanet datasets
Handles cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.label_encoder = LabelEncoder()
        self.mission_scalers = {}
        self.is_fitted = False
        
        # Feature ranges for validation
        self.feature_ranges = {
            "orbital_period_days": (0.1, 1000),
            "transit_duration_hours": (0.05, 50),
            "transit_depth_ppm": (10, 500000),
            "stellar_radius_solar": (0.1, 100),
            "stellar_teff_K": (2000, 15000),
            "snr": (0, 10000)
        }
    
    def load_and_preprocess_koi_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess KOI (Kepler) data"""
        logger.info("Loading KOI data...")
        
        # Read the CSV file, skipping comment lines
        df = pd.read_csv(file_path, comment='#')
        
        # Map KOI columns to standard format
        column_mapping = {
            'koi_period': 'orbital_period_days',
            'koi_duration': 'transit_duration_hours',
            'koi_depth': 'transit_depth_ppm',
            'koi_srad': 'stellar_radius_solar',
            'koi_steff': 'stellar_teff_K',
            'koi_model_snr': 'snr',  # Use koi_model_snr instead of koi_snr
            'kepoi_name': 'id'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Add mission column
        df['mission'] = 'kepler'
        
        # Map disposition
        df['target'] = df['koi_pdisposition'].map({
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE_POSITIVE'
        })
        
        # Select required columns
        required_cols = ['mission', 'id'] + list(self.feature_ranges.keys()) + ['target']
        df = df[required_cols]
        
        logger.info(f"KOI data loaded: {len(df)} rows")
        return df
    
    def load_and_preprocess_toi_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess TOI (TESS) data"""
        logger.info("Loading TOI data...")
        
        # Read the CSV file, skipping comment lines
        df = pd.read_csv(file_path, comment='#')
        
        # Map TOI columns to standard format
        column_mapping = {
            'pl_orbper': 'orbital_period_days',
            'pl_trandurh': 'transit_duration_hours',  # pl_trandurh is in hours
            'pl_trandep': 'transit_depth_ppm',
            'st_rad': 'stellar_radius_solar',
            'st_teff': 'stellar_teff_K',
            'toi': 'id'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Add mission column
        df['mission'] = 'tess'
        
        # Add SNR column (use transit depth as proxy if not available)
        df['snr'] = df['transit_depth_ppm'] / 100  # Simple proxy
        
        # Map TFOPWG disposition
        df['target'] = df['tfopwg_disp'].map({
            'CP': 'CONFIRMED',
            'PC': 'CANDIDATE',
            'FP': 'FALSE_POSITIVE',
            'KP': 'CANDIDATE',  # Keep as candidate
            'APC': 'CANDIDATE'  # Keep as candidate
        })
        
        # Select required columns
        required_cols = ['mission', 'id'] + list(self.feature_ranges.keys()) + ['target']
        df = df[required_cols]
        
        logger.info(f"TOI data loaded: {len(df)} rows")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("Cleaning data...")
        
        original_len = len(df)
        
        # Remove rows with missing target
        df = df.dropna(subset=['target'])
        
        # Remove rows with missing critical features
        critical_features = ['orbital_period_days', 'transit_depth_ppm', 'stellar_radius_solar']
        df = df.dropna(subset=critical_features)
        
        # Clip outliers based on feature ranges
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                df[feature] = df[feature].clip(min_val, max_val)
        
        # Handle NaN values with imputation
        numeric_features = list(self.feature_ranges.keys())
        df[numeric_features] = self.imputer.fit_transform(df[numeric_features])
        
        logger.info(f"Data cleaning: {original_len} -> {len(df)} rows")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering"""
        logger.info("Engineering features...")
        
        # Log transformations for skewed features
        df['log_orbital_period'] = np.log10(df['orbital_period_days'] + 1e-6)
        df['log_transit_depth'] = np.log10(df['transit_depth_ppm'] + 1e-6)
        
        # Ratio features
        df['depth_duration_ratio'] = df['transit_depth_ppm'] / (df['transit_duration_hours'] + 1e-6)
        df['period_depth_ratio'] = df['orbital_period_days'] / (df['transit_depth_ppm'] + 1e-6)
        
        # Mission encoding (simple one-hot)
        df['mission_kepler'] = (df['mission'] == 'kepler').astype(int)
        df['mission_tess'] = (df['mission'] == 'tess').astype(int)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data"""
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode target variable
        if 'target' in df.columns:
            df['target_encoded'] = self.label_encoder.fit_transform(df['target'])
        
        self.is_fitted = True
        logger.info("Preprocessor fitted and data transformed")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Clean data (without removing rows)
        df = df.copy()
        
        # Clip outliers
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in df.columns:
                df[feature] = df[feature].clip(min_val, max_val)
        
        # Handle NaN values
        numeric_features = list(self.feature_ranges.keys())
        df[numeric_features] = self.imputer.transform(df[numeric_features])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Remove non-numeric columns that shouldn't be in the model
        columns_to_remove = ['mission', 'id', 'target', 'target_encoded']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of feature names for model training"""
        base_features = list(self.feature_ranges.keys())
        engineered_features = [
            'log_orbital_period', 'log_transit_depth',
            'depth_duration_ratio', 'period_depth_ratio',
            'mission_kepler', 'mission_tess'
        ]
        
        return base_features + engineered_features
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor to file"""
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'mission_scalers': self.mission_scalers,
            'is_fitted': self.is_fitted,
            'feature_ranges': self.feature_ranges
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor from file"""
        import joblib
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.label_encoder = data['label_encoder']
        self.mission_scalers = data['mission_scalers']
        self.is_fitted = data['is_fitted']
        self.feature_ranges = data['feature_ranges']
        logger.info(f"Preprocessor loaded from {filepath}")
