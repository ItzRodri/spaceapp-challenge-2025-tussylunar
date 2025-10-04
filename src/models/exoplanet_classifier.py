"""
Exoplanet Classification Model using XGBoost with calibration and SHAP explanations
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from typing import Dict, List, Any, Tuple
import os

logger = logging.getLogger(__name__)

class ExoplanetClassifier:
    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        self.shap_explainer = None
        self.is_trained = False
        self.training_metrics = {}
        
    def train_model(self, koi_path: str = "cumulative_2025.10.04_08.00.51.csv", 
                   toi_path: str = "TOI_2025.10.04_08.32.36.csv"):
        """Train the exoplanet classification model"""
        logger.info("Starting model training...")
        
        # Load and preprocess data
        from ..data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        # Load KOI and TOI data
        koi_df = preprocessor.load_and_preprocess_koi_data(koi_path)
        toi_df = preprocessor.load_and_preprocess_toi_data(toi_path)
        
        # Combine datasets
        combined_df = pd.concat([koi_df, toi_df], ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} rows")
        
        # Preprocess combined data
        processed_df = preprocessor.fit_transform(combined_df)
        
        # Prepare features and target
        self.feature_names = preprocessor.get_feature_names()
        X = processed_df[self.feature_names].fillna(0)
        y = processed_df['target_encoded']
        
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Classes distribution: {np.bincount(y)}")
        
        # Stratified train-test split by mission and class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=processed_df[['mission', 'target_encoded']]
        )
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=50
        )
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Create a new model for calibration (without early stopping)
        model_for_calibration = xgb.XGBClassifier(
            n_estimators=self.model.best_iteration if hasattr(self.model, 'best_iteration') else 1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Train the calibration model
        model_for_calibration.fit(X_train, y_train)
        
        # Calibrate the model
        self.calibrated_model = CalibratedClassifierCV(
            model_for_calibration, method='isotonic', cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        # Evaluate model
        self._evaluate_model(X_test, y_test)
        
        # Save model and preprocessor
        os.makedirs("models", exist_ok=True)
        self.save_model("models/trained_model.pkl")
        preprocessor.save_preprocessor("models/preprocessor.pkl")
        
        self.is_trained = True
        logger.info("Model training completed successfully")
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """Evaluate model performance"""
        # Predictions
        y_pred = self.calibrated_model.predict(X_test)
        y_pred_proba = self.calibrated_model.predict_proba(X_test)
        
        # Calculate metrics
        self.training_metrics = {
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.class_names, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc_macro': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro'),
            'accuracy': (y_pred == y_test).mean()
        }
        
        logger.info(f"Model accuracy: {self.training_metrics['accuracy']:.3f}")
        logger.info(f"ROC-AUC (macro): {self.training_metrics['roc_auc_macro']:.3f}")
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with probabilities and explanations"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get calibrated probabilities
        probas = self.calibrated_model.predict_proba(X)
        
        # Get predictions
        predictions = self.calibrated_model.predict(X)
        
        # Convert predictions to class names
        prediction_names = [self.class_names[pred] for pred in predictions]
        
        # Calculate confidence levels
        confidence = []
        for proba in probas:
            max_proba = np.max(proba)
            if max_proba >= 0.7:
                confidence.append("HIGH")
            elif max_proba >= 0.5:
                confidence.append("MEDIUM")
            else:
                confidence.append("LOW")
        
        # Format probabilities
        proba_dicts = []
        for proba in probas:
            proba_dict = {}
            for i, class_name in enumerate(self.class_names):
                proba_dict[class_name] = float(proba[i])
            proba_dicts.append(proba_dict)
        
        # Calculate SHAP values
        try:
            shap_values = self.shap_explainer.shap_values(X)
            
            # Format SHAP values for top 5 features
            shap_top5_list = []
            for i, shap_val in enumerate(shap_values):
                # Get top 5 features by absolute SHAP value
                feature_importance = list(zip(self.feature_names, shap_val))
                feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
                
                shap_top5 = []
                for feature, shap_score in feature_importance[:5]:
                    shap_top5.append({
                        "feature": feature,
                        "value": float(X.iloc[i][feature]) if feature in X.columns else 0.0,
                        "shap": float(shap_score)
                    })
                
                shap_top5_list.append(shap_top5)
        except Exception as e:
            # Fallback: use feature importance instead of SHAP
            logger.warning(f"SHAP calculation failed: {e}. Using feature importance instead.")
            shap_top5_list = []
            for i in range(len(X)):
                shap_top5 = []
                for j, feature in enumerate(self.feature_names[:5]):
                    shap_top5.append({
                        "feature": feature,
                        "value": float(X.iloc[i][feature]) if feature in X.columns else 0.0,
                        "shap": float(self.model.feature_importances_[j])
                    })
                shap_top5_list.append(shap_top5)
        
        return {
            "predictions": prediction_names,
            "probabilities": proba_dicts,
            "confidence": confidence,
            "shap_values": shap_top5_list
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        return self.training_metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'shap_explainer': self.shap_explainer,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.calibrated_model = model_data['calibrated_model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.shap_explainer = model_data['shap_explainer']
        self.is_trained = model_data['is_trained']
        self.training_metrics = model_data['training_metrics']
        
        # Ensure the model is marked as trained
        if self.model is not None and self.calibrated_model is not None:
            self.is_trained = True
            
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get global feature importance"""
        if not self.is_trained:
            return []
        
        importance = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {"feature": feature, "importance": float(imp)}
            for feature, imp in feature_importance
        ]
