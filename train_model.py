"""
Model training script for Exoplanet AI Classifier
Trains XGBoost model on KOI/TOI datasets with calibration and SHAP
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessor import DataPreprocessor
from src.models.exoplanet_classifier import ExoplanetClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    logger.info("Starting model training...")
    
    # Check if data files exist
    koi_file = "cumulative_2025.10.04_08.00.51.csv"
    toi_file = "TOI_2025.10.04_08.32.36.csv"
    
    if not os.path.exists(koi_file):
        logger.error(f"KOI file not found: {koi_file}")
        return
    
    if not os.path.exists(toi_file):
        logger.error(f"TOI file not found: {toi_file}")
        return
    
    try:
        # Initialize classifier
        classifier = ExoplanetClassifier()
        
        # Train model
        classifier.train_model(koi_file, toi_file)
        
        # Get feature importance
        feature_importance = classifier.get_feature_importance()
        
        logger.info("Top 10 most important features:")
        for i, feature in enumerate(feature_importance[:10]):
            logger.info(f"{i+1}. {feature['feature']}: {feature['importance']:.4f}")
        
        # Get metrics
        metrics = classifier.get_metrics()
        logger.info(f"Model accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.3f}")
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
