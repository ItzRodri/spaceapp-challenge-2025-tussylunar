"""
Quick start script for Exoplanet AI Classifier
Checks dependencies, trains model if needed, and starts the web application
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'shap',
        'fastapi', 'uvicorn', 'plotly', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        logger.info("Dependencies installed successfully!")
    else:
        logger.info("All dependencies are installed!")

def check_model():
    """Check if trained model exists, train if not"""
    model_path = Path("models/trained_model.pkl")
    
    if not model_path.exists():
        logger.info("Trained model not found. Training new model...")
        try:
            subprocess.check_call([sys.executable, "train_model.py"])
            logger.info("Model training completed!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error training model: {e}")
            return False
    else:
        logger.info("Trained model found!")
    
    return True

def start_app():
    """Start the FastAPI application"""
    logger.info("Starting Exoplanet AI Classifier...")
    logger.info("Open your browser and navigate to: http://localhost:8000")
    
    try:
        subprocess.check_call([sys.executable, "main.py"])
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting application: {e}")

def main():
    """Main function"""
    logger.info("🚀 Exoplanet AI Classifier - NASA Space Apps Challenge 2025")
    logger.info("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        logger.error("Please run this script from the project root directory")
        return
    
    # Check dependencies
    check_dependencies()
    
    # Check/train model
    if not check_model():
        logger.error("Failed to prepare model. Exiting.")
        return
    
    # Start application
    start_app()

if __name__ == "__main__":
    main()
