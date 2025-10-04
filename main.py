"""
Exoplanet Classification App - NASA Space Apps Challenge 2025
Main FastAPI application with Basic and Expert modes
"""

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import pandas as pd
import numpy as np
import json
import io
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

from src.models.exoplanet_classifier import ExoplanetClassifier
from src.data.preprocessor import DataPreprocessor
from src.utils.validators import validate_input_data
from src.utils.metrics import calculate_metrics
from src.utils.scientific_descriptions import ExoplanetDescriptor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exoplanet AI Classifier",
    description="NASA Space Apps Challenge 2025 - AI/ML Exoplanet Classification",
    version="1.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize models and preprocessor
classifier = None
preprocessor = None
descriptor = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global classifier, preprocessor, descriptor
    try:
        logger.info("Loading models...")
        classifier = ExoplanetClassifier()
        preprocessor = DataPreprocessor()
        descriptor = ExoplanetDescriptor()
        
        # Load pre-trained model if exists, otherwise train new one
        if os.path.exists("models/trained_model.pkl") and os.path.exists("models/preprocessor.pkl"):
            classifier.load_model("models/trained_model.pkl")
            preprocessor.load_preprocessor("models/preprocessor.pkl")
            logger.info("Pre-trained model and preprocessor loaded successfully")
        else:
            logger.info("Training new model...")
            classifier.train_model()
            logger.info("Model training completed")
            
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        import traceback
        traceback.print_exc()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with mode selection"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/basic", response_class=HTMLResponse)
async def basic_mode(request: Request):
    """Basic mode - single prediction form"""
    return templates.TemplateResponse("basic_mode.html", {"request": request})

@app.get("/expert", response_class=HTMLResponse)
async def expert_mode(request: Request):
    """Expert mode - batch CSV processing"""
    return templates.TemplateResponse("expert_mode.html", {"request": request})

@app.post("/api/predict/single")
async def predict_single(
    orbital_period_days: float = Form(...),
    transit_duration_hours: float = Form(...),
    transit_depth_ppm: float = Form(...),
    stellar_radius_solar: float = Form(...),
    stellar_teff_K: float = Form(...),
    snr: Optional[float] = Form(None),
    mission: str = Form("tess"),
    target_id: Optional[str] = Form(None)
):
    """API endpoint for single prediction (Basic Mode)"""
    try:
        # Validate input
        input_data = {
            "orbital_period_days": orbital_period_days,
            "transit_duration_hours": transit_duration_hours,
            "transit_depth_ppm": transit_depth_ppm,
            "stellar_radius_solar": stellar_radius_solar,
            "stellar_teff_K": stellar_teff_K,
            "snr": snr,
            "mission": mission
        }
        
        validation_result = validate_input_data(input_data)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["errors"])
        
        # Create DataFrame for prediction
        df = pd.DataFrame([input_data])
        
        # Preprocess and predict
        processed_data = preprocessor.transform(df)
        prediction_result = classifier.predict(processed_data)
        
        # Generate scientific description
        scientific_desc = descriptor.generate_scientific_description(
            prediction=prediction_result["predictions"][0],
            probabilities=prediction_result["probabilities"][0],
            orbital_period=orbital_period_days,
            transit_depth=transit_depth_ppm,
            transit_duration=transit_duration_hours,
            stellar_temp=stellar_teff_K,
            stellar_radius=stellar_radius_solar,
            snr=snr
        )
        
        # Format response
        result = {
            "id": target_id or f"PRED-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "mission": mission,
            "prediction": prediction_result["predictions"][0],
            "probas": prediction_result["probabilities"][0],
            "confidence": prediction_result["confidence"][0],
            "explainability": {
                "shap_top5": prediction_result["shap_values"][0]
            },
            "scientific_description": scientific_desc
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """API endpoint for batch prediction (Expert Mode)"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate CSV structure
        required_columns = [
            "mission", "orbital_period_days", "transit_duration_hours", 
            "transit_depth_ppm", "stellar_radius_solar", "stellar_teff_K"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Preprocess and predict
        processed_data = preprocessor.transform(df)
        prediction_result = classifier.predict(processed_data)
        
        # Add predictions to original DataFrame
        df["prediction"] = prediction_result["predictions"]
        df["confidence"] = prediction_result["confidence"]
        
        # Add probability columns
        for i, class_name in enumerate(["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]):
            df[f"prob_{class_name.lower()}"] = [probs[class_name] for probs in prediction_result["probabilities"]]
        
        # Return results
        return JSONResponse(content={
            "success": True,
            "total_rows": len(df),
            "predictions": df.to_dict("records"),
            "shap_values": prediction_result["shap_values"]
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Get model performance metrics"""
    try:
        metrics = classifier.get_metrics()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_stats():
    """Get prediction statistics for dashboard"""
    # Return static statistics
    stats = {
        "total_predictions": 1247,
        "confirmed_planets": 89,
        "candidates": 156,
        "false_positives": 1002
    }
    return stats

@app.get("/api/template")
async def download_template():
    """Download CSV template for expert mode"""
    template_data = {
        "mission": ["tess", "kepler"],
        "id": ["TOI-1234", "KOI-5678"],
        "orbital_period_days": [3.701, 12.5],
        "transit_duration_hours": [2.10, 4.5],
        "transit_depth_ppm": [850, 1200],
        "stellar_radius_solar": [0.92, 1.1],
        "stellar_teff_K": [5700, 6100],
        "snr": [12.4, 9.8]
    }
    
    template_df = pd.DataFrame(template_data)
    
    # Save to temporary file
    template_path = "static/exoplanet_template.csv"
    template_df.to_csv(template_path, index=False)
    
    return FileResponse(
        template_path, 
        media_type="text/csv",
        filename="exoplanet_template.csv"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
