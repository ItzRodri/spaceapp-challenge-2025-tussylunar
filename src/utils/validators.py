"""
Input validation utilities for exoplanet classification
"""

from typing import Dict, List, Any
import pandas as pd

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data for single prediction"""
    errors = []
    
    # Required fields
    required_fields = [
        "orbital_period_days", "transit_duration_hours", "transit_depth_ppm",
        "stellar_radius_solar", "stellar_teff_K"
    ]
    
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Range validation
    ranges = {
        "orbital_period_days": (0.1, 1000),
        "transit_duration_hours": (0.05, 50),
        "transit_depth_ppm": (10, 500000),
        "stellar_radius_solar": (0.1, 100),
        "stellar_teff_K": (2000, 15000),
        "snr": (0, 10000)
    }
    
    for field, (min_val, max_val) in ranges.items():
        if field in data and data[field] is not None:
            try:
                value = float(data[field])
                if value < min_val or value > max_val:
                    errors.append(f"{field} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid number")
    
    # Mission validation
    if "mission" in data and data["mission"] not in ["kepler", "tess"]:
        errors.append("mission must be 'kepler' or 'tess'")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate CSV structure for batch prediction"""
    errors = []
    warnings = []
    
    # Required columns
    required_columns = [
        "mission", "orbital_period_days", "transit_duration_hours",
        "transit_depth_ppm", "stellar_radius_solar", "stellar_teff_K"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    numeric_columns = [
        "orbital_period_days", "transit_duration_hours", "transit_depth_ppm",
        "stellar_radius_solar", "stellar_teff_K", "snr"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
            except:
                errors.append(f"Column {col} must contain only numeric values")
    
    # Check mission values
    if "mission" in df.columns:
        valid_missions = ["kepler", "tess"]
        invalid_missions = df[~df["mission"].isin(valid_missions)]["mission"].unique()
        if len(invalid_missions) > 0:
            errors.append(f"Invalid mission values: {list(invalid_missions)}")
    
    # Check for empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        warnings.append(f"Found {empty_rows} empty rows")
    
    # Check for missing values in critical columns
    critical_columns = ["orbital_period_days", "transit_depth_ppm", "stellar_radius_solar"]
    for col in critical_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                warnings.append(f"Column {col} has {missing_count} missing values")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
