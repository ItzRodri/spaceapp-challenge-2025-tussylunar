"""
Metrics calculation utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from typing import Dict, List, Any

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive model metrics"""
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    
    # Precision-Recall AUC
    precision_recall_aucs = []
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(
            (y_true == i).astype(int), y_pred_proba[:, i]
        )
        pr_auc = auc(recall, precision)
        precision_recall_aucs.append(pr_auc)
    
    pr_auc_macro = np.mean(precision_recall_aucs)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification Report
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # F1 Scores
    f1_scores = [class_report[class_name]['f1-score'] for class_name in class_names]
    
    return {
        "roc_auc_macro": float(roc_auc),
        "pr_auc_macro": float(pr_auc_macro),
        "pr_auc_per_class": [float(score) for score in precision_recall_aucs],
        "f1_scores": [float(score) for score in f1_scores],
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report
    }

def calculate_reliability_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                 n_bins: int = 10) -> Dict[str, Any]:
    """Calculate reliability diagram metrics"""
    
    n_classes = y_pred_proba.shape[1]
    reliability_data = []
    
    for class_idx in range(n_classes):
        class_proba = y_pred_proba[:, class_idx]
        class_true = (y_true == class_idx).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_data = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (class_proba > bin_lower) & (class_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = class_true[in_bin].mean()
                avg_confidence_in_bin = class_proba[in_bin].mean()
                
                bin_data.append({
                    "bin_center": (bin_lower + bin_upper) / 2,
                    "accuracy": accuracy_in_bin,
                    "confidence": avg_confidence_in_bin,
                    "count": in_bin.sum(),
                    "proportion": prop_in_bin
                })
        
        reliability_data.append({
            "class": class_idx,
            "bins": bin_data
        })
    
    # Calculate Expected Calibration Error (ECE)
    ece = 0
    total_samples = len(y_true)
    
    for class_idx in range(n_classes):
        class_proba = y_pred_proba[:, class_idx]
        class_true = (y_true == class_idx).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (class_proba > bin_lower) & (class_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = class_true[in_bin].mean()
                avg_confidence_in_bin = class_proba[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        "reliability_data": reliability_data,
        "expected_calibration_error": float(ece)
    }
