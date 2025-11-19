# -*- coding: utf-8 -*-
"""
Evaluation metrics for time series forecasting
"""

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> float:
    """
    Calculate MAPE with safety check for small values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        threshold: Minimum value threshold
        
    Returns:
        MAPE percentage (or np.nan if no valid values)
    """
    mask = y_true >= threshold
    if not mask.any():
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (MAE, RMSE, R², MAPE)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    
    return mae, rmse, r2, mape


def print_metrics(
        name: str,
        mae: float,
        rmse: float,
        r2: float,
        mape: float
) -> None:
    """
    Pretty print evaluation metrics
    
    Args:
        name: Model/channel name
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: R² score
        mape: Mean Absolute Percentage Error
    """
    print(f"\n{'='*50}")
    print(f"Metrics for: {name}")
    print(f"{'='*50}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    if not np.isnan(mape):
        print(f"  MAPE : {mape:.2f}%")
    else:
        print(f"  MAPE : N/A (insufficient data)")
    print(f"{'='*50}\n")


def aggregate_metrics(results: list) -> dict:
    """
    Aggregate metrics across multiple channels
    
    Args:
        results: List of dicts with metrics
        
    Returns:
        Dict with mean and std of each metric
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Filter out NaN MAPE values
    mape_values = df['MAPE'].dropna()
    
    agg = {
        'MAE_mean': df['MAE'].mean(),
        'MAE_std': df['MAE'].std(),
        'RMSE_mean': df['RMSE'].mean(),
        'RMSE_std': df['RMSE'].std(),
        'R2_mean': df['R2'].mean(),
        'R2_std': df['R2'].std(),
        'MAPE_mean': mape_values.mean() if len(mape_values) > 0 else np.nan,
        'MAPE_std': mape_values.std() if len(mape_values) > 0 else np.nan,
    }
    
    return agg
