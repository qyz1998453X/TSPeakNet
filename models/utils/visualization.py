# -*- coding: utf-8 -*-
"""
Visualization utilities for time series forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional


def setup_matplotlib(font_family: str = "SimHei") -> None:
    """
    Setup matplotlib with Chinese font support
    
    Args:
        font_family: Font family name
    """
    rcParams["font.sans-serif"] = [font_family]
    rcParams["axes.unicode_minus"] = False


def plot_forecast(
        dates: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 4),
        dpi: int = 150
) -> None:
    """
    Plot true vs predicted time series
    
    Args:
        dates: Date series
        y_true: True values (full series)
        y_pred: Predicted values (last part)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        dpi: DPI for saving
    """
    plt.figure(figsize=figsize)
    
    # Plot full true series
    plt.plot(dates, y_true, 'k-', label='原始数据', linewidth=1.5, alpha=0.7)
    
    # Plot predictions (last portion)
    pred_dates = dates.iloc[-len(y_pred):]
    plt.plot(pred_dates, y_pred, 'C1--', label='预测', linewidth=2)
    
    plt.xlabel('日期')
    plt.ylabel('值')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.close()


def plot_multi_channel(
        dates: pd.Series,
        true_values: dict,
        pred_values: dict,
        save_dir: str,
        title_prefix: str = "Forecast",
        figsize: tuple = (12, 4),
        dpi: int = 150
) -> None:
    """
    Plot forecasts for multiple channels
    
    Args:
        dates: Date series
        true_values: Dict of {channel: true_values}
        pred_values: Dict of {channel: pred_values}
        save_dir: Directory to save figures
        title_prefix: Prefix for plot titles
        figsize: Figure size
        dpi: DPI for saving
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for channel in true_values.keys():
        plot_forecast(
            dates=dates,
            y_true=true_values[channel],
            y_pred=pred_values[channel],
            title=f"{title_prefix} - {channel}",
            save_path=os.path.join(save_dir, f"{channel}.png"),
            figsize=figsize,
            dpi=dpi
        )


def plot_training_curve(
        losses: list,
        save_path: Optional[str] = None,
        title: str = "Training Loss",
        figsize: tuple = (10, 6),
        dpi: int = 150
) -> None:
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        dpi: DPI for saving
    """
    plt.figure(figsize=figsize)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.close()


def plot_metrics_comparison(
        results_df: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 6),
        dpi: int = 150
) -> None:
    """
    Plot metrics comparison across channels
    
    Args:
        results_df: DataFrame with columns [Node, MAE, RMSE, R2, MAPE]
        save_path: Path to save figure
        figsize: Figure size
        dpi: DPI for saving
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric in results_df.columns:
            results_df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(f'{metric} by Channel')
            ax.set_xlabel('Channel')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.close()
