# -*- coding: utf-8 -*-
"""
Training script for Dual-Branch KAN-TimesNet model
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from dual_branch import DualBranchForecast
from data.dataloader import TimeSeriesDataLoader, create_dual_dataloaders
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import setup_matplotlib, plot_forecast

# Setup
warnings.filterwarnings("ignore", category=UserWarning)
setup_matplotlib(FONT_FAMILY)


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
        model: nn.Module,
        dataloader_s: DataLoader,
        dataloader_l: DataLoader,
        epochs: int,
        lr: float,
        device: torch.device,
        reg_weight: float = REG_WEIGHT
) -> list:
    """
    Train the dual-branch model
    
    Args:
        model: DualBranchForecast model
        dataloader_s: DataLoader for short windows
        dataloader_l: DataLoader for long windows
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        reg_weight: Regularization weight for KAN
        
    Returns:
        List of epoch losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    losses = []
    
    print(f"\n{'='*60}")
    print(f"Training for {epochs} epochs...")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Iterate through both dataloaders simultaneously
        for (xs, ys), (xl, yl) in zip(dataloader_s, dataloader_l):
            # Move to device
            xs = xs.to(device).unsqueeze(1)  # [B, 1, window_s]
            ys = ys.to(device)                # [B]
            xl = xl.to(device).unsqueeze(1)  # [B, 1, window_l]
            yl = yl.to(device)                # [B]
            
            # Forward pass
            pred = model(xs, xl)
            
            # Compute loss with KAN regularization
            loss = criterion(pred, ys)
            
            # Add KAN regularization from both branches
            reg_loss = (
                model.net_s.blocks[0].kan.regularization_loss()
                + model.net_l.blocks[0].kan.regularization_loss()
            )
            total_loss = loss + reg_weight * reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    print(f"\nTraining completed!")
    return losses


def online_finetune(
        model: nn.Module,
        xs: torch.Tensor,
        xl: torch.Tensor,
        y_true: torch.Tensor,
        steps: int,
        lr: float,
        device: torch.device
) -> None:
    """
    Perform online fine-tuning with one sample
    
    Args:
        model: Trained model
        xs: Short window input
        xl: Long window input
        y_true: True target value
        steps: Number of fine-tuning steps
        lr: Learning rate
        device: Device
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(xs, xl)
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()


def rolling_forecast(
        model: nn.Module,
        data_loader: TimeSeriesDataLoader,
        channel: str,
        window_s: int,
        window_l: int,
        device: torch.device,
        online_steps: int = ONLINE_STEPS,
        online_lr: float = LR_ONLINE
) -> tuple:
    """
    Perform rolling forecast with online fine-tuning
    
    Args:
        model: Trained model
        data_loader: Data loader with train_val and test splits
        channel: Channel name
        window_s: Short window size
        window_l: Long window size
        device: Device
        online_steps: Steps for online fine-tuning
        online_lr: Learning rate for online fine-tuning
        
    Returns:
        Tuple of (predictions, true_values)
    """
    train_val_df, test_df, _ = data_loader.get_data_splits()
    
    # Start with train_val history
    hist = train_val_df[channel].tolist()
    preds = []
    
    model.train()  # Keep in train mode for online fine-tuning
    
    print(f"\nPerforming rolling forecast for {len(test_df)} steps...")
    
    for idx, true_val in enumerate(test_df[channel].values):
        # Prepare inputs
        xs = torch.tensor(hist[-window_s:], dtype=torch.float32, device=device)
        xl = torch.tensor(hist[-window_l:], dtype=torch.float32, device=device)
        xs = xs.view(1, 1, -1)
        xl = xl.view(1, 1, -1)
        
        # Predict
        with torch.no_grad():
            pred = model(xs, xl).item()
        preds.append(pred)
        
        # Online fine-tuning with true value
        y_true = torch.tensor([true_val], dtype=torch.float32, device=device)
        online_finetune(model, xs, xl, y_true, online_steps, online_lr, device)
        
        # Update history with true value
        hist.append(true_val)
        
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{len(test_df)}")
    
    return np.array(preds), test_df[channel].values


def train_and_evaluate_channel(
        channel: str,
        data_loader: TimeSeriesDataLoader,
        device: torch.device,
        save_dir: str = PLOT_DIR
) -> dict:
    """
    Complete training and evaluation pipeline for one channel
    
    Args:
        channel: Channel name
        data_loader: Data loader
        device: Device
        save_dir: Directory to save plots
        
    Returns:
        Dict with metrics and predictions
    """
    print(f"\n{'#'*60}")
    print(f"# Processing Channel: {channel}")
    print(f"{'#'*60}")
    
    # Create dataloaders
    dl_s, dl_l = create_dual_dataloaders(
        data_loader, channel, WINDOW_S, WINDOW_L, BATCH_SZ
    )
    
    # Initialize model
    model = DualBranchForecast(
        c_in=C_IN,
        c_hid=C_HID,
        periods=PERIODS,
        out_len=OUT_LEN,
        kernel_size=KS
    ).to(device)
    
    # Train
    losses = train_model(model, dl_s, dl_l, EPOCHS, LR, device)
    
    # Rolling forecast
    y_pred_scaled, y_true_scaled = rolling_forecast(
        model, data_loader, channel, WINDOW_S, WINDOW_L, device
    )
    
    # Inverse transform to original scale
    y_pred = data_loader.inverse_transform(channel, y_pred_scaled)
    _, test_df, full_df = data_loader.get_data_splits()
    y_true = full_df[channel].iloc[-len(y_pred):].values
    
    # Clip negative predictions to zero
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    
    # Calculate metrics
    mae, rmse, r2, mape = calculate_metrics(y_true, y_pred)
    print_metrics(channel, mae, rmse, r2, mape)
    
    # Plot
    plot_forecast(
        dates=full_df['Date'],
        y_true=full_df[channel].values,
        y_pred=y_pred,
        title=f"{channel} · Dual-Branch Fusion",
        save_path=os.path.join(save_dir, f"{channel}.png"),
        figsize=FIGURE_SIZE,
        dpi=DPI
    )
    
    return {
        'Node': channel,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'predictions': y_pred,
        'losses': losses
    }


def main():
    """Main training pipeline"""
    # Set seed
    set_seed(SEED)
    
    # Load and prepare data
    print("Loading data...")
    data_loader = TimeSeriesDataLoader(
        EXCEL_PATH,
        CHANNELS,
        TRAIN_RATIO,
        VAL_RATIO
    )
    data_loader.prepare_data()
    
    # Train all channels
    all_results = []
    all_predictions = {}
    
    for channel in CHANNELS:
        result = train_and_evaluate_channel(channel, data_loader, DEVICE, PLOT_DIR)
        all_results.append({
            'Node': result['Node'],
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'R2': result['R2'],
            'MAPE': result['MAPE']
        })
        all_predictions[channel] = result['predictions']
    
    # Save results
    results_df = pd.DataFrame(all_results).set_index('Node')
    results_df.to_csv(RESULT_CSV)
    print(f"\nResults saved to: {RESULT_CSV}")
    
    # Save predictions
    _, test_df, _ = data_loader.get_data_splits()
    preds_df = pd.DataFrame(all_predictions, index=test_df['Date'])
    preds_df.to_excel(PREDICTION_XLSX, index_label='Date')
    print(f"Predictions saved to: {PREDICTION_XLSX}")
    
    print(f"\n{'='*60}")
    print("All channels processed successfully!")
    print(f"Plots saved in: {PLOT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
