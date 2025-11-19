# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


class TimeSeriesDataLoader:
    """
    Load and preprocess time series data from Excel
    """
    
    def __init__(
            self,
            file_path: str,
            channels: list,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2
    ):
        """
        Args:
            file_path: Path to Excel file
            channels: List of channel names (columns)
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
        """
        self.file_path = file_path
        self.channels = channels
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Will be populated by load_data()
        self.df = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.df_train_val = None
        self.scalers = {}
        
    def load_data(self) -> None:
        """Load data from Excel and perform train/val/test split"""
        # Read Excel
        self.df = pd.read_excel(self.file_path, parse_dates=['Date'])
        
        # Calculate split indices
        n = len(self.df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        # Split data
        self.df_train = self.df.iloc[:n_train].copy()
        self.df_val = self.df.iloc[n_train:n_train+n_val].copy()
        self.df_test = self.df.iloc[n_train+n_val:].copy()
        
        # Combine train and val for scaling
        self.df_train_val = pd.concat(
            [self.df_train, self.df_val],
            ignore_index=True
        )
        
        print(f"Data loaded: Total={n}, Train={n_train}, Val={n_val}, Test={n-n_train-n_val}")
    
    def fit_scalers(self) -> None:
        """Fit scalers on training+validation data"""
        self.scalers = {}
        for ch in self.channels:
            scaler = StandardScaler()
            arr = self.df_train_val[ch].values.reshape(-1, 1)
            scaler.fit(arr)
            self.scalers[ch] = scaler
        
        print(f"Fitted {len(self.scalers)} scalers (z-score normalization)")
    
    def transform_data(self) -> None:
        """Apply scaling transformation to all splits"""
        for ch in self.channels:
            # Transform train_val
            self.df_train_val[ch] = self.scalers[ch].transform(
                self.df_train_val[[ch]]
            ).ravel()
            
            # Transform test
            self.df_test[ch] = self.scalers[ch].transform(
                self.df_test[[ch]]
            ).ravel()
        
        print("Data transformed (z-score scaled)")
    
    def inverse_transform(
            self,
            channel: str,
            values: np.ndarray
    ) -> np.ndarray:
        """
        Inverse transform scaled values back to original scale
        
        Args:
            channel: Channel name
            values: Scaled values
            
        Returns:
            Original scale values
        """
        return self.scalers[channel].inverse_transform(
            values.reshape(-1, 1)
        ).ravel()
    
    def get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get train_val, test dataframes and full dataframe
        
        Returns:
            Tuple of (train_val_df, test_df, full_df)
        """
        return self.df_train_val, self.df_test, self.df
    
    def prepare_data(self) -> None:
        """Complete data preparation pipeline"""
        self.load_data()
        self.fit_scalers()
        self.transform_data()


def create_dual_dataloaders(
        data_loader: TimeSeriesDataLoader,
        channel: str,
        window_s: int,
        window_l: int,
        batch_size: int = 128
) -> Tuple:
    """
    Create DataLoader objects for dual-branch model
    
    Args:
        data_loader: TimeSeriesDataLoader instance
        channel: Channel name
        window_s: Short window size
        window_l: Long window size
        batch_size: Batch size
        
    Returns:
        Tuple of (dataloader_s, dataloader_l)
    """
    from torch.utils.data import DataLoader
    from .dataset import WindowDataset
    
    # Get train_val data for the channel
    train_val_df, _, _ = data_loader.get_data_splits()
    series = train_val_df[channel].values
    
    # Create datasets
    ds_s = WindowDataset(series, window_s)
    ds_l = WindowDataset(series, window_l)
    
    # Create dataloaders
    dl_s = DataLoader(ds_s, batch_size=batch_size, shuffle=True)
    dl_l = DataLoader(ds_l, batch_size=batch_size, shuffle=True)
    
    return dl_s, dl_l
