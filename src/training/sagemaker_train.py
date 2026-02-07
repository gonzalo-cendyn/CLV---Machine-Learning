"""
SageMaker Training Entry Point

This script is the entry point for SageMaker Training Jobs.
It handles the SageMaker-specific environment and imports.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import dill

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In SageMaker, we need to handle imports differently
# The source_dir is mounted and may not have the full package structure
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    from lifelines import CoxPHFitter
except ImportError as e:
    logger.error(f"Failed to import required libraries: {e}")
    logger.error("Make sure lifetimes and lifelines are in requirements.txt")
    raise


class DataProcessor:
    """Simplified data processor for SageMaker training."""
    
    def __init__(self, snapshot_date: str, xdays: int = 365):
        self.snapshot_date = pd.to_datetime(snapshot_date)
        self.xdays = xdays
    
    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """Load and clean transaction data."""
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Convert dates
        date_cols = ['Arrival Date', 'Departure Date', 'Booked Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Remove invalid dates
        df = df.dropna(subset=['Arrival Date'])
        df = df[df['Arrival Date'] <= self.snapshot_date]
        
        # Ensure revenue column exists
        if 'Total Revenue USD' not in df.columns:
            # Try alternatives
            revenue_cols = [c for c in df.columns if 'revenue' in c.lower() or 'amount' in c.lower()]
            if revenue_cols:
                df['Total Revenue USD'] = df[revenue_cols[0]]
            else:
                df['Total Revenue USD'] = 100.0  # Default
        
        # Clean revenue
        df['Total Revenue USD'] = pd.to_numeric(df['Total Revenue USD'], errors='coerce').fillna(0)
        df = df[df['Total Revenue USD'] > 0]
        
        logger.info(f"After cleaning: {len(df)} rows")
        return df
    
    def identify_customer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify one-timers and repeaters."""
        # Count transactions per customer
        tx_counts = df.groupby('CustomerID').size().reset_index(name='tx_count')
        
        # Merge back
        df = df.merge(tx_counts, on='CustomerID', how='left')
        df['customer_type'] = df['tx_count'].apply(lambda x: 'repeater' if x > 1 else 'one_timer')
        
        return df
    
    def prepare_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare customer-level data for Cox model."""
        # Group by customer
        customer_df = df.groupby('CustomerID').agg({
            'Arrival Date': ['min', 'max', 'count'],
            'Total Revenue USD': ['sum', 'mean'],
            'customer_type': 'first'
        }).reset_index()
        
        # Flatten column names
        customer_df.columns = ['CustomerID', 'first_arrival', 'last_arrival', 
                               'tx_count', 'total_revenue', 'avg_revenue', 'customer_type']
        
        # Calculate duration (T) - time since first transaction
        customer_df['T'] = (self.snapshot_date - customer_df['first_arrival']).dt.days
        
        # Event = 1 if customer returned (more than 1 transaction)
        customer_df['event'] = (customer_df['tx_count'] > 1).astype(int)
        
        # Calculate recency for returning customers
        customer_df['recency'] = (self.snapshot_date - customer_df['last_arrival']).dt.days
        
        return customer_df
    
    def preprocess_for_training(self, file_path: str) -> dict:
        """Full preprocessing pipeline."""
        # Load data
        df = self.load_and_clean_data(file_path)
        
        # Identify customer types
        df = self.identify_customer_types(df)
        
        # Prepare customer data
        customer_df = self.prepare_customer_data(df)
        
        # Split by customer type
        one_timers = df[df['customer_type'] == 'one_timer']
        repeaters = df[df['customer_type'] == 'repeater']
        
        logger.info(f"One-timers: {len(one_timers['CustomerID'].unique())}")
        logger.info(f"Repeaters: {len(repeaters['CustomerID'].unique())}")
        
        return {
            'transactions': df,
            'customer_data': customer_df,
            'one_timers': one_timers,
            'repeaters': repeaters
        }


def train_cox_model(customer_df: pd.DataFrame, penalizer: float = 0.0) -> CoxPHFitter:
    """Train Cox Proportional Hazards model."""
    logger.info("Training Cox PH Model...")
    
    # Prepare features
    df_model = customer_df[['T', 'event', 'avg_revenue', 'tx_count']].copy()
    df_model = df_model.dropna()
    
    # Ensure no zero or negative durations
    df_model = df_model[df_model['T'] > 0]
    
    logger.info(f"Training on {len(df_model)} customers")
    logger.info(f"Event rate: {df_model['event'].mean():.2%}")
    
    # Train model
    model = CoxPHFitter(penalizer=penalizer)
    model.fit(df_model, duration_col='T', event_col='event')
    
    logger.info("Cox model training complete")
    model.print_summary()
    
    return model


def train_bgnbd_model(summary_df: pd.DataFrame, penalizer: float = 0.0) -> BetaGeoFitter:
    """Train BG/NBD model for frequency prediction."""
    logger.info("Training BG/NBD Model...")
    
    model = BetaGeoFitter(penalizer_coef=penalizer)
    model.fit(
        summary_df['frequency'],
        summary_df['recency'],
        summary_df['T']
    )
    
    logger.info("BG/NBD model training complete")
    logger.info(f"Parameters: {model.summary}")
    
    return model


def train_gamma_gamma_model(summary_df: pd.DataFrame, penalizer: float = 0.0) -> GammaGammaFitter:
    """Train Gamma-Gamma model for monetary value prediction."""
    logger.info("Training Gamma-Gamma Model...")
    
    # Filter for valid monetary values
    df_valid = summary_df[summary_df['frequency'] > 0].copy()
    df_valid = df_valid[df_valid['monetary_value'] > 0]
    
    logger.info(f"Training on {len(df_valid)} customers with valid monetary data")
    
    model = GammaGammaFitter(penalizer_coef=penalizer)
    model.fit(
        df_valid['frequency'],
        df_valid['monetary_value']
    )
    
    logger.info("Gamma-Gamma model training complete")
    logger.info(f"Parameters: {model.summary}")
    
    return model


def save_models(models: dict, output_dir: str):
    """Save all trained models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Cox model with joblib
    cox_path = os.path.join(output_dir, 'cox_model.joblib')
    joblib.dump(models['cox'], cox_path)
    logger.info(f"Cox model saved to {cox_path}")
    
    # Save BG/NBD and Gamma-Gamma with dill (they have lambda functions)
    bgnbd_path = os.path.join(output_dir, 'bgnbd_model.pkl')
    with open(bgnbd_path, 'wb') as f:
        dill.dump(models['bgnbd'], f)
    logger.info(f"BG/NBD model saved to {bgnbd_path}")
    
    gg_path = os.path.join(output_dir, 'gamma_gamma_model.pkl')
    with open(gg_path, 'wb') as f:
        dill.dump(models['gamma_gamma'], f)
    logger.info(f"Gamma-Gamma model saved to {gg_path}")
    
    # Save config
    config = {
        'models': {
            'cox': 'cox_model.joblib',
            'bgnbd': 'bgnbd_model.pkl',
            'gamma_gamma': 'gamma_gamma_model.pkl'
        }
    }
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")


def main():
    """Main entry point for SageMaker training."""
    parser = argparse.ArgumentParser(description='Train CLV models on SageMaker')
    
    # SageMaker passes these environment variables
    parser.add_argument('--model-dir', type=str, 
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, 
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str, 
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    # Custom hyperparameters
    parser.add_argument('--snapshot-date', type=str, required=True,
                        help='Snapshot date (YYYY-MM-DD)')
    parser.add_argument('--xdays', type=int, default=365,
                        help='Prediction horizon in days')
    parser.add_argument('--penalizer', type=float, default=0.0,
                        help='L2 regularization')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("CLV MODEL TRAINING - SAGEMAKER")
    logger.info("=" * 60)
    logger.info(f"Model dir: {args.model_dir}")
    logger.info(f"Train dir: {args.train}")
    logger.info(f"Snapshot date: {args.snapshot_date}")
    logger.info(f"XDays: {args.xdays}")
    
    # Find training data
    train_file = None
    if os.path.exists(args.train):
        files = os.listdir(args.train)
        logger.info(f"Files in train dir: {files}")
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            train_file = os.path.join(args.train, csv_files[0])
    
    if not train_file:
        raise FileNotFoundError(f"No CSV file found in {args.train}")
    
    logger.info(f"Using training file: {train_file}")
    
    # Initialize processor
    processor = DataProcessor(
        snapshot_date=args.snapshot_date,
        xdays=args.xdays
    )
    
    # Load and preprocess data
    data = processor.preprocess_for_training(train_file)
    
    # Train Cox model on all customers
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COX MODEL")
    logger.info("=" * 60)
    cox_model = train_cox_model(data['customer_data'], args.penalizer)
    
    # Prepare summary data for repeaters
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING SUMMARY DATA FOR REPEATERS")
    logger.info("=" * 60)
    
    summary_df = summary_data_from_transaction_data(
        data['repeaters'],
        customer_id_col='CustomerID',
        datetime_col='Arrival Date',
        monetary_value_col='Total Revenue USD',
        observation_period_end=args.snapshot_date
    )
    logger.info(f"Summary data: {len(summary_df)} repeaters")
    
    # Train BG/NBD model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING BG/NBD MODEL")
    logger.info("=" * 60)
    bgnbd_model = train_bgnbd_model(summary_df, args.penalizer)
    
    # Train Gamma-Gamma model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING GAMMA-GAMMA MODEL")
    logger.info("=" * 60)
    gg_model = train_gamma_gamma_model(summary_df, args.penalizer)
    
    # Save models
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODELS")
    logger.info("=" * 60)
    
    models = {
        'cox': cox_model,
        'bgnbd': bgnbd_model,
        'gamma_gamma': gg_model
    }
    save_models(models, args.model_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
