"""
Cox Proportional Hazards Model Training Script

Trains a Cox PH model for predicting return probability of one-time customers.
Compatible with AWS SageMaker Training Jobs.
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib
from lifelines import CoxPHFitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoxModelTrainer:
    """
    Trainer for Cox Proportional Hazards model.
    
    Used for predicting return probability of one-time customers.
    """
    
    def __init__(self, penalizer: float = 0.0):
        """
        Initialize the trainer.
        
        Args:
            penalizer: L2 regularization strength (default: 0.0)
        """
        self.penalizer = penalizer
        self.model = None
        self.training_metrics = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        duration_col: str = 'duration',
        event_col: str = 'event',
        feature_cols: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Prepare data for Cox model training.
        
        Args:
            df: Customer-level DataFrame
            duration_col: Name of duration column
            event_col: Name of event column (1 = returned, 0 = did not return)
            feature_cols: List of feature columns to include
            
        Returns:
            Prepared DataFrame for training
        """
        if feature_cols is None:
            feature_cols = ['avg_spend']
        
        required_cols = [duration_col, event_col] + feature_cols
        df_model = df[required_cols].dropna()
        
        logger.info(f"Prepared {len(df_model)} samples for training")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Event rate: {df_model[event_col].mean():.2%}")
        
        return df_model
    
    def train(
        self,
        df: pd.DataFrame,
        duration_col: str = 'duration',
        event_col: str = 'event'
    ) -> CoxPHFitter:
        """
        Train the Cox PH model.
        
        Args:
            df: Prepared DataFrame with duration, event, and features
            duration_col: Name of duration column
            event_col: Name of event column
            
        Returns:
            Trained CoxPHFitter model
        """
        logger.info("Training Cox Proportional Hazards model...")
        
        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(df, duration_col=duration_col, event_col=event_col)
        
        # Log training metrics
        self.training_metrics = {
            'concordance_index': self.model.concordance_index_,
            'log_likelihood': self.model.log_likelihood_,
            'n_samples': len(df),
            'n_events': int(df[event_col].sum()),
            'penalizer': self.penalizer
        }
        
        logger.info(f"Training complete!")
        logger.info(f"Concordance Index: {self.training_metrics['concordance_index']:.4f}")
        logger.info(f"Log Likelihood: {self.training_metrics['log_likelihood']:.4f}")
        
        return self.model
    
    def predict_survival_probability(
        self,
        df: pd.DataFrame,
        times: list = [365]
    ) -> pd.DataFrame:
        """
        Predict survival probability at given times.
        
        Args:
            df: DataFrame with features
            times: List of time points for prediction
            
        Returns:
            DataFrame with survival probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        surv_probs = self.model.predict_survival_function(df, times=times).T
        return surv_probs
    
    def predict_return_probability(
        self,
        df: pd.DataFrame,
        xdays: int = 365
    ) -> np.ndarray:
        """
        Predict return probability at xdays.
        
        Args:
            df: DataFrame with features
            xdays: Prediction horizon in days
            
        Returns:
            Array of return probabilities
        """
        surv_probs = self.predict_survival_probability(df, times=[xdays])
        return 1 - surv_probs[xdays].values
    
    def save_model(self, output_path: str) -> str:
        """
        Save the trained model to disk.
        
        Args:
            output_path: Directory to save the model
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(output_path, exist_ok=True)
        
        model_file = os.path.join(output_path, 'cox_model.joblib')
        metrics_file = os.path.join(output_path, 'cox_metrics.json')
        
        joblib.dump(self.model, model_file)
        
        with open(metrics_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        
        return model_file
    
    @classmethod
    def load_model(cls, model_path: str) -> 'CoxModelTrainer':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            CoxModelTrainer instance with loaded model
        """
        trainer = cls()
        trainer.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return trainer


def train_cox_model_by_channel(
    df: pd.DataFrame,
    channel_cols: list,
    output_path: str,
    xdays: int = 365
) -> Dict[str, CoxModelTrainer]:
    """
    Train separate Cox models for each channel group.
    
    Args:
        df: Customer-level DataFrame with channel dummy columns
        channel_cols: List of channel dummy column names
        output_path: Directory to save models
        xdays: Prediction horizon
        
    Returns:
        Dictionary of channel -> trained model
    """
    models = {}
    
    for channel in channel_cols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for channel: {channel}")
        logger.info(f"{'='*50}")
        
        # Filter data for this channel
        df_channel = df[df[channel] == True].copy()
        
        if len(df_channel) < 100:
            logger.warning(f"Skipping {channel}: insufficient data ({len(df_channel)} samples)")
            continue
        
        # Train model
        trainer = CoxModelTrainer()
        df_model = trainer.prepare_data(df_channel)
        trainer.train(df_model)
        
        # Save model
        channel_output = os.path.join(output_path, channel.replace('/', '_'))
        trainer.save_model(channel_output)
        
        models[channel] = trainer
    
    return models


def main():
    """Main training function for SageMaker."""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './models'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    # Training arguments
    parser.add_argument('--snapshot-date', type=str, required=True)
    parser.add_argument('--xdays', type=int, default=365)
    parser.add_argument('--penalizer', type=float, default=0.0)
    parser.add_argument('--by-channel', action='store_true', help='Train separate models per channel')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Cox model training...")
    logger.info(f"Arguments: {args}")
    
    # Import preprocessing (add parent to path)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.preprocessing.data_processor import DataProcessor
    
    # Initialize processor
    processor = DataProcessor(snapshot_date=args.snapshot_date, xdays=args.xdays)
    
    # Load and preprocess data
    train_file = os.path.join(args.train, 'transactions.csv')
    if os.path.exists(train_file):
        data = processor.preprocess_for_training(train_file)
    else:
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Get one-timer customer data
    df_cust = data['customer_data']
    df_onetimers = df_cust[df_cust['visit_count'] == 1].copy()
    
    logger.info(f"Total customers: {len(df_cust)}")
    logger.info(f"One-timers: {len(df_onetimers)}")
    
    if args.by_channel:
        # Train separate models per channel
        channel_cols = [col for col in df_onetimers.columns if col.startswith('Channel_')]
        models = train_cox_model_by_channel(
            df_onetimers, 
            channel_cols, 
            args.model_dir,
            args.xdays
        )
    else:
        # Train single model
        trainer = CoxModelTrainer(penalizer=args.penalizer)
        df_model = trainer.prepare_data(df_onetimers)
        trainer.train(df_model)
        trainer.save_model(args.model_dir)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
