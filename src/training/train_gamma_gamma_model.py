"""
Gamma-Gamma Model Training Script

Trains a Gamma-Gamma model for predicting expected monetary value
of repeat customers.
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
import dill
from lifetimes import GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GammaGammaModelTrainer:
    """
    Trainer for Gamma-Gamma model.
    
    Used for predicting expected monetary value of repeat customers.
    The Gamma-Gamma model assumes that monetary value is independent of
    purchase frequency.
    """
    
    def __init__(self, penalizer_coef: float = 0.0):
        """
        Initialize the trainer.
        
        Args:
            penalizer_coef: L2 regularization strength (default: 0.0)
        """
        self.penalizer_coef = penalizer_coef
        self.model = None
        self.training_metrics = {}
    
    def prepare_data(
        self,
        summary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare data for Gamma-Gamma model training.
        
        Filters to customers with monetary_value > 0 and frequency > 0.
        
        Args:
            summary_df: Summary DataFrame with frequency, recency, T, monetary_value
            
        Returns:
            Filtered DataFrame ready for training
        """
        # Filter to valid customers
        df_valid = summary_df[
            (summary_df['monetary_value'] > 0) & 
            (summary_df['frequency'] > 0)
        ].copy()
        
        logger.info(f"Original customers: {len(summary_df)}")
        logger.info(f"Valid customers (monetary > 0, frequency > 0): {len(df_valid)}")
        logger.info(f"Average monetary value: ${df_valid['monetary_value'].mean():.2f}")
        logger.info(f"Median monetary value: ${df_valid['monetary_value'].median():.2f}")
        
        return df_valid
    
    def train(self, summary_df: pd.DataFrame) -> GammaGammaFitter:
        """
        Train the Gamma-Gamma model.
        
        Args:
            summary_df: Summary DataFrame with frequency and monetary_value
            
        Returns:
            Trained GammaGammaFitter model
        """
        logger.info("Training Gamma-Gamma model...")
        
        self.model = GammaGammaFitter(penalizer_coef=self.penalizer_coef)
        self.model.fit(
            summary_df['frequency'],
            summary_df['monetary_value']
        )
        
        # Store training metrics
        self.training_metrics = {
            'n_customers': len(summary_df),
            'avg_monetary_value': float(summary_df['monetary_value'].mean()),
            'median_monetary_value': float(summary_df['monetary_value'].median()),
            'avg_frequency': float(summary_df['frequency'].mean()),
            'penalizer_coef': self.penalizer_coef,
            'params': {k: float(v) for k, v in self.model.params_.items()}
        }
        
        logger.info("Training complete!")
        logger.info(f"Model parameters: {self.model.params_}")
        
        return self.model
    
    def predict_expected_value(self, summary_df: pd.DataFrame) -> np.ndarray:
        """
        Predict expected average profit per transaction.
        
        Args:
            summary_df: Summary DataFrame with frequency and monetary_value
            
        Returns:
            Array of expected values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.conditional_expected_average_profit(
            summary_df['frequency'],
            summary_df['monetary_value']
        )
    
    def calculate_clv(
        self,
        summary_df: pd.DataFrame,
        bgf_model,
        time: int = 365,
        discount_rate: float = 0.0
    ) -> np.ndarray:
        """
        Calculate Customer Lifetime Value using both BG/NBD and Gamma-Gamma models.
        
        CLV = Expected Purchases * Expected Value per Purchase
        
        Args:
            summary_df: Summary DataFrame
            bgf_model: Trained BetaGeoFitter model
            time: Prediction period in days
            discount_rate: Annual discount rate (default: 0.0)
            
        Returns:
            Array of CLV values
        """
        if self.model is None:
            raise ValueError("Gamma-Gamma model not trained. Call train() first.")
        
        # Get expected purchases from BG/NBD
        expected_purchases = bgf_model.conditional_expected_number_of_purchases_up_to_time(
            time,
            summary_df['frequency'],
            summary_df['recency'],
            summary_df['T']
        )
        
        # Get expected value per purchase from Gamma-Gamma
        expected_value = self.predict_expected_value(summary_df)
        
        # CLV = Expected Purchases * Expected Value
        clv = expected_purchases * expected_value
        
        # Apply discount if specified
        if discount_rate > 0:
            # Simple annual discounting
            years = time / 365
            discount_factor = 1 / (1 + discount_rate) ** years
            clv = clv * discount_factor
        
        return clv
    
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
        
        model_file = os.path.join(output_path, 'gamma_gamma_model.pkl')
        metrics_file = os.path.join(output_path, 'gamma_gamma_metrics.json')
        
        with open(model_file, 'wb') as f:
            dill.dump(self.model, f)
        
        with open(metrics_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        
        return model_file
    
    @classmethod
    def load_model(cls, model_path: str) -> 'GammaGammaModelTrainer':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            GammaGammaModelTrainer instance with loaded model
        """
        trainer = cls()
        with open(model_path, 'rb') as f:
            trainer.model = dill.load(f)
        logger.info(f"Model loaded from {model_path}")
        return trainer


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
    parser.add_argument('--penalizer-coef', type=float, default=0.0)
    
    args = parser.parse_args()
    
    logger.info(f"Starting Gamma-Gamma model training...")
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
    
    # Get repeater transactions
    df_repeaters = data['repeaters']
    
    logger.info(f"Repeater transactions: {len(df_repeaters)}")
    logger.info(f"Unique repeaters: {df_repeaters['CustomerID'].nunique()}")
    
    # Prepare summary data using lifetimes utility
    summary_df = summary_data_from_transaction_data(
        df_repeaters,
        customer_id_col='CustomerID',
        datetime_col='Arrival Date',
        monetary_value_col='Total Revenue USD',
        observation_period_end=args.snapshot_date
    )
    
    # Train model
    trainer = GammaGammaModelTrainer(penalizer_coef=args.penalizer_coef)
    df_valid = trainer.prepare_data(summary_df)
    trainer.train(df_valid)
    
    # Save model
    trainer.save_model(args.model_dir)
    
    # Log sample predictions
    sample_values = trainer.predict_expected_value(df_valid.head(5))
    logger.info(f"Sample expected values: {sample_values}")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
