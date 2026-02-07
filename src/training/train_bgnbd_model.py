"""
BG/NBD Model Training Script

Trains a Beta-Geometric/Negative Binomial Distribution model for predicting
purchase frequency of repeat customers.
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
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BGNBDModelTrainer:
    """
    Trainer for BG/NBD (Beta-Geometric/Negative Binomial Distribution) model.
    
    Used for predicting purchase frequency of repeat customers.
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
    
    def prepare_summary_data(
        self,
        df_transactions: pd.DataFrame,
        customer_id_col: str = 'CustomerID',
        datetime_col: str = 'Arrival Date',
        monetary_value_col: str = 'Total Revenue USD',
        observation_period_end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare RFM summary data from transaction-level data.
        
        Args:
            df_transactions: Transaction-level DataFrame
            customer_id_col: Name of customer ID column
            datetime_col: Name of datetime column
            monetary_value_col: Name of monetary value column
            observation_period_end: End date for observation period
            
        Returns:
            Summary DataFrame with frequency, recency, T, monetary_value
        """
        logger.info("Preparing summary data from transactions...")
        
        summary_df = summary_data_from_transaction_data(
            df_transactions,
            customer_id_col=customer_id_col,
            datetime_col=datetime_col,
            monetary_value_col=monetary_value_col,
            observation_period_end=observation_period_end
        )
        
        logger.info(f"Prepared summary for {len(summary_df)} customers")
        logger.info(f"Columns: {list(summary_df.columns)}")
        logger.info(f"Average frequency: {summary_df['frequency'].mean():.2f}")
        logger.info(f"Average recency: {summary_df['recency'].mean():.2f}")
        
        return summary_df
    
    def train(self, summary_df: pd.DataFrame) -> BetaGeoFitter:
        """
        Train the BG/NBD model.
        
        Args:
            summary_df: Summary DataFrame with frequency, recency, T columns
            
        Returns:
            Trained BetaGeoFitter model
        """
        logger.info("Training BG/NBD model...")
        
        self.model = BetaGeoFitter(penalizer_coef=self.penalizer_coef)
        self.model.fit(
            summary_df['frequency'],
            summary_df['recency'],
            summary_df['T']
        )
        
        # Store training metrics
        self.training_metrics = {
            'n_customers': len(summary_df),
            'avg_frequency': float(summary_df['frequency'].mean()),
            'avg_recency': float(summary_df['recency'].mean()),
            'avg_T': float(summary_df['T'].mean()),
            'penalizer_coef': self.penalizer_coef,
            'params': {k: float(v) for k, v in self.model.params_.items()}
        }
        
        logger.info("Training complete!")
        logger.info(f"Model parameters: {self.model.params_}")
        
        return self.model
    
    def predict_purchases(
        self,
        summary_df: pd.DataFrame,
        t: int = 365
    ) -> np.ndarray:
        """
        Predict expected number of purchases in next t days.
        
        Args:
            summary_df: Summary DataFrame with frequency, recency, T
            t: Prediction period in days
            
        Returns:
            Array of expected purchase counts
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.conditional_expected_number_of_purchases_up_to_time(
            t,
            summary_df['frequency'],
            summary_df['recency'],
            summary_df['T']
        )
    
    def predict_alive_probability(self, summary_df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability that customer is still "alive" (active).
        
        Args:
            summary_df: Summary DataFrame with frequency, recency, T
            
        Returns:
            Array of alive probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.conditional_probability_alive(
            summary_df['frequency'],
            summary_df['recency'],
            summary_df['T']
        )
    
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
        
        model_file = os.path.join(output_path, 'bgnbd_model.joblib')
        metrics_file = os.path.join(output_path, 'bgnbd_metrics.json')
        
        joblib.dump(self.model, model_file)
        
        with open(metrics_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        
        return model_file
    
    @classmethod
    def load_model(cls, model_path: str) -> 'BGNBDModelTrainer':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            BGNBDModelTrainer instance with loaded model
        """
        trainer = cls()
        trainer.model = joblib.load(model_path)
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
    
    logger.info(f"Starting BG/NBD model training...")
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
    
    # Prepare summary data
    trainer = BGNBDModelTrainer(penalizer_coef=args.penalizer_coef)
    summary_df = trainer.prepare_summary_data(
        df_repeaters,
        observation_period_end=args.snapshot_date
    )
    
    # Train model
    trainer.train(summary_df)
    
    # Save model
    trainer.save_model(args.model_dir)
    
    # Log sample predictions
    sample_predictions = trainer.predict_purchases(summary_df.head(5), t=args.xdays)
    logger.info(f"Sample predictions (next {args.xdays} days): {sample_predictions}")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
