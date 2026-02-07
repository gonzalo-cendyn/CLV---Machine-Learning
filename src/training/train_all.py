"""
Complete CLV Model Training Pipeline

Orchestrates training of all CLV models:
- Cox Proportional Hazards (one-timers)
- BG/NBD (repeater frequency)
- Gamma-Gamma (repeater monetary value)

Compatible with AWS SageMaker Training Jobs.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lifetimes.utils import summary_data_from_transaction_data

from src.preprocessing.data_processor import DataProcessor
from src.training.train_cox_model import CoxModelTrainer
from src.training.train_bgnbd_model import BGNBDModelTrainer
from src.training.train_gamma_gamma_model import GammaGammaModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLVModelPipeline:
    """
    Complete CLV model training pipeline.
    
    Orchestrates training of all models and saves them for deployment.
    """
    
    def __init__(
        self,
        snapshot_date: str,
        xdays: int = 365,
        penalizer: float = 0.0
    ):
        """
        Initialize the pipeline.
        
        Args:
            snapshot_date: Date when data snapshot was taken (YYYY-MM-DD)
            xdays: Prediction horizon in days
            penalizer: L2 regularization strength for models
        """
        self.snapshot_date = snapshot_date
        self.xdays = xdays
        self.penalizer = penalizer
        
        self.processor = DataProcessor(snapshot_date=snapshot_date, xdays=xdays)
        
        # Model trainers
        self.cox_trainer = None
        self.bgnbd_trainer = None
        self.gamma_gamma_trainer = None
        
        # Training data
        self.data = None
        self.summary_df = None
        
        # Metrics
        self.pipeline_metrics = {
            'snapshot_date': snapshot_date,
            'xdays': xdays,
            'penalizer': penalizer,
            'training_timestamp': None,
            'models': {}
        }
    
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess training data.
        
        Args:
            data_path: Path to transaction CSV file
            
        Returns:
            Dictionary of preprocessed DataFrames
        """
        logger.info(f"Loading data from {data_path}")
        self.data = self.processor.preprocess_for_training(data_path)
        
        # Log data statistics
        logger.info(f"Total transactions: {len(self.data['transactions'])}")
        logger.info(f"Total customers: {len(self.data['customer_data'])}")
        logger.info(f"One-timers: {len(self.data['one_timers']['CustomerID'].unique())}")
        logger.info(f"Repeaters: {len(self.data['repeaters']['CustomerID'].unique())}")
        
        return self.data
    
    def train_cox_model(self) -> CoxModelTrainer:
        """
        Train Cox PH model for predicting return probability.
        
        Note: We train on ALL customers to learn return patterns,
        then use the model to predict for one-timers only.
        
        Returns:
            Trained CoxModelTrainer
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING COX PROPORTIONAL HAZARDS MODEL")
        logger.info("="*60)
        
        # Get ALL customer data (need events=1 from repeaters to train)
        df_cust = self.data['customer_data']
        
        logger.info(f"Total customers: {len(df_cust)}")
        logger.info(f"Event rate (returned): {df_cust['event'].mean():.2%}")
        
        # Train model on all customers
        self.cox_trainer = CoxModelTrainer(penalizer=self.penalizer)
        df_model = self.cox_trainer.prepare_data(df_cust)
        self.cox_trainer.train(df_model)
        
        self.pipeline_metrics['models']['cox'] = self.cox_trainer.training_metrics
        
        return self.cox_trainer
    
    def train_bgnbd_model(self) -> BGNBDModelTrainer:
        """
        Train BG/NBD model for repeater frequency.
        
        Returns:
            Trained BGNBDModelTrainer
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING BG/NBD MODEL (Repeater Frequency)")
        logger.info("="*60)
        
        # Prepare summary data from repeater transactions
        df_repeaters = self.data['repeaters']
        
        self.summary_df = summary_data_from_transaction_data(
            df_repeaters,
            customer_id_col='CustomerID',
            datetime_col='Arrival Date',
            monetary_value_col='Total Revenue USD',
            observation_period_end=self.snapshot_date
        )
        
        logger.info(f"Repeater customers in summary: {len(self.summary_df)}")
        
        # Train model
        self.bgnbd_trainer = BGNBDModelTrainer(penalizer_coef=self.penalizer)
        self.bgnbd_trainer.train(self.summary_df)
        
        self.pipeline_metrics['models']['bgnbd'] = self.bgnbd_trainer.training_metrics
        
        return self.bgnbd_trainer
    
    def train_gamma_gamma_model(self) -> GammaGammaModelTrainer:
        """
        Train Gamma-Gamma model for repeater monetary value.
        
        Returns:
            Trained GammaGammaModelTrainer
        """
        logger.info("\n" + "="*60)
        logger.info("TRAINING GAMMA-GAMMA MODEL (Repeater Monetary Value)")
        logger.info("="*60)
        
        if self.summary_df is None:
            raise ValueError("Must train BG/NBD model first to prepare summary data")
        
        # Train model
        self.gamma_gamma_trainer = GammaGammaModelTrainer(penalizer_coef=self.penalizer)
        df_valid = self.gamma_gamma_trainer.prepare_data(self.summary_df)
        self.gamma_gamma_trainer.train(df_valid)
        
        self.pipeline_metrics['models']['gamma_gamma'] = self.gamma_gamma_trainer.training_metrics
        
        return self.gamma_gamma_trainer
    
    def train_all(self) -> Dict[str, Any]:
        """
        Train all models in sequence.
        
        Returns:
            Dictionary of all trained models
        """
        logger.info("\n" + "#"*60)
        logger.info("STARTING COMPLETE CLV MODEL TRAINING PIPELINE")
        logger.info("#"*60)
        logger.info(f"Snapshot Date: {self.snapshot_date}")
        logger.info(f"Prediction Horizon: {self.xdays} days")
        logger.info(f"Penalizer: {self.penalizer}")
        
        start_time = datetime.now()
        
        # Train all models
        self.train_cox_model()
        self.train_bgnbd_model()
        self.train_gamma_gamma_model()
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        self.pipeline_metrics['training_timestamp'] = end_time.isoformat()
        self.pipeline_metrics['training_duration_seconds'] = training_duration
        
        logger.info("\n" + "#"*60)
        logger.info("TRAINING COMPLETE")
        logger.info("#"*60)
        logger.info(f"Total training time: {training_duration:.2f} seconds")
        
        return {
            'cox': self.cox_trainer,
            'bgnbd': self.bgnbd_trainer,
            'gamma_gamma': self.gamma_gamma_trainer
        }
    
    def save_models(self, output_path: str) -> Dict[str, str]:
        """
        Save all trained models to disk.
        
        Args:
            output_path: Directory to save models
            
        Returns:
            Dictionary of model names to file paths
        """
        logger.info(f"\nSaving models to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        saved_paths = {}
        
        # Save Cox model
        if self.cox_trainer and self.cox_trainer.model:
            cox_path = os.path.join(output_path, 'cox')
            self.cox_trainer.save_model(cox_path)
            saved_paths['cox'] = cox_path
        
        # Save BG/NBD model
        if self.bgnbd_trainer and self.bgnbd_trainer.model:
            bgnbd_path = os.path.join(output_path, 'bgnbd')
            self.bgnbd_trainer.save_model(bgnbd_path)
            saved_paths['bgnbd'] = bgnbd_path
        
        # Save Gamma-Gamma model
        if self.gamma_gamma_trainer and self.gamma_gamma_trainer.model:
            gg_path = os.path.join(output_path, 'gamma_gamma')
            self.gamma_gamma_trainer.save_model(gg_path)
            saved_paths['gamma_gamma'] = gg_path
        
        # Save pipeline metrics
        metrics_file = os.path.join(output_path, 'pipeline_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.pipeline_metrics, f, indent=2, default=str)
        logger.info(f"Pipeline metrics saved to {metrics_file}")
        
        # Save configuration for inference
        config = {
            'snapshot_date': self.snapshot_date,
            'xdays': self.xdays,
            'penalizer': self.penalizer,
            'model_files': {
                'cox': 'cox/cox_model.joblib',
                'bgnbd': 'bgnbd/bgnbd_model.pkl',
                'gamma_gamma': 'gamma_gamma/gamma_gamma_model.pkl'
            }
        }
        config_file = os.path.join(output_path, 'model_config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Model config saved to {config_file}")
        
        return saved_paths
    
    @classmethod
    def load_models(cls, model_path: str) -> 'CLVModelPipeline':
        """
        Load trained models from disk.
        
        Args:
            model_path: Directory containing saved models
            
        Returns:
            CLVModelPipeline instance with loaded models
        """
        # Load configuration
        config_file = os.path.join(model_path, 'model_config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        pipeline = cls(
            snapshot_date=config['snapshot_date'],
            xdays=config['xdays'],
            penalizer=config['penalizer']
        )
        
        # Load models
        cox_path = os.path.join(model_path, config['model_files']['cox'])
        bgnbd_path = os.path.join(model_path, config['model_files']['bgnbd'])
        gg_path = os.path.join(model_path, config['model_files']['gamma_gamma'])
        
        pipeline.cox_trainer = CoxModelTrainer.load_model(cox_path)
        pipeline.bgnbd_trainer = BGNBDModelTrainer.load_model(bgnbd_path)
        pipeline.gamma_gamma_trainer = GammaGammaModelTrainer.load_model(gg_path)
        
        logger.info("All models loaded successfully")
        
        return pipeline


def main():
    """Main entry point for SageMaker Training Job."""
    parser = argparse.ArgumentParser(description='Train CLV models')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, 
                        default=os.environ.get('SM_MODEL_DIR', './models'))
    parser.add_argument('--train', type=str, 
                        default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--output-data-dir', type=str, 
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    
    # Training arguments
    parser.add_argument('--snapshot-date', type=str, required=True,
                        help='Snapshot date in YYYY-MM-DD format')
    parser.add_argument('--xdays', type=int, default=365,
                        help='Prediction horizon in days')
    parser.add_argument('--penalizer', type=float, default=0.0,
                        help='L2 regularization strength')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("CLV MODEL TRAINING - SAGEMAKER JOB")
    logger.info("="*60)
    logger.info(f"Arguments: {args}")
    
    # Initialize pipeline
    pipeline = CLVModelPipeline(
        snapshot_date=args.snapshot_date,
        xdays=args.xdays,
        penalizer=args.penalizer
    )
    
    # Find training data file
    train_file = None
    possible_names = ['transactions.csv', 'data.csv', 'train.csv', 'clv_data.csv', 'clv_training_data.csv']
    
    # First try direct file names
    for name in possible_names:
        candidate = os.path.join(args.train, name)
        if os.path.exists(candidate):
            train_file = candidate
            break
    
    # If not found, look for any CSV file
    if train_file is None and os.path.exists(args.train):
        csv_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
        if csv_files:
            train_file = os.path.join(args.train, csv_files[0])
            logger.info(f"Found CSV file: {csv_files[0]}")
    
    if train_file is None or not os.path.exists(train_file):
        available = os.listdir(args.train) if os.path.exists(args.train) else []
        raise FileNotFoundError(
            f"Training file not found. Available files in {args.train}: {available}"
        )
    
    logger.info(f"Using training file: {train_file}")
    
    # Load data
    pipeline.load_data(train_file)
    
    # Train all models
    pipeline.train_all()
    
    # Save models
    pipeline.save_models(args.model_dir)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING JOB COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()
