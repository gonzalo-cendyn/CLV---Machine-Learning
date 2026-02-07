"""
Full CLV Deployment Pipeline

Complete end-to-end deployment:
1. Train models locally
2. Upload to S3
3. Create SageMaker model
4. Deploy endpoint

Usage:
    python scripts/deploy_full_pipeline.py --data-file path/to/data.csv --snapshot-date 2025-06-04
"""

import os
import sys
import argparse
import logging
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.train_all import CLVModelPipeline
from src.deploy.deploy_endpoint import SageMakerDeployer
from src.deploy.sagemaker_utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deploy_pipeline(
    data_file: str,
    snapshot_date: str,
    xdays: int = 365,
    config_path: str = None,
    wait_for_endpoint: bool = True
) -> dict:
    """
    Full deployment pipeline.
    
    Args:
        data_file: Path to training data CSV
        snapshot_date: Snapshot date for training
        xdays: Prediction horizon in days
        config_path: Path to SageMaker config
        wait_for_endpoint: Whether to wait for endpoint to be ready
        
    Returns:
        Deployment information dictionary
    """
    logger.info("="*60)
    logger.info("CLV FULL DEPLOYMENT PIPELINE")
    logger.info("="*60)
    logger.info(f"Data file: {data_file}")
    logger.info(f"Snapshot date: {snapshot_date}")
    logger.info(f"Prediction horizon: {xdays} days")
    
    # Create temp directory for models
    model_dir = tempfile.mkdtemp(prefix='clv_models_')
    logger.info(f"Model directory: {model_dir}")
    
    try:
        # Step 1: Train models locally
        logger.info("\n" + "="*60)
        logger.info("STEP 1: TRAINING MODELS LOCALLY")
        logger.info("="*60)
        
        pipeline = CLVModelPipeline(
            snapshot_date=snapshot_date,
            xdays=xdays,
            penalizer=0.0
        )
        
        pipeline.load_data(data_file)
        pipeline.train_all()
        pipeline.save_models(model_dir)
        
        logger.info("[OK] Models trained and saved")
        
        # Step 2: Deploy to SageMaker
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DEPLOYING TO SAGEMAKER")
        logger.info("="*60)
        
        deployer = SageMakerDeployer(config_path=config_path)
        
        deployment_info = deployer.deploy(
            model_dir=model_dir,
            wait=wait_for_endpoint
        )
        
        logger.info("[OK] Deployment complete")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("="*60)
        logger.info(f"Endpoint Name: {deployment_info['endpoint_name']}")
        logger.info(f"Region: {deployment_info['region']}")
        logger.info(f"Model Data: {deployment_info['model_data_url']}")
        
        return deployment_info
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            logger.info(f"Cleaned up temp directory: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Full CLV Deployment Pipeline')
    
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--snapshot-date', type=str, required=True,
                        help='Snapshot date (YYYY-MM-DD)')
    parser.add_argument('--xdays', type=int, default=365,
                        help='Prediction horizon in days')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to SageMaker config file')
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for endpoint to be ready')
    
    args = parser.parse_args()
    
    deployment_info = deploy_pipeline(
        data_file=args.data_file,
        snapshot_date=args.snapshot_date,
        xdays=args.xdays,
        config_path=args.config,
        wait_for_endpoint=not args.no_wait
    )
    
    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE!")
    print("="*60)
    print(f"\nEndpoint: {deployment_info['endpoint_name']}")
    print(f"Region: {deployment_info['region']}")
    print("\nTo test the endpoint:")
    print(f"  python scripts/invoke_endpoint.py --test")


if __name__ == '__main__':
    main()
