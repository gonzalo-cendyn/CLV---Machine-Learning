"""
SageMaker Training Job Script

Submits a training job to AWS SageMaker.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import boto3
import sagemaker
from sagemaker.sklearn import SKLearn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.deploy.sagemaker_utils import load_config, get_tags

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def submit_training_job(
    data_s3_uri: str,
    snapshot_date: str,
    xdays: int = 365,
    config_path: str = None,
    wait: bool = True
) -> str:
    """
    Submit a training job to SageMaker.
    
    Args:
        data_s3_uri: S3 URI of training data
        snapshot_date: Snapshot date for training
        xdays: Prediction horizon
        config_path: Path to config file
        wait: Whether to wait for job completion
        
    Returns:
        S3 URI of trained model artifacts
    """
    config = load_config(config_path)
    
    region = config['aws']['region']
    role = config['aws']['role_arn']
    bucket = config['s3']['bucket']
    prefix = config['s3']['prefix']
    
    # Create SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Define output location
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = f"s3://{bucket}/{prefix}/training-jobs/{timestamp}"
    
    # Create SKLearn estimator
    sklearn_estimator = SKLearn(
        entry_point='train_all.py',
        source_dir=os.path.join(os.path.dirname(__file__), '..', 'src', 'training'),
        framework_version=config['framework']['version'],
        py_version=config['framework']['python_version'],
        instance_type=config['training_job']['instance_type'],
        instance_count=config['training_job']['instance_count'],
        role=role,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        base_job_name='clv-training',
        hyperparameters={
            'snapshot-date': snapshot_date,
            'xdays': xdays,
            'penalizer': 0.0
        },
        dependencies=['requirements.txt'],
        tags=get_tags(config)
    )
    
    logger.info(f"Submitting training job...")
    logger.info(f"Data: {data_s3_uri}")
    logger.info(f"Output: {output_path}")
    
    # Start training
    sklearn_estimator.fit(
        inputs={'train': data_s3_uri},
        wait=wait,
        logs='All'
    )
    
    if wait:
        model_artifacts = sklearn_estimator.model_data
        logger.info(f"Training complete. Model artifacts: {model_artifacts}")
        return model_artifacts
    else:
        logger.info("Training job submitted. Check SageMaker console for status.")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Submit SageMaker training job')
    
    parser.add_argument('--data-s3-uri', type=str, required=True,
                        help='S3 URI of training data')
    parser.add_argument('--snapshot-date', type=str, required=True,
                        help='Snapshot date (YYYY-MM-DD)')
    parser.add_argument('--xdays', type=int, default=365,
                        help='Prediction horizon in days')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for job completion')
    
    args = parser.parse_args()
    
    model_uri = submit_training_job(
        data_s3_uri=args.data_s3_uri,
        snapshot_date=args.snapshot_date,
        xdays=args.xdays,
        config_path=args.config,
        wait=not args.no_wait
    )
    
    print(f"\nModel artifacts: {model_uri}")


if __name__ == '__main__':
    main()
