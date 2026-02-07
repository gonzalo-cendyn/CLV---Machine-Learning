"""
Launch SageMaker Training Job - Manual Script

Run this script locally to submit a training job to SageMaker.
The training will execute on an EC2 instance managed by SageMaker.

Usage:
    python scripts/launch_training.py --snapshot-date 2025-06-04
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import boto3
import sagemaker
from sagemaker.sklearn import SKLearn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = 'us-east-2'
SAGEMAKER_ROLE = 'arn:aws:iam::443331476484:role/SageMakerCLVExecutionRole'
S3_BUCKET = 'sagemaker-us-east-2-443331476484'
S3_PREFIX = 'clv-ml'


def launch_training_job(snapshot_date: str, xdays: int = 365, wait: bool = True):
    """
    Launch a SageMaker training job.
    
    Args:
        snapshot_date: Date snapshot for training (YYYY-MM-DD)
        xdays: Prediction horizon in days
        wait: Whether to wait for job completion
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    logger.info("=" * 60)
    logger.info("LAUNCHING SAGEMAKER TRAINING JOB")
    logger.info("=" * 60)
    logger.info(f"Region: {AWS_REGION}")
    logger.info(f"Role: {SAGEMAKER_ROLE}")
    logger.info(f"Snapshot Date: {snapshot_date}")
    logger.info(f"Prediction Horizon: {xdays} days")
    logger.info(f"Timestamp: {timestamp}")
    
    # Create SageMaker session
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, 'src', 'training')
    requirements_file = os.path.join(project_root, 'requirements.txt')
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Source directory: {source_dir}")
    
    # Verify source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # S3 paths
    train_data_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/"
    output_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/models/{timestamp}"
    
    logger.info(f"Training data: {train_data_uri}")
    logger.info(f"Output path: {output_path}")
    
    # Create SKLearn estimator
    # SageMaker will:
    # 1. Package the source_dir and upload to S3
    # 2. Launch an EC2 instance with the SKLearn container
    # 3. Download training data and source code
    # 4. Run train_all.py with the specified hyperparameters
    # 5. Upload trained model to output_path
    
    estimator = SKLearn(
        entry_point='sagemaker_train.py',
        source_dir=source_dir,
        framework_version='1.2-1',
        py_version='py3',
        instance_type='ml.m5.large',  # 2 vCPU, 8 GB RAM
        instance_count=1,
        role=SAGEMAKER_ROLE,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        base_job_name='clv-training',
        hyperparameters={
            'snapshot-date': snapshot_date,
            'xdays': xdays,
            'penalizer': 0.0
        },
        max_run=3600,  # Max 1 hour
        tags=[
            {'Key': 'Project', 'Value': 'CLV-ML'},
            {'Key': 'Environment', 'Value': 'development'},
            {'Key': 'Owner', 'Value': 'ggiosa@cendyn.com'}
        ]
    )
    
    logger.info("\nSubmitting training job to SageMaker...")
    logger.info("SageMaker will provision an EC2 instance and run training.")
    logger.info("This may take 5-10 minutes.\n")
    
    # Submit training job
    estimator.fit(
        inputs={'train': train_data_uri},
        wait=wait,
        logs='All' if wait else None
    )
    
    if wait:
        # Training complete
        model_artifacts = estimator.model_data
        training_job_name = estimator.latest_training_job.name
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Training Job: {training_job_name}")
        logger.info(f"Model Artifacts: {model_artifacts}")
        logger.info("\nNext steps:")
        logger.info("1. Deploy endpoint: python scripts/deploy_endpoint.py")
        logger.info("2. Or use GitHub Actions for automated deployment")
        
        return {
            'training_job_name': training_job_name,
            'model_artifacts': model_artifacts
        }
    else:
        logger.info("\nTraining job submitted (not waiting for completion)")
        logger.info("Check AWS SageMaker Console for status:")
        logger.info(f"https://{AWS_REGION}.console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/jobs")
        return {'status': 'submitted'}


def main():
    parser = argparse.ArgumentParser(
        description='Launch SageMaker Training Job for CLV Models'
    )
    parser.add_argument(
        '--snapshot-date', 
        type=str, 
        required=True,
        help='Snapshot date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--xdays', 
        type=int, 
        default=365,
        help='Prediction horizon in days (default: 365)'
    )
    parser.add_argument(
        '--no-wait', 
        action='store_true',
        help='Submit job and exit without waiting for completion'
    )
    
    args = parser.parse_args()
    
    try:
        result = launch_training_job(
            snapshot_date=args.snapshot_date,
            xdays=args.xdays,
            wait=not args.no_wait
        )
        print(f"\nResult: {result}")
    except Exception as e:
        logger.error(f"Training job failed: {e}")
        raise


if __name__ == '__main__':
    main()
