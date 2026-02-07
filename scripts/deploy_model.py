"""
Deploy SageMaker Endpoint - Manual Script

Run this script to deploy a trained model as a SageMaker endpoint.

Usage:
    python scripts/deploy_model.py --model-uri s3://bucket/path/model.tar.gz
"""

import os
import sys
import argparse
import logging

import boto3
import sagemaker
from sagemaker.sklearn import SKLearnModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = 'us-east-2'
SAGEMAKER_ROLE = 'arn:aws:iam::443331476484:role/SageMakerCLVExecutionRole'
ENDPOINT_NAME = 'clv-prediction-endpoint'


def deploy_endpoint(model_uri: str, endpoint_name: str = ENDPOINT_NAME):
    """
    Deploy a trained model as a SageMaker endpoint.
    
    Args:
        model_uri: S3 URI of the model.tar.gz file
        endpoint_name: Name for the endpoint
    """
    logger.info("=" * 60)
    logger.info("DEPLOYING SAGEMAKER ENDPOINT")
    logger.info("=" * 60)
    logger.info(f"Model URI: {model_uri}")
    logger.info(f"Endpoint Name: {endpoint_name}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(project_root, 'src', 'inference')
    
    logger.info(f"Source directory: {source_dir}")
    
    # Create SageMaker session
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Create SKLearn model
    model = SKLearnModel(
        model_data=model_uri,
        role=SAGEMAKER_ROLE,
        entry_point='sagemaker_entry.py',
        source_dir=source_dir,
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session,
        dependencies=[os.path.join(source_dir, 'requirements.txt')]
    )
    
    logger.info("\nDeploying endpoint... (this may take 5-10 minutes)")
    
    # Check if endpoint exists
    sm_client = boto_session.client('sagemaker')
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} exists. Updating...")
        # Delete existing endpoint
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info("Waiting for existing endpoint to be deleted...")
        waiter = sm_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
    except sm_client.exceptions.ClientError:
        logger.info(f"Creating new endpoint: {endpoint_name}")
    
    # Deploy
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name,
        wait=True
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("ENDPOINT DEPLOYED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Endpoint Name: {endpoint_name}")
    logger.info(f"Endpoint ARN: arn:aws:sagemaker:{AWS_REGION}:443331476484:endpoint/{endpoint_name}")
    logger.info("\nTo invoke the endpoint, use:")
    logger.info(f"  python scripts/invoke_endpoint.py --endpoint-name {endpoint_name}")
    
    return predictor


def main():
    parser = argparse.ArgumentParser(description='Deploy SageMaker Endpoint')
    
    parser.add_argument('--model-uri', type=str, required=True,
                        help='S3 URI of model.tar.gz')
    parser.add_argument('--endpoint-name', type=str, default=ENDPOINT_NAME,
                        help=f'Endpoint name (default: {ENDPOINT_NAME})')
    
    args = parser.parse_args()
    
    deploy_endpoint(args.model_uri, args.endpoint_name)


if __name__ == '__main__':
    main()
