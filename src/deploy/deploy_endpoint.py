"""
SageMaker Endpoint Deployment Script

Deploys the CLV model as a real-time inference endpoint on AWS SageMaker.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import boto3

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.deploy.sagemaker_utils import (
    load_config,
    get_execution_role,
    upload_to_s3,
    create_model_tarball,
    get_tags
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """
    Handles deployment of CLV model to SageMaker.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the deployer.
        
        Args:
            config_path: Path to SageMaker config file
        """
        self.config = load_config(config_path)
        self.region = self.config['aws']['region']
        self.role_arn = self.config['aws']['role_arn']
        
        # Initialize clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # Get configuration
        self.bucket = self.config['s3']['bucket']
        self.prefix = self.config['s3']['prefix']
        self.model_name = self.config['model']['name']
        self.endpoint_name = self.config['endpoint']['name']
        
        logger.info(f"Deployer initialized for region: {self.region}")
    
    def ensure_bucket_exists(self) -> None:
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"Bucket {self.bucket} exists")
        except:
            logger.info(f"Creating bucket {self.bucket}")
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"Bucket {self.bucket} created")
    
    def upload_model(self, model_dir: str) -> str:
        """
        Upload model artifacts to S3.
        
        Args:
            model_dir: Local directory containing trained models
            
        Returns:
            S3 URI of uploaded model tarball
        """
        logger.info(f"Uploading model from {model_dir}")
        
        # Create tarball
        tarball_path = create_model_tarball(model_dir)
        
        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f"{self.prefix}/models/{self.model_name}-{timestamp}/model.tar.gz"
        
        self.s3_client.upload_file(tarball_path, self.bucket, s3_key)
        
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.info(f"Model uploaded to {s3_uri}")
        
        # Clean up local tarball
        os.remove(tarball_path)
        
        return s3_uri
    
    def upload_inference_code(self) -> str:
        """
        Upload inference code to S3.
        
        Returns:
            S3 URI of source directory
        """
        # Get source directory
        src_dir = os.path.join(os.path.dirname(__file__), '..')
        
        # Upload key files
        files_to_upload = [
            ('preprocessing/data_processor.py', 'code/preprocessing/data_processor.py'),
            ('inference/inference_handler.py', 'code/inference/inference_handler.py'),
        ]
        
        for local_rel, s3_rel in files_to_upload:
            local_path = os.path.join(src_dir, local_rel)
            s3_key = f"{self.prefix}/{s3_rel}"
            if os.path.exists(local_path):
                self.s3_client.upload_file(local_path, self.bucket, s3_key)
                logger.info(f"Uploaded {local_rel}")
        
        return f"s3://{self.bucket}/{self.prefix}/code"
    
    def get_sklearn_image_uri(self) -> str:
        """
        Get the SageMaker SKLearn container image URI.
        
        Returns:
            Container image URI
        """
        import sagemaker
        
        return sagemaker.image_uris.retrieve(
            framework='sklearn',
            region=self.region,
            version=self.config['framework']['version'],
            py_version=self.config['framework']['python_version'],
            instance_type=self.config['endpoint']['instance_type']
        )
    
    def create_model(self, model_data_url: str) -> str:
        """
        Create a SageMaker model.
        
        Args:
            model_data_url: S3 URI of model artifacts
            
        Returns:
            Model name
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_name = f"{self.model_name}-{timestamp}"
        
        image_uri = self.get_sklearn_image_uri()
        
        logger.info(f"Creating model: {model_name}")
        logger.info(f"Image URI: {image_uri}")
        
        create_model_response = self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference_handler.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url,
                }
            },
            ExecutionRoleArn=self.role_arn,
            Tags=get_tags(self.config)
        )
        
        logger.info(f"Model created: {model_name}")
        return model_name
    
    def create_endpoint_config(self, model_name: str) -> str:
        """
        Create an endpoint configuration.
        
        Args:
            model_name: Name of the SageMaker model
            
        Returns:
            Endpoint configuration name
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_config_name = f"{self.endpoint_name}-config-{timestamp}"
        
        logger.info(f"Creating endpoint config: {endpoint_config_name}")
        
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': self.config['endpoint']['instance_type'],
                    'InitialInstanceCount': self.config['endpoint']['instance_count'],
                    'InitialVariantWeight': 1.0
                }
            ],
            Tags=get_tags(self.config)
        )
        
        logger.info(f"Endpoint config created: {endpoint_config_name}")
        return endpoint_config_name
    
    def create_endpoint(self, endpoint_config_name: str) -> str:
        """
        Create or update an endpoint.
        
        Args:
            endpoint_config_name: Name of endpoint configuration
            
        Returns:
            Endpoint name
        """
        endpoint_name = self.endpoint_name
        
        # Check if endpoint exists
        try:
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"Updating existing endpoint: {endpoint_name}")
            
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
        except self.sagemaker_client.exceptions.ClientError:
            logger.info(f"Creating new endpoint: {endpoint_name}")
            
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=get_tags(self.config)
            )
        
        logger.info(f"Endpoint deployment initiated: {endpoint_name}")
        return endpoint_name
    
    def wait_for_endpoint(self, endpoint_name: str, timeout_minutes: int = 15) -> bool:
        """
        Wait for endpoint to be in service.
        
        Args:
            endpoint_name: Name of the endpoint
            timeout_minutes: Maximum wait time
            
        Returns:
            True if endpoint is ready, False otherwise
        """
        import time
        
        logger.info(f"Waiting for endpoint {endpoint_name} to be ready...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            status = response['EndpointStatus']
            
            if status == 'InService':
                logger.info(f"Endpoint {endpoint_name} is now InService!")
                return True
            elif status == 'Failed':
                logger.error(f"Endpoint {endpoint_name} failed to deploy")
                return False
            
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.error(f"Timeout waiting for endpoint {endpoint_name}")
                return False
            
            logger.info(f"Endpoint status: {status}. Waiting...")
            time.sleep(30)
    
    def deploy(
        self,
        model_dir: str,
        wait: bool = True
    ) -> Dict[str, str]:
        """
        Full deployment pipeline.
        
        Args:
            model_dir: Local directory containing trained models
            wait: Whether to wait for endpoint to be ready
            
        Returns:
            Dictionary with deployment info
        """
        logger.info("="*60)
        logger.info("STARTING SAGEMAKER DEPLOYMENT")
        logger.info("="*60)
        
        # Ensure bucket exists
        self.ensure_bucket_exists()
        
        # Upload model
        model_data_url = self.upload_model(model_dir)
        
        # Create model
        model_name = self.create_model(model_data_url)
        
        # Create endpoint config
        endpoint_config_name = self.create_endpoint_config(model_name)
        
        # Create/update endpoint
        endpoint_name = self.create_endpoint(endpoint_config_name)
        
        # Wait for endpoint
        if wait:
            success = self.wait_for_endpoint(endpoint_name)
            if not success:
                raise RuntimeError("Endpoint deployment failed")
        
        deployment_info = {
            'model_name': model_name,
            'endpoint_config_name': endpoint_config_name,
            'endpoint_name': endpoint_name,
            'model_data_url': model_data_url,
            'region': self.region
        }
        
        logger.info("="*60)
        logger.info("DEPLOYMENT COMPLETE")
        logger.info("="*60)
        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"Region: {self.region}")
        
        return deployment_info


def main():
    """Main deployment entry point."""
    parser = argparse.ArgumentParser(description='Deploy CLV model to SageMaker')
    
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to SageMaker config file')
    parser.add_argument('--no-wait', action='store_true',
                        help='Do not wait for endpoint to be ready')
    
    args = parser.parse_args()
    
    deployer = SageMakerDeployer(config_path=args.config)
    
    deployment_info = deployer.deploy(
        model_dir=args.model_dir,
        wait=not args.no_wait
    )
    
    print("\nDeployment Info:")
    for key, value in deployment_info.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
