"""
Delete SageMaker Endpoint

Cleans up SageMaker resources (endpoint, endpoint config, model).
"""

import os
import sys
import argparse
import logging

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.deploy.sagemaker_utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_endpoint(
    endpoint_name: str = None,
    delete_model: bool = True,
    delete_config: bool = True,
    config_path: str = None
) -> None:
    """
    Delete SageMaker endpoint and associated resources.
    
    Args:
        endpoint_name: Name of endpoint to delete. If None, uses config.
        delete_model: Also delete the model
        delete_config: Also delete the endpoint config
        config_path: Path to config file
    """
    config = load_config(config_path)
    region = config['aws']['region']
    
    if endpoint_name is None:
        endpoint_name = config['endpoint']['name']
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    # Get endpoint info before deleting
    try:
        endpoint_info = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        endpoint_config_name = endpoint_info['EndpointConfigName']
    except sagemaker_client.exceptions.ClientError as e:
        logger.error(f"Endpoint {endpoint_name} not found")
        return
    
    # Delete endpoint
    logger.info(f"Deleting endpoint: {endpoint_name}")
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    logger.info(f"Endpoint {endpoint_name} deleted")
    
    # Delete endpoint config
    if delete_config:
        try:
            # Get model name from config
            config_info = sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            model_name = config_info['ProductionVariants'][0]['ModelName']
            
            logger.info(f"Deleting endpoint config: {endpoint_config_name}")
            sagemaker_client.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Endpoint config {endpoint_config_name} deleted")
            
            # Delete model
            if delete_model:
                logger.info(f"Deleting model: {model_name}")
                sagemaker_client.delete_model(ModelName=model_name)
                logger.info(f"Model {model_name} deleted")
                
        except sagemaker_client.exceptions.ClientError as e:
            logger.warning(f"Could not delete associated resources: {e}")
    
    logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description='Delete SageMaker endpoint')
    
    parser.add_argument('--endpoint-name', type=str, default=None,
                        help='Endpoint name to delete')
    parser.add_argument('--keep-model', action='store_true',
                        help='Do not delete the model')
    parser.add_argument('--keep-config', action='store_true',
                        help='Do not delete the endpoint config')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    args = parser.parse_args()
    
    delete_endpoint(
        endpoint_name=args.endpoint_name,
        delete_model=not args.keep_model,
        delete_config=not args.keep_config,
        config_path=args.config
    )


if __name__ == '__main__':
    main()
