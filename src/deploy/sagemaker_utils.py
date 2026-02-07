"""
SageMaker Utility Functions

Common utilities for SageMaker training and deployment.
"""

import os
import yaml
import boto3
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load SageMaker configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to project config
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'config', 'sagemaker_config.yaml'
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_sagemaker_session(region: str = None) -> 'sagemaker.Session':
    """
    Create a SageMaker session.
    
    Args:
        region: AWS region. If None, uses config default.
        
    Returns:
        SageMaker session
    """
    import sagemaker
    
    config = load_config()
    region = region or config['aws']['region']
    
    boto_session = boto3.Session(region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def get_execution_role(config: Dict[str, Any] = None) -> str:
    """
    Get the SageMaker execution role ARN.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        IAM role ARN
    """
    if config is None:
        config = load_config()
    
    return config['aws']['role_arn']


def get_s3_uri(bucket: str, prefix: str, filename: str = "") -> str:
    """
    Construct S3 URI.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        filename: Optional filename
        
    Returns:
        S3 URI string
    """
    if filename:
        return f"s3://{bucket}/{prefix}/{filename}"
    return f"s3://{bucket}/{prefix}"


def upload_to_s3(
    local_path: str,
    bucket: str,
    s3_key: str,
    region: str = None
) -> str:
    """
    Upload a file or directory to S3.
    
    Args:
        local_path: Local file or directory path
        bucket: S3 bucket name
        s3_key: S3 key (path in bucket)
        region: AWS region
        
    Returns:
        S3 URI of uploaded file
    """
    config = load_config()
    region = region or config['aws']['region']
    
    s3_client = boto3.client('s3', region_name=region)
    
    if os.path.isfile(local_path):
        s3_client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
    elif os.path.isdir(local_path):
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                s3_file_key = os.path.join(s3_key, relative_path).replace("\\", "/")
                s3_client.upload_file(local_file, bucket, s3_file_key)
                logger.info(f"Uploaded {local_file} to s3://{bucket}/{s3_file_key}")
    
    return f"s3://{bucket}/{s3_key}"


def create_model_tarball(model_dir: str, output_path: str = None) -> str:
    """
    Create a model.tar.gz file for SageMaker deployment.
    
    Args:
        model_dir: Directory containing model artifacts
        output_path: Output path for tarball. If None, creates in same dir.
        
    Returns:
        Path to created tarball
    """
    import tarfile
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_dir), 'model.tar.gz')
    
    with tarfile.open(output_path, 'w:gz') as tar:
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            tar.add(item_path, arcname=item)
    
    logger.info(f"Created model tarball: {output_path}")
    return output_path


def get_tags(config: Dict[str, Any] = None) -> list:
    """
    Get resource tags from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of tag dictionaries for SageMaker
    """
    if config is None:
        config = load_config()
    
    tags = config.get('tags', {})
    return [{'Key': k, 'Value': v} for k, v in tags.items()]
