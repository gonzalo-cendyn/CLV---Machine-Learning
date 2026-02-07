"""
Invoke SageMaker Endpoint

Client for making predictions using the deployed CLV model endpoint.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any

import boto3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.deploy.sagemaker_utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLVEndpointClient:
    """
    Client for invoking the CLV prediction endpoint.
    """
    
    def __init__(self, endpoint_name: str = None, region: str = None):
        """
        Initialize the client.
        
        Args:
            endpoint_name: Name of SageMaker endpoint
            region: AWS region
        """
        config = load_config()
        
        self.endpoint_name = endpoint_name or config['endpoint']['name']
        self.region = region or config['aws']['region']
        
        self.runtime_client = boto3.client(
            'sagemaker-runtime',
            region_name=self.region
        )
        
        logger.info(f"Client initialized for endpoint: {self.endpoint_name}")
    
    def predict(
        self,
        customers: List[Dict[str, Any]],
        snapshot_date: str = None,
        xdays: int = None
    ) -> Dict[str, Any]:
        """
        Get CLV predictions for customers.
        
        Args:
            customers: List of customer data dictionaries
            snapshot_date: Override snapshot date
            xdays: Override prediction horizon
            
        Returns:
            Prediction response dictionary
        """
        # Build request
        request_body = {
            'customers': customers
        }
        
        if snapshot_date:
            request_body['snapshot_date'] = snapshot_date
        if xdays:
            request_body['xdays'] = xdays
        
        # Invoke endpoint
        response = self.runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['Body'].read().decode())
        
        return response_body
    
    def predict_single(
        self,
        customer_id: str,
        transactions: List[Dict[str, Any]],
        snapshot_date: str = None,
        xdays: int = None
    ) -> Dict[str, Any]:
        """
        Get CLV prediction for a single customer.
        
        Args:
            customer_id: Customer identifier
            transactions: List of transaction dictionaries
            snapshot_date: Override snapshot date
            xdays: Override prediction horizon
            
        Returns:
            Single customer prediction
        """
        customer_data = [{
            'CustomerID': customer_id,
            'transactions': transactions
        }]
        
        response = self.predict(customer_data, snapshot_date, xdays)
        
        if response.get('predictions'):
            return response['predictions'][0]
        return None


def test_endpoint():
    """Test the endpoint with sample data."""
    
    # Sample customer data
    sample_customers = [
        {
            "CustomerID": "TEST001",
            "transactions": [
                {"Arrival Date": "2024-01-15", "Total Revenue USD": 500, "Channel": "WEB"},
                {"Arrival Date": "2024-06-20", "Total Revenue USD": 750, "Channel": "WEB"}
            ]
        },
        {
            "CustomerID": "TEST002",
            "transactions": [
                {"Arrival Date": "2024-03-10", "Total Revenue USD": 300, "Channel": "GDS"}
            ]
        }
    ]
    
    # Create client
    client = CLVEndpointClient()
    
    # Make prediction
    print("\nSending prediction request...")
    response = client.predict(
        customers=sample_customers,
        snapshot_date='2025-06-04',
        xdays=365
    )
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for pred in response.get('predictions', []):
        print(f"\nCustomer: {pred['CustomerID']}")
        print(f"  CLV: ${pred['CLV']:.2f}")
        print(f"  Group: {pred['group']}")
        print(f"  Segment: {pred['Segment']}")
        print(f"  CLV Level: {pred['CLV_Level']}")
        print(f"  Mismatch: {pred['Mismatch']}")
    
    print("\n" + "="*60)
    
    return response


def main():
    parser = argparse.ArgumentParser(description='Invoke CLV endpoint')
    
    parser.add_argument('--endpoint-name', type=str, default=None,
                        help='Endpoint name')
    parser.add_argument('--test', action='store_true',
                        help='Run test with sample data')
    parser.add_argument('--customer-data', type=str, default=None,
                        help='JSON file with customer data')
    parser.add_argument('--snapshot-date', type=str, default='2025-06-04',
                        help='Snapshot date')
    parser.add_argument('--xdays', type=int, default=365,
                        help='Prediction horizon')
    
    args = parser.parse_args()
    
    if args.test:
        test_endpoint()
    elif args.customer_data:
        with open(args.customer_data, 'r') as f:
            customers = json.load(f)
        
        client = CLVEndpointClient(endpoint_name=args.endpoint_name)
        response = client.predict(
            customers=customers,
            snapshot_date=args.snapshot_date,
            xdays=args.xdays
        )
        
        print(json.dumps(response, indent=2))
    else:
        print("Use --test for sample test or --customer-data <file> for custom data")


if __name__ == '__main__':
    main()
