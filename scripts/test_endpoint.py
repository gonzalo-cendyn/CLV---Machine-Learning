"""
Test SageMaker CLV Endpoint

Invoke the deployed endpoint with sample data and display results.
"""

import json
import logging
import boto3

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = 'us-east-2'
ENDPOINT_NAME = 'clv-prediction-endpoint'


def test_endpoint():
    """Test the CLV prediction endpoint with sample data."""
    
    logger.info("=" * 60)
    logger.info("TESTING CLV PREDICTION ENDPOINT")
    logger.info("=" * 60)
    
    # Create SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    
    # Sample test data - mix of one-timers and repeaters
    test_payload = {
        "snapshot_date": "2025-06-04",
        "xdays": 365,
        "customers": [
            {
                "CustomerID": "GUEST-001",
                "transactions": [
                    {
                        "Arrival Date": "2024-11-15",
                        "Total Revenue USD": 350.00,
                        "Channel": "Direct"
                    }
                ]
            },
            {
                "CustomerID": "GUEST-002",
                "transactions": [
                    {
                        "Arrival Date": "2024-01-10",
                        "Total Revenue USD": 280.00,
                        "Channel": "OTA"
                    },
                    {
                        "Arrival Date": "2024-06-22",
                        "Total Revenue USD": 310.00,
                        "Channel": "Direct"
                    },
                    {
                        "Arrival Date": "2025-02-14",
                        "Total Revenue USD": 420.00,
                        "Channel": "Direct"
                    }
                ]
            },
            {
                "CustomerID": "GUEST-003",
                "transactions": [
                    {
                        "Arrival Date": "2023-05-20",
                        "Total Revenue USD": 150.00,
                        "Channel": "OTA"
                    },
                    {
                        "Arrival Date": "2024-03-18",
                        "Total Revenue USD": 200.00,
                        "Channel": "Direct"
                    }
                ]
            }
        ]
    }
    
    logger.info("\nTest Payload:")
    logger.info(f"- Snapshot Date: {test_payload['snapshot_date']}")
    logger.info(f"- Prediction Horizon: {test_payload['xdays']} days")
    logger.info(f"- Customers: {len(test_payload['customers'])}")
    for c in test_payload['customers']:
        logger.info(f"  - {c['CustomerID']}: {len(c['transactions'])} transaction(s)")
    
    logger.info("\nInvoking endpoint...")
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(test_payload)
    )
    
    # Parse response
    result = json.loads(response['Body'].read().decode())
    
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\nTotal predictions: {result['count']}")
    
    for pred in result['predictions']:
        logger.info("\n" + "-" * 40)
        logger.info(f"Customer: {pred['CustomerID']}")
        logger.info(f"  Type: {pred['group']}")
        logger.info(f"  CLV: ${pred['CLV']:,.2f}")
        logger.info(f"  CLV Level: {pred['CLV_Level']}")
        logger.info(f"  RFM Score: {pred['RFM_Score']}")
        logger.info(f"  Segment: {pred['Segment']}")
        
        if pred['group'] == 'onetimer':
            logger.info(f"  P(Return): {pred['P_return']:.2%}")
            logger.info(f"  Avg Spend: ${pred['avg_spend']:,.2f}")
        else:
            logger.info(f"  Expected Purchases: {pred['expected_purchases']:.2f}")
            logger.info(f"  Expected Value: ${pred['expected_value']:,.2f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return result


if __name__ == '__main__':
    test_endpoint()
