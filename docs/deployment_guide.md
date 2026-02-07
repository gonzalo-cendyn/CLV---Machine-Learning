# CLV Model Deployment Guide

This guide explains how to deploy the CLV prediction model to AWS SageMaker.

## Prerequisites

1. **AWS Account** with SageMaker access
2. **AWS CLI** configured with credentials
3. **Python 3.9+** with required packages installed
4. **IAM Role** for SageMaker execution

### Required IAM Permissions

The SageMaker execution role needs these policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or specific bucket access)
- `CloudWatchLogsFullAccess`

## Configuration

Edit `config/sagemaker_config.yaml` with your AWS settings:

```yaml
aws:
  account_id: "YOUR_ACCOUNT_ID"
  region: "us-east-2"
  role_arn: "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE"

s3:
  bucket: "your-bucket-name"
  prefix: "clv-ml"

endpoint:
  name: "clv-prediction-endpoint"
  instance_type: "ml.m5.large"
  instance_count: 1
```

## Deployment Options

### Option 1: Full Pipeline (Recommended)

Train and deploy in one command:

```bash
python scripts/deploy_full_pipeline.py \
    --data-file path/to/transactions.csv \
    --snapshot-date 2025-06-04 \
    --xdays 365
```

### Option 2: Step by Step

#### Step 1: Train Models Locally

```bash
python scripts/test_pipeline.py \
    --data-file path/to/transactions.csv \
    --snapshot-date 2025-06-04
```

#### Step 2: Deploy to SageMaker

```bash
python -m src.deploy.deploy_endpoint \
    --model-dir ./models
```

### Option 3: SageMaker Training Job

Train on SageMaker infrastructure:

```bash
# First, upload data to S3
aws s3 cp transactions.csv s3://your-bucket/clv-ml/data/

# Submit training job
python scripts/train_sagemaker.py \
    --data-s3-uri s3://your-bucket/clv-ml/data/ \
    --snapshot-date 2025-06-04
```

## Testing the Endpoint

### Using the CLI

```bash
python -m src.deploy.invoke_endpoint --test
```

### Using Python

```python
from src.deploy.invoke_endpoint import CLVEndpointClient

client = CLVEndpointClient()

customers = [
    {
        "CustomerID": "C001",
        "transactions": [
            {"Arrival Date": "2024-01-15", "Total Revenue USD": 500, "Channel": "WEB"}
        ]
    }
]

response = client.predict(customers, snapshot_date='2025-06-04')
print(response)
```

### Using AWS CLI

```bash
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name clv-prediction-endpoint \
    --content-type application/json \
    --body '{"customers": [...], "snapshot_date": "2025-06-04"}' \
    response.json
```

## API Reference

### Request Format

```json
{
    "snapshot_date": "2025-06-04",
    "xdays": 365,
    "customers": [
        {
            "CustomerID": "C001",
            "transactions": [
                {
                    "Arrival Date": "2024-01-15",
                    "Total Revenue USD": 500,
                    "Channel": "WEB"
                }
            ]
        }
    ]
}
```

### Response Format

```json
{
    "predictions": [
        {
            "CustomerID": "C001",
            "CLV": 125.50,
            "group": "onetimer",
            "P_return": 0.251,
            "avg_spend": 500.0,
            "RFM_Score": 8,
            "Segment": "Potential Loyalists",
            "CLV_Level": "Mid",
            "Mismatch": "Aligned"
        }
    ],
    "count": 1
}
```

## Cleanup

Delete the endpoint when not needed:

```bash
python -m src.deploy.delete_endpoint
```

Or keep the model but delete the endpoint:

```bash
python -m src.deploy.delete_endpoint --keep-model
```

## Costs

Estimated costs (us-east-2):

| Resource | Instance | Cost/Hour |
|----------|----------|-----------|
| Endpoint | ml.t2.medium | ~$0.05 |
| Endpoint | ml.m5.large | ~$0.12 |
| Training | ml.m5.large | ~$0.12 |

**Note**: Endpoints incur costs while running. Delete when not in use.

## Troubleshooting

### Endpoint Creation Failed

1. Check CloudWatch logs for the endpoint
2. Verify IAM role permissions
3. Ensure S3 bucket is accessible

### Model Loading Errors

1. Verify model artifacts are correctly packaged
2. Check that all dependencies are in requirements.txt
3. Review SageMaker container logs

### Invocation Errors

1. Verify request JSON format
2. Check content type is `application/json`
3. Review endpoint CloudWatch logs
