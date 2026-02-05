# CLV Machine Learning

Customer Lifetime Value (CLV) prediction models deployed as an API endpoint on AWS SageMaker.

## Overview

This project implements a CLV prediction system using:
- **Cox Proportional Hazards Model** (lifelines) for one-time customers
- **BG/NBD Model** (lifetimes) for repeat customer purchase frequency
- **Gamma-Gamma Model** (lifetimes) for repeat customer monetary value
- **RFM Analysis** for customer segmentation comparison

## Project Structure

```
CLV---Machine-Learning/
├── src/
│   ├── preprocessing/     # Data cleaning and feature engineering
│   ├── training/          # Model training scripts
│   ├── inference/         # SageMaker inference handler
│   └── deploy/            # Deployment scripts
├── config/                # Configuration files
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── notebooks/             # Jupyter notebooks (reference)
├── docs/                  # Documentation
├── docker/                # Docker configuration
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/gonzalo-cendyn/CLV---Machine-Learning.git
cd CLV---Machine-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```python
from src.preprocessing import DataProcessor

processor = DataProcessor(snapshot_date='2025-06-04')
df_clean = processor.load_and_clean('data/transactions.csv')
df_onetimers, df_repeaters = processor.split_customers(df_clean)
```

### Model Training

```bash
# Train all models
python -m src.training.train_all --input-data s3://bucket/data/ --output-model s3://bucket/models/
```

### API Endpoint

Once deployed, the API accepts POST requests with customer transaction data:

```json
{
  "snapshot_date": "2025-06-04",
  "xdays": 365,
  "customers": [
    {
      "CustomerID": "C001",
      "transactions": [
        {"Arrival Date": "2024-01-15", "Total Revenue USD": 500, "Channel": "WEB"}
      ]
    }
  ]
}
```

Response:

```json
{
  "predictions": [
    {
      "CustomerID": "C001",
      "CLV": 342.50,
      "group": "onetimer",
      "Segment": "Potential Loyalists",
      "CLV_Level": "Mid",
      "Mismatch": "Aligned"
    }
  ]
}
```

## Models

| Model | Purpose | Library |
|-------|---------|---------|
| Cox PH | Return probability for one-timers | lifelines |
| BG/NBD | Purchase frequency for repeaters | lifetimes |
| Gamma-Gamma | Expected monetary value for repeaters | lifetimes |

## License

Proprietary - Cendyn

## Contact

- Email: ggiosa@cendyn.com
