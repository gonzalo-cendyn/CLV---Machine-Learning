"""
SageMaker SKLearn Container Entry Point

This module provides the entry point functions for SageMaker SKLearn inference.
It is loaded by the SageMaker SKLearn container and provides model loading,
input parsing, prediction, and output formatting.

Required functions:
- model_fn: Load models
- input_fn: Parse request
- predict_fn: Generate predictions
- output_fn: Format response
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import joblib
import dill

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
_cox_model = None
_bgnbd_model = None
_gamma_gamma_model = None
_config = None


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load models from the model directory.
    
    This function is called once when the container starts.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Dictionary containing all loaded models
    """
    global _cox_model, _bgnbd_model, _gamma_gamma_model, _config
    
    logger.info(f"Loading models from {model_dir}")
    
    # Load configuration
    config_path = os.path.join(model_dir, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            _config = json.load(f)
        logger.info(f"Config loaded: {_config}")
    else:
        _config = {
            'snapshot_date': '2025-06-04',
            'xdays': 365,
            'models': {
                'cox': 'cox_model.joblib',
                'bgnbd': 'bgnbd_model.pkl',
                'gamma_gamma': 'gamma_gamma_model.pkl'
            }
        }
    
    # Get model files - support both old and new config formats
    model_files = _config.get('models', _config.get('model_files', {}))
    
    # Load Cox model
    cox_path = os.path.join(model_dir, model_files.get('cox', 'cox_model.joblib'))
    if os.path.exists(cox_path):
        _cox_model = joblib.load(cox_path)
        logger.info("Cox model loaded")
    
    # Load BG/NBD model
    bgnbd_path = os.path.join(model_dir, model_files.get('bgnbd', 'bgnbd_model.pkl'))
    if os.path.exists(bgnbd_path):
        with open(bgnbd_path, 'rb') as f:
            _bgnbd_model = dill.load(f)
        logger.info("BG/NBD model loaded")
    
    # Load Gamma-Gamma model
    gg_path = os.path.join(model_dir, model_files.get('gamma_gamma', 'gamma_gamma_model.pkl'))
    if os.path.exists(gg_path):
        with open(gg_path, 'rb') as f:
            _gamma_gamma_model = dill.load(f)
        logger.info("Gamma-Gamma model loaded")
    
    return {
        'cox': _cox_model,
        'bgnbd': _bgnbd_model,
        'gamma_gamma': _gamma_gamma_model,
        'config': _config
    }


def input_fn(request_body: str, request_content_type: str) -> Dict[str, Any]:
    """
    Parse incoming request.
    
    Args:
        request_body: Raw request body
        request_content_type: Content type of request
        
    Returns:
        Parsed request dictionary
    """
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        logger.info(f"Received request with {len(request.get('customers', []))} customers")
        return request
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: Dict[str, Any], models: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate predictions.
    
    Args:
        input_data: Parsed request data
        models: Dictionary of loaded models
        
    Returns:
        List of predictions
    """
    from lifetimes.utils import summary_data_from_transaction_data
    
    customer_data = input_data.get('customers', [])
    snapshot_date = input_data.get('snapshot_date', _config.get('snapshot_date', '2025-06-04'))
    xdays = input_data.get('xdays', _config.get('xdays', 365))
    
    snapshot_date = pd.to_datetime(snapshot_date)
    
    if not customer_data:
        return []
    
    # Convert to DataFrame
    all_transactions = []
    for customer in customer_data:
        customer_id = customer['CustomerID']
        for txn in customer['transactions']:
            all_transactions.append({
                'CustomerID': customer_id,
                'Arrival Date': pd.to_datetime(txn['Arrival Date']),
                'Total Revenue USD': txn['Total Revenue USD'],
                'Channel': txn.get('Channel', 'Unknown')
            })
    
    df_txn = pd.DataFrame(all_transactions)
    
    # Identify one-timers vs repeaters
    visit_counts = df_txn['CustomerID'].value_counts()
    one_timer_ids = visit_counts[visit_counts == 1].index.tolist()
    repeater_ids = visit_counts[visit_counts > 1].index.tolist()
    
    # Aggregate customer data
    df_cust = df_txn.groupby('CustomerID').agg({
        'Arrival Date': ['min', 'max', 'count'],
        'Total Revenue USD': 'mean'
    }).reset_index()
    df_cust.columns = ['CustomerID', 'first_date', 'last_date', 'visit_count', 'avg_spend']
    df_cust['duration'] = (snapshot_date - df_cust['first_date']).dt.days
    df_cust['recency'] = (snapshot_date - df_cust['last_date']).dt.days
    
    # Calculate RFM
    df_rfm = df_txn.groupby('CustomerID').agg({
        'Arrival Date': lambda x: (snapshot_date - x.max()).days,
        'CustomerID': 'count',
        'Total Revenue USD': 'sum'
    })
    df_rfm.columns = ['Recency', 'Frequency', 'Monetary']
    df_rfm = df_rfm.reset_index()
    
    # Simple RFM scoring
    df_rfm['RFM_Score'] = 9  # Default middle score
    df_rfm['Segment'] = 'Potential Loyalists'
    
    predictions = []
    
    # Process one-timers
    for cid in one_timer_ids:
        cust = df_cust[df_cust['CustomerID'] == cid].iloc[0]
        
        # Predict with Cox model
        if _cox_model is not None:
            # Use the same column names as training: avg_revenue, tx_count
            df_pred = pd.DataFrame({
                'avg_revenue': [cust['avg_spend']],
                'tx_count': [int(cust['visit_count'])]
            })
            try:
                surv_prob = _cox_model.predict_survival_function(df_pred, times=[xdays])
                p_return = 1 - surv_prob.iloc[0, 0]
            except Exception as e:
                logger.warning(f"Cox prediction failed for {cid}: {e}")
                p_return = 0.1
        else:
            p_return = 0.1
        
        clv = p_return * cust['avg_spend']
        
        # Get RFM
        rfm = df_rfm[df_rfm['CustomerID'] == cid]
        rfm_score = int(rfm['RFM_Score'].values[0]) if len(rfm) > 0 else 5
        segment = rfm['Segment'].values[0] if len(rfm) > 0 else 'Unknown'
        
        predictions.append({
            'CustomerID': cid,
            'CLV': round(float(clv), 2),
            'group': 'onetimer',
            'P_return': round(float(p_return), 4),
            'avg_spend': round(float(cust['avg_spend']), 2),
            'RFM_Score': rfm_score,
            'Segment': segment,
            'CLV_Level': _assign_clv_level(clv, True),
            'Mismatch': 'Aligned'
        })
    
    # Process repeaters
    if repeater_ids:
        df_repeaters = df_txn[df_txn['CustomerID'].isin(repeater_ids)]
        
        summary_df = summary_data_from_transaction_data(
            df_repeaters,
            customer_id_col='CustomerID',
            datetime_col='Arrival Date',
            monetary_value_col='Total Revenue USD',
            observation_period_end=snapshot_date
        )
        
        for cid in repeater_ids:
            if cid not in summary_df.index:
                continue
            
            cust_summary = summary_df.loc[cid]
            
            # Predict with BG/NBD
            if _bgnbd_model is not None:
                expected_purchases = _bgnbd_model.conditional_expected_number_of_purchases_up_to_time(
                    xdays,
                    cust_summary['frequency'],
                    cust_summary['recency'],
                    cust_summary['T']
                )
            else:
                expected_purchases = cust_summary['frequency'] * (xdays / max(cust_summary['T'], 1))
            
            # Predict with Gamma-Gamma
            if _gamma_gamma_model is not None and cust_summary['frequency'] > 0:
                expected_value = _gamma_gamma_model.conditional_expected_average_profit(
                    cust_summary['frequency'],
                    cust_summary['monetary_value']
                )
            else:
                expected_value = cust_summary['monetary_value']
            
            clv = expected_purchases * expected_value
            
            # Get RFM
            rfm = df_rfm[df_rfm['CustomerID'] == cid]
            rfm_score = int(rfm['RFM_Score'].values[0]) if len(rfm) > 0 else 5
            segment = rfm['Segment'].values[0] if len(rfm) > 0 else 'Unknown'
            
            predictions.append({
                'CustomerID': cid,
                'CLV': round(float(clv), 2),
                'group': 'repeater',
                'expected_purchases': round(float(expected_purchases), 2),
                'expected_value': round(float(expected_value), 2),
                'frequency': int(cust_summary['frequency']),
                'RFM_Score': rfm_score,
                'Segment': segment,
                'CLV_Level': _assign_clv_level(clv, False),
                'Mismatch': 'Aligned'
            })
    
    return predictions


def _assign_clv_level(clv: float, is_onetimer: bool) -> str:
    """Assign CLV level based on value."""
    if is_onetimer:
        if clv < 50:
            return 'Low'
        elif clv < 150:
            return 'Mid'
        elif clv < 300:
            return 'High'
        else:
            return 'Top'
    else:
        if clv < 200:
            return 'Low'
        elif clv < 500:
            return 'Mid'
        elif clv < 1000:
            return 'High'
        else:
            return 'Top'


def output_fn(prediction: List[Dict[str, Any]], response_content_type: str) -> str:
    """
    Format response.
    
    Args:
        prediction: List of predictions
        response_content_type: Desired response content type
        
    Returns:
        Formatted response string
    """
    if response_content_type == 'application/json':
        response = {
            'predictions': prediction,
            'count': len(prediction)
        }
        return json.dumps(response)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
