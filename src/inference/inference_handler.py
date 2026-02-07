"""
CLV Inference Handler

Handles model loading and prediction requests for the CLV API endpoint.
Compatible with AWS SageMaker Inference.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
import dill

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lifetimes.utils import summary_data_from_transaction_data

from src.preprocessing.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLVInferenceHandler:
    """
    Inference handler for CLV predictions.
    
    Loads trained models and processes prediction requests.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the handler.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.cox_model = None
        self.bgnbd_model = None
        self.gamma_gamma_model = None
        self.config = None
        self.processor = None
        
        if model_dir:
            self.load_models(model_dir)
    
    def load_models(self, model_dir: str) -> None:
        """
        Load all trained models from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        logger.info(f"Loading models from {model_dir}")
        
        # Load configuration
        config_file = os.path.join(model_dir, 'model_config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config: {self.config}")
        else:
            # Default config
            self.config = {
                'snapshot_date': '2025-06-04',
                'xdays': 365,
                'model_files': {
                    'cox': 'cox/cox_model.joblib',
                    'bgnbd': 'bgnbd/bgnbd_model.pkl',
                    'gamma_gamma': 'gamma_gamma/gamma_gamma_model.pkl'
                }
            }
        
        # Initialize processor
        self.processor = DataProcessor(
            snapshot_date=self.config['snapshot_date'],
            xdays=self.config['xdays']
        )
        
        # Load Cox model
        cox_path = os.path.join(model_dir, self.config['model_files']['cox'])
        if os.path.exists(cox_path):
            self.cox_model = joblib.load(cox_path)
            logger.info("Cox model loaded")
        else:
            logger.warning(f"Cox model not found at {cox_path}")
        
        # Load BG/NBD model
        bgnbd_path = os.path.join(model_dir, self.config['model_files']['bgnbd'])
        if os.path.exists(bgnbd_path):
            with open(bgnbd_path, 'rb') as f:
                self.bgnbd_model = dill.load(f)
            logger.info("BG/NBD model loaded")
        else:
            logger.warning(f"BG/NBD model not found at {bgnbd_path}")
        
        # Load Gamma-Gamma model
        gg_path = os.path.join(model_dir, self.config['model_files']['gamma_gamma'])
        if os.path.exists(gg_path):
            with open(gg_path, 'rb') as f:
                self.gamma_gamma_model = dill.load(f)
            logger.info("Gamma-Gamma model loaded")
        else:
            logger.warning(f"Gamma-Gamma model not found at {gg_path}")
        
        logger.info("All models loaded successfully")
    
    def predict(
        self,
        customer_data: List[Dict[str, Any]],
        snapshot_date: Optional[str] = None,
        xdays: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate CLV predictions for customers.
        
        Args:
            customer_data: List of customer dictionaries with transactions
            snapshot_date: Override snapshot date (optional)
            xdays: Override prediction horizon (optional)
            
        Returns:
            List of prediction dictionaries
        """
        if snapshot_date:
            self.processor.snapshot_date = pd.to_datetime(snapshot_date)
        if xdays:
            self.processor.xdays = xdays
            xdays_val = xdays
        else:
            xdays_val = self.config.get('xdays', 365)
        
        # Preprocess data
        data = self.processor.preprocess_for_inference(customer_data)
        
        predictions = []
        
        # Get RFM data for all customers
        df_rfm = data['rfm']
        
        # Process one-timers
        if len(data['one_timers']) > 0:
            one_timer_preds = self._predict_one_timers(
                data['one_timers'], 
                df_rfm, 
                xdays_val
            )
            predictions.extend(one_timer_preds)
        
        # Process repeaters
        if len(data['repeaters']) > 0:
            repeater_preds = self._predict_repeaters(
                data['repeaters'],
                data['repeaters_txn'],
                df_rfm,
                xdays_val
            )
            predictions.extend(repeater_preds)
        
        return predictions
    
    def _predict_one_timers(
        self,
        df_onetimers: pd.DataFrame,
        df_rfm: pd.DataFrame,
        xdays: int
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for one-time customers using Cox model.
        
        Args:
            df_onetimers: One-timer customer data
            df_rfm: RFM features
            xdays: Prediction horizon
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for _, row in df_onetimers.iterrows():
            customer_id = row['CustomerID']
            
            # Prepare features for Cox model
            df_pred = pd.DataFrame({
                'duration': [row['duration']],
                'event': [0],  # Not returned yet
                'avg_spend': [row['avg_spend']]
            })
            
            # Predict return probability
            if self.cox_model is not None:
                surv_prob = self.cox_model.predict_survival_function(
                    df_pred[['avg_spend']], 
                    times=[xdays]
                )
                p_return = 1 - surv_prob.iloc[0, 0]
            else:
                # Fallback: use simple heuristic
                p_return = 0.1
            
            # Calculate CLV
            clv = p_return * row['avg_spend']
            
            # Get RFM info
            rfm_row = df_rfm[df_rfm['CustomerID'] == customer_id]
            if len(rfm_row) > 0:
                rfm_score = int(rfm_row['RFM_Score'].values[0])
                segment = rfm_row['Segment'].values[0]
            else:
                rfm_score = 0
                segment = 'Unknown'
            
            # Assign CLV level
            clv_level = self._assign_clv_level(clv, is_onetimer=True)
            
            # Identify mismatch
            mismatch = self.processor.identify_mismatch(segment, clv_level)
            
            predictions.append({
                'CustomerID': customer_id,
                'CLV': round(float(clv), 2),
                'group': 'onetimer',
                'P_return': round(float(p_return), 4),
                'avg_spend': round(float(row['avg_spend']), 2),
                'RFM_Score': rfm_score,
                'Segment': segment,
                'CLV_Level': clv_level,
                'Mismatch': mismatch
            })
        
        return predictions
    
    def _predict_repeaters(
        self,
        df_repeaters: pd.DataFrame,
        df_repeaters_txn: pd.DataFrame,
        df_rfm: pd.DataFrame,
        xdays: int
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for repeat customers using BG/NBD and Gamma-Gamma.
        
        Args:
            df_repeaters: Repeater customer data
            df_repeaters_txn: Repeater transactions
            df_rfm: RFM features
            xdays: Prediction horizon
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        # Create summary data for lifetimes models
        summary_df = summary_data_from_transaction_data(
            df_repeaters_txn,
            customer_id_col='CustomerID',
            datetime_col='Arrival Date',
            monetary_value_col='Total Revenue USD',
            observation_period_end=self.processor.snapshot_date
        )
        
        for customer_id in df_repeaters['CustomerID'].unique():
            if customer_id not in summary_df.index:
                continue
            
            cust_summary = summary_df.loc[customer_id]
            
            # Predict expected purchases using BG/NBD
            if self.bgnbd_model is not None:
                expected_purchases = self.bgnbd_model.conditional_expected_number_of_purchases_up_to_time(
                    xdays,
                    cust_summary['frequency'],
                    cust_summary['recency'],
                    cust_summary['T']
                )
                
                p_alive = self.bgnbd_model.conditional_probability_alive(
                    cust_summary['frequency'],
                    cust_summary['recency'],
                    cust_summary['T']
                )
            else:
                expected_purchases = cust_summary['frequency'] * (xdays / cust_summary['T'])
                p_alive = 0.5
            
            # Predict expected value using Gamma-Gamma
            if self.gamma_gamma_model is not None and cust_summary['frequency'] > 0:
                expected_value = self.gamma_gamma_model.conditional_expected_average_profit(
                    cust_summary['frequency'],
                    cust_summary['monetary_value']
                )
            else:
                expected_value = cust_summary['monetary_value']
            
            # Calculate CLV
            clv = expected_purchases * expected_value
            
            # Get RFM info
            rfm_row = df_rfm[df_rfm['CustomerID'] == customer_id]
            if len(rfm_row) > 0:
                rfm_score = int(rfm_row['RFM_Score'].values[0])
                segment = rfm_row['Segment'].values[0]
            else:
                rfm_score = 0
                segment = 'Unknown'
            
            # Assign CLV level
            clv_level = self._assign_clv_level(clv, is_onetimer=False)
            
            # Identify mismatch
            mismatch = self.processor.identify_mismatch(segment, clv_level)
            
            predictions.append({
                'CustomerID': customer_id,
                'CLV': round(float(clv), 2),
                'group': 'repeater',
                'expected_purchases': round(float(expected_purchases), 2),
                'expected_value': round(float(expected_value), 2),
                'P_alive': round(float(p_alive), 4),
                'frequency': int(cust_summary['frequency']),
                'recency': round(float(cust_summary['recency']), 0),
                'RFM_Score': rfm_score,
                'Segment': segment,
                'CLV_Level': clv_level,
                'Mismatch': mismatch
            })
        
        return predictions
    
    def _assign_clv_level(self, clv: float, is_onetimer: bool) -> str:
        """
        Assign CLV level based on value.
        
        Uses different thresholds for one-timers vs repeaters.
        
        Args:
            clv: CLV value
            is_onetimer: Whether customer is a one-timer
            
        Returns:
            CLV level (Low, Mid, High, Top)
        """
        if is_onetimer:
            # Thresholds for one-timers (lower expected CLV)
            if clv < 50:
                return 'Low'
            elif clv < 150:
                return 'Mid'
            elif clv < 300:
                return 'High'
            else:
                return 'Top'
        else:
            # Thresholds for repeaters (higher expected CLV)
            if clv < 200:
                return 'Low'
            elif clv < 500:
                return 'Mid'
            elif clv < 1000:
                return 'High'
            else:
                return 'Top'


# ============================================================
# SageMaker Inference Functions
# ============================================================

# Global handler instance (loaded once when container starts)
_handler = None


def model_fn(model_dir: str) -> CLVInferenceHandler:
    """
    Load models for SageMaker inference.
    
    Called once when the inference container starts.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        CLVInferenceHandler instance
    """
    global _handler
    logger.info(f"Loading models from {model_dir}")
    _handler = CLVInferenceHandler(model_dir=model_dir)
    return _handler


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


def predict_fn(input_data: Dict[str, Any], model: CLVInferenceHandler) -> List[Dict[str, Any]]:
    """
    Generate predictions.
    
    Args:
        input_data: Parsed request data
        model: CLVInferenceHandler instance
        
    Returns:
        List of predictions
    """
    customer_data = input_data.get('customers', [])
    snapshot_date = input_data.get('snapshot_date')
    xdays = input_data.get('xdays')
    
    predictions = model.predict(
        customer_data=customer_data,
        snapshot_date=snapshot_date,
        xdays=xdays
    )
    
    return predictions


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


# ============================================================
# Local Testing
# ============================================================

def test_local():
    """Test the inference handler locally."""
    
    # Sample customer data
    test_customers = [
        {
            "CustomerID": "C001",
            "transactions": [
                {"Arrival Date": "2024-01-15", "Total Revenue USD": 500, "Channel": "WEB"},
                {"Arrival Date": "2024-06-20", "Total Revenue USD": 750, "Channel": "WEB"}
            ]
        },
        {
            "CustomerID": "C002",
            "transactions": [
                {"Arrival Date": "2024-03-10", "Total Revenue USD": 300, "Channel": "GDS"}
            ]
        },
        {
            "CustomerID": "C003",
            "transactions": [
                {"Arrival Date": "2024-02-01", "Total Revenue USD": 1000, "Channel": "CHOPRA"},
                {"Arrival Date": "2024-05-15", "Total Revenue USD": 800, "Channel": "CHOPRA"},
                {"Arrival Date": "2024-08-20", "Total Revenue USD": 1200, "Channel": "CHOPRA"}
            ]
        }
    ]
    
    # Create handler without models (will use fallback predictions)
    handler = CLVInferenceHandler()
    handler.config = {'snapshot_date': '2025-06-04', 'xdays': 365}
    handler.processor = DataProcessor(snapshot_date='2025-06-04', xdays=365)
    
    # Generate predictions
    predictions = handler.predict(test_customers)
    
    print("\n" + "="*60)
    print("LOCAL TEST RESULTS")
    print("="*60)
    
    for pred in predictions:
        print(f"\nCustomer: {pred['CustomerID']}")
        print(f"  Group: {pred['group']}")
        print(f"  CLV: ${pred['CLV']:.2f}")
        print(f"  CLV Level: {pred['CLV_Level']}")
        print(f"  RFM Score: {pred['RFM_Score']}")
        print(f"  Segment: {pred['Segment']}")
        print(f"  Mismatch: {pred['Mismatch']}")
    
    return predictions


if __name__ == '__main__':
    test_local()
