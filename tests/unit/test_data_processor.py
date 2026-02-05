"""
Unit tests for the DataProcessor class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.data_processor import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor(snapshot_date='2025-06-04', xdays=365)
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data for testing."""
        data = {
            'CustomerID': ['C001', 'C001', 'C002', 'C003', 'C003', 'C003'],
            'Arrival Date': [
                '2024-01-15', '2024-06-20', '2024-03-10', 
                '2024-02-01', '2024-05-15', '2024-08-20'
            ],
            'Total Revenue USD': [500, 750, 300, 1000, 800, 1200],
            'Channel': ['WEB', 'WEB', 'GDS', 'CHOPRA', 'CHOPRA', 'CHOPRA']
        }
        df = pd.DataFrame(data)
        df['Arrival Date'] = pd.to_datetime(df['Arrival Date'])
        return df
    
    def test_init(self, processor):
        """Test DataProcessor initialization."""
        assert processor.snapshot_date == pd.to_datetime('2025-06-04')
        assert processor.xdays == 365
    
    def test_categorize_channel_web(self, processor):
        """Test channel categorization for WEB."""
        assert processor.categorize_channel('WEB') == 'WEB'
    
    def test_categorize_channel_ota(self, processor):
        """Test channel categorization for OTA/GDS channels."""
        assert processor.categorize_channel('GDS') == 'OTA/GDS'
        assert processor.categorize_channel('CRS') == 'OTA/GDS'
        assert processor.categorize_channel('EMAIL') == 'OTA/GDS'
    
    def test_categorize_channel_direct(self, processor):
        """Test channel categorization for Direct channels."""
        assert processor.categorize_channel('RL') == 'Direct'
        assert processor.categorize_channel('PMS') == 'Direct'
    
    def test_categorize_channel_corporate(self, processor):
        """Test channel categorization for Corporate channels."""
        assert processor.categorize_channel('SYDC') == 'Corporate'
        assert processor.categorize_channel('GCI') == 'Corporate'
    
    def test_categorize_channel_specialty(self, processor):
        """Test channel categorization for Specialty channels."""
        assert processor.categorize_channel('CHOPRA') == 'Specialty'
    
    def test_categorize_channel_unknown(self, processor):
        """Test channel categorization for unknown channels."""
        assert processor.categorize_channel('RANDOM') == 'Unknown'
        assert processor.categorize_channel(None) == 'Unknown'
    
    def test_add_channel_groups(self, processor, sample_transactions):
        """Test adding channel groups to transactions."""
        result = processor.add_channel_groups(sample_transactions)
        
        assert 'Channel_Group' in result.columns
        assert 'Channel_WEB' in result.columns
        assert 'Channel_OTA/GDS' in result.columns
        assert 'Channel_Specialty' in result.columns
        
        # Check correct groupings
        web_rows = result[result['Channel'] == 'WEB']
        assert all(web_rows['Channel_Group'] == 'WEB')
        
        gds_rows = result[result['Channel'] == 'GDS']
        assert all(gds_rows['Channel_Group'] == 'OTA/GDS')
    
    def test_identify_customer_types(self, processor, sample_transactions):
        """Test customer type identification (one-timer vs repeater)."""
        result = processor.identify_customer_types(sample_transactions)
        
        assert 'repeat' in result.columns
        
        # C001 has 2 transactions -> repeater
        c001 = result[result['CustomerID'] == 'C001']
        assert all(c001['repeat'] == 'Y')
        
        # C002 has 1 transaction -> one-timer
        c002 = result[result['CustomerID'] == 'C002']
        assert all(c002['repeat'] == 'N')
        
        # C003 has 3 transactions -> repeater
        c003 = result[result['CustomerID'] == 'C003']
        assert all(c003['repeat'] == 'Y')
    
    def test_split_customers(self, processor, sample_transactions):
        """Test splitting customers into one-timers and repeaters."""
        df = processor.identify_customer_types(sample_transactions)
        one_timers, repeaters = processor.split_customers(df)
        
        # C002 should be in one-timers
        assert 'C002' in one_timers['CustomerID'].values
        assert len(one_timers) == 1
        
        # C001 and C003 should be in repeaters
        assert 'C001' in repeaters['CustomerID'].values
        assert 'C003' in repeaters['CustomerID'].values
        assert len(repeaters) == 5  # Total transactions for repeaters
    
    def test_create_rfm_features(self, processor, sample_transactions):
        """Test RFM feature creation."""
        result = processor.create_rfm_features(sample_transactions)
        
        assert 'Recency' in result.columns
        assert 'Frequency' in result.columns
        assert 'Monetary' in result.columns
        
        # Check customer count
        assert len(result) == 3  # 3 unique customers
        
        # Check C001 (2 transactions, total 1250)
        c001 = result[result['CustomerID'] == 'C001'].iloc[0]
        assert c001['Frequency'] == 2
        assert c001['Monetary'] == 1250
        
        # Check C003 (3 transactions, total 3000)
        c003 = result[result['CustomerID'] == 'C003'].iloc[0]
        assert c003['Frequency'] == 3
        assert c003['Monetary'] == 3000
    
    def test_assign_rfm_segment(self, processor):
        """Test RFM segment assignment based on score."""
        assert processor._assign_rfm_segment(15) == 'Champions'
        assert processor._assign_rfm_segment(13) == 'Champions'
        assert processor._assign_rfm_segment(12) == 'Loyal Customers'
        assert processor._assign_rfm_segment(10) == 'Loyal Customers'
        assert processor._assign_rfm_segment(9) == 'Potential Loyalists'
        assert processor._assign_rfm_segment(7) == 'Potential Loyalists'
        assert processor._assign_rfm_segment(6) == 'At Risk'
        assert processor._assign_rfm_segment(4) == 'At Risk'
        assert processor._assign_rfm_segment(3) == 'Lost'
    
    def test_aggregate_customer_data(self, processor, sample_transactions):
        """Test customer data aggregation."""
        df = processor.add_channel_groups(sample_transactions)
        result = processor.aggregate_customer_data(df)
        
        assert 'first_date' in result.columns
        assert 'last_date' in result.columns
        assert 'visit_count' in result.columns
        assert 'avg_spend' in result.columns
        assert 'duration' in result.columns
        assert 'event' in result.columns
        
        # Check customer count
        assert len(result) == 3
        
        # Check C001
        c001 = result[result['CustomerID'] == 'C001'].iloc[0]
        assert c001['visit_count'] == 2
        assert c001['avg_spend'] == 625  # (500 + 750) / 2
        assert c001['event'] == 1  # Returned
        
        # Check C002
        c002 = result[result['CustomerID'] == 'C002'].iloc[0]
        assert c002['visit_count'] == 1
        assert c002['event'] == 0  # Did not return
    
    def test_identify_mismatch_aligned(self, processor):
        """Test mismatch identification for aligned cases."""
        assert processor.identify_mismatch('Champions', 'Top') == 'Aligned'
        assert processor.identify_mismatch('At Risk', 'Low') == 'Aligned'
        assert processor.identify_mismatch('Potential Loyalists', 'Mid') == 'Aligned'
    
    def test_identify_mismatch_overrated(self, processor):
        """Test mismatch identification for overrated cases."""
        result = processor.identify_mismatch('Champions', 'Low')
        assert result == 'Overrated (High RFM, Low CLV)'
        
        result = processor.identify_mismatch('Loyal Customers', 'Low')
        assert result == 'Overrated (High RFM, Low CLV)'
    
    def test_identify_mismatch_underrated(self, processor):
        """Test mismatch identification for underrated cases."""
        result = processor.identify_mismatch('At Risk', 'Top')
        assert result == 'Underrated (Low RFM, High CLV)'
        
        result = processor.identify_mismatch('Lost', 'High')
        assert result == 'Underrated (Low RFM, High CLV)'
    
    def test_get_latest_channel_per_customer(self, processor, sample_transactions):
        """Test getting latest channel per customer."""
        df = processor.add_channel_groups(sample_transactions)
        result = processor.get_latest_channel_per_customer(df)
        
        assert len(result) == 3
        
        # C001's latest transaction is WEB on 2024-06-20
        c001 = result[result['CustomerID'] == 'C001'].iloc[0]
        assert c001['Channel'] == 'WEB'
        
        # C003's latest transaction is CHOPRA on 2024-08-20
        c003 = result[result['CustomerID'] == 'C003'].iloc[0]
        assert c003['Channel'] == 'CHOPRA'
        assert c003['Channel_Group'] == 'Specialty'
    
    def test_preprocess_for_inference(self, processor):
        """Test preprocessing for inference API."""
        customer_data = [
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
            }
        ]
        
        result = processor.preprocess_for_inference(customer_data)
        
        assert 'transactions' in result
        assert 'customer_data' in result
        assert 'rfm' in result
        assert 'one_timers' in result
        assert 'repeaters' in result
        
        # Check transactions
        assert len(result['transactions']) == 3
        
        # Check customer data
        assert len(result['customer_data']) == 2
        
        # C001 is repeater, C002 is one-timer
        assert len(result['one_timers']) == 1
        assert len(result['repeaters']) == 1


class TestDataProcessorEdgeCases:
    """Test edge cases for DataProcessor."""
    
    @pytest.fixture
    def processor(self):
        return DataProcessor(snapshot_date='2025-06-04', xdays=365)
    
    def test_empty_transactions(self, processor):
        """Test handling of empty transaction list."""
        result = processor.preprocess_for_inference([])
        
        assert len(result['transactions']) == 0
        assert len(result['customer_data']) == 0
    
    def test_missing_channel(self, processor):
        """Test handling of missing channel."""
        customer_data = [
            {
                "CustomerID": "C001",
                "transactions": [
                    {"Arrival Date": "2024-01-15", "Total Revenue USD": 500}
                ]
            }
        ]
        
        result = processor.preprocess_for_inference(customer_data)
        
        # Should default to Unknown
        assert result['transactions'].iloc[0]['Channel'] == 'Unknown'
        assert result['transactions'].iloc[0]['Channel_Group'] == 'Unknown'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
