"""
Data Preprocessing Module for CLV Machine Learning

This module contains functions to load, clean, and transform transaction data
for Customer Lifetime Value prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any


class DataProcessor:
    """
    Data processor for CLV model preprocessing.
    
    Handles loading, cleaning, feature engineering, and customer segmentation
    for transaction-level data.
    
    Attributes:
        snapshot_date: The date when the data snapshot was taken
        xdays: Prediction horizon in days (default: 365)
    """
    
    # Channel grouping mapping
    CHANNEL_GROUPS = {
        'WEB': 'WEB',
        'GDS': 'OTA/GDS',
        'CRS': 'OTA/GDS',
        'INTERNET': 'OTA/GDS',
        'GOOG': 'OTA/GDS',
        'EMAIL': 'OTA/GDS',
        'IDS': 'OTA/GDS',
        'RL': 'Direct',
        'PMS': 'Direct',
        'DCI': 'Direct',
        'WI': 'Direct',
        'HOUSE': 'Direct',
        'SAL': 'Direct',
        'SYDC': 'Corporate',
        'GCI': 'Corporate',
        'MARKETING': 'Corporate',
        'CHOPRA': 'Specialty'
    }
    
    # RFM segment thresholds
    RFM_SEGMENTS = {
        (13, 15): 'Champions',
        (10, 12): 'Loyal Customers',
        (7, 9): 'Potential Loyalists',
        (4, 6): 'At Risk',
        (3, 3): 'Lost'
    }
    
    def __init__(self, snapshot_date: str, xdays: int = 365):
        """
        Initialize the DataProcessor.
        
        Args:
            snapshot_date: Date string in format 'YYYY-MM-DD' when data was taken
            xdays: Prediction horizon in days (default: 365)
        """
        self.snapshot_date = pd.to_datetime(snapshot_date)
        self.xdays = xdays
    
    def load_and_clean(
        self, 
        filepath: str,
        date_column: str = 'Arrival Date',
        revenue_column: str = 'Total Revenue USD',
        customer_id_column: str = 'CustomerID',
        channel_column: str = 'Channel'
    ) -> pd.DataFrame:
        """
        Load transaction data from CSV and perform initial cleaning.
        
        Args:
            filepath: Path to the CSV file
            date_column: Name of the date column
            revenue_column: Name of the revenue column
            customer_id_column: Name of the customer ID column
            channel_column: Name of the channel column
            
        Returns:
            Cleaned DataFrame with transactions
        """
        # Load data
        df = pd.read_csv(filepath, parse_dates=[date_column], low_memory=False)
        
        # Select relevant columns
        columns_to_keep = [customer_id_column, date_column, revenue_column, channel_column]
        df_txn = df[columns_to_keep].copy()
        
        # Ensure date is datetime
        df_txn[date_column] = pd.to_datetime(df_txn[date_column])
        
        # Sort by customer and date
        df_txn = df_txn.sort_values([customer_id_column, date_column])
        
        # Filter: only past transactions with positive revenue
        df_txn = df_txn[
            (df_txn[date_column] < self.snapshot_date) &
            (df_txn[revenue_column] > 0)
        ].copy()
        
        return df_txn
    
    def categorize_channel(self, channel: str) -> str:
        """
        Categorize a channel into a channel group.
        
        Args:
            channel: Original channel name
            
        Returns:
            Channel group name
        """
        if pd.isna(channel):
            return 'Unknown'
        return self.CHANNEL_GROUPS.get(channel, 'Unknown')
    
    def add_channel_groups(
        self, 
        df: pd.DataFrame, 
        channel_column: str = 'Channel'
    ) -> pd.DataFrame:
        """
        Add channel group column and one-hot encoded channel features.
        
        Args:
            df: DataFrame with transactions
            channel_column: Name of the channel column
            
        Returns:
            DataFrame with Channel_Group and one-hot encoded columns
        """
        df = df.copy()
        
        # Create channel group
        df['Channel_Group'] = df[channel_column].fillna('Unknown').apply(self.categorize_channel)
        
        # One-hot encode
        channel_dummies = pd.get_dummies(df['Channel_Group'], prefix='Channel', drop_first=False)
        df = pd.concat([df, channel_dummies], axis=1)
        
        return df
    
    def identify_customer_types(
        self, 
        df: pd.DataFrame, 
        customer_id_column: str = 'CustomerID'
    ) -> pd.DataFrame:
        """
        Identify one-time vs repeat customers.
        
        Args:
            df: DataFrame with transactions
            customer_id_column: Name of the customer ID column
            
        Returns:
            DataFrame with 'repeat' column ('Y' or 'N')
        """
        df = df.copy()
        repeat_counts = df[customer_id_column].value_counts()
        df['repeat'] = df[customer_id_column].map(
            lambda x: 'Y' if repeat_counts[x] > 1 else 'N'
        )
        return df
    
    def split_customers(
        self, 
        df: pd.DataFrame, 
        customer_id_column: str = 'CustomerID'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split transactions into one-timers and repeaters.
        
        Args:
            df: DataFrame with transactions (must have 'repeat' column)
            customer_id_column: Name of the customer ID column
            
        Returns:
            Tuple of (one_timers_df, repeaters_df)
        """
        if 'repeat' not in df.columns:
            df = self.identify_customer_types(df, customer_id_column)
        
        one_timers = df[df['repeat'] == 'N'].copy()
        repeaters = df[df['repeat'] == 'Y'].copy()
        
        return one_timers, repeaters
    
    def create_rfm_features(
        self, 
        df: pd.DataFrame,
        customer_id_column: str = 'CustomerID',
        date_column: str = 'Arrival Date',
        revenue_column: str = 'Total Revenue USD'
    ) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features per customer.
        
        Args:
            df: DataFrame with transactions
            customer_id_column: Name of the customer ID column
            date_column: Name of the date column
            revenue_column: Name of the revenue column
            
        Returns:
            DataFrame with one row per customer and RFM features
        """
        df_rfm = df.groupby(customer_id_column).agg({
            date_column: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            customer_id_column: 'count',  # Frequency (will rename)
            revenue_column: 'sum'  # Monetary
        })
        
        # Fix column names after aggregation
        df_rfm.columns = ['Recency', 'Frequency', 'Monetary']
        df_rfm = df_rfm.reset_index()
        
        return df_rfm
    
    def calculate_rfm_scores(self, df_rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM scores (1-5) for each dimension.
        
        Args:
            df_rfm: DataFrame with Recency, Frequency, Monetary columns
            
        Returns:
            DataFrame with R_Score, F_Score, M_Score, RFM_Segment, RFM_Score columns
        """
        df = df_rfm.copy()
        
        # Recency: lower is better → reverse the score
        df['R_Score'] = pd.qcut(
            df['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop'
        ).astype(int)
        
        # Frequency: higher is better
        df['F_Score'] = pd.qcut(
            df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        ).astype(int)
        
        # Monetary: higher is better
        df['M_Score'] = pd.qcut(
            df['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        ).astype(int)
        
        # Combine to get RFM segment code
        df['RFM_Segment'] = (
            df['R_Score'].astype(str) + 
            df['F_Score'].astype(str) + 
            df['M_Score'].astype(str)
        )
        
        # Total RFM score
        df['RFM_Score'] = df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
        
        # Assign segment name
        df['Segment'] = df['RFM_Score'].apply(self._assign_rfm_segment)
        
        return df
    
    def _assign_rfm_segment(self, score: int) -> str:
        """
        Assign RFM segment name based on total score.
        
        Args:
            score: Total RFM score (3-15)
            
        Returns:
            Segment name
        """
        if score >= 13:
            return 'Champions'
        elif score >= 10:
            return 'Loyal Customers'
        elif score >= 7:
            return 'Potential Loyalists'
        elif score >= 4:
            return 'At Risk'
        else:
            return 'Lost'
    
    def aggregate_customer_data(
        self,
        df: pd.DataFrame,
        customer_id_column: str = 'CustomerID',
        date_column: str = 'Arrival Date',
        revenue_column: str = 'Total Revenue USD'
    ) -> pd.DataFrame:
        """
        Aggregate transaction data to one row per customer for survival analysis.
        
        Args:
            df: DataFrame with transactions (with channel dummies)
            customer_id_column: Name of the customer ID column
            date_column: Name of the date column
            revenue_column: Name of the revenue column
            
        Returns:
            DataFrame with one row per customer including:
            - first_date, last_date, visit_count
            - avg_spend, duration, event
            - channel dummy columns
        """
        # Find channel dummy columns
        channel_cols = [col for col in df.columns if col.startswith('Channel_')]
        
        # Group by customer
        df_cust = df.groupby(customer_id_column).agg({
            date_column: ['min', 'max', 'count'],
            revenue_column: 'mean'
        }).reset_index()
        
        # Flatten column names
        df_cust.columns = [customer_id_column, 'first_date', 'last_date', 'visit_count', 'avg_spend']
        
        # Duration = time between first visit and snapshot
        df_cust['duration'] = (self.snapshot_date - df_cust['first_date']).dt.days
        
        # Event = 1 if they had more than one visit (returned)
        df_cust['event'] = (df_cust['visit_count'] > 1).astype(int)
        
        # Add channel dummies (first occurrence per customer)
        if channel_cols:
            df_channels = df[[customer_id_column] + channel_cols].drop_duplicates(customer_id_column)
            df_cust = df_cust.merge(df_channels, on=customer_id_column, how='left')
        
        return df_cust
    
    def assign_clv_level(self, clv_values: pd.Series) -> pd.Series:
        """
        Assign CLV level based on quartiles.
        
        Args:
            clv_values: Series of CLV values
            
        Returns:
            Series with CLV levels (Low, Mid, High, Top)
        """
        return pd.qcut(
            clv_values, 
            q=4, 
            labels=['Low', 'Mid', 'High', 'Top'],
            duplicates='drop'
        )
    
    def identify_mismatch(
        self, 
        segment: str, 
        clv_level: str
    ) -> str:
        """
        Identify mismatch between RFM segment and CLV prediction.
        
        Args:
            segment: RFM segment name
            clv_level: CLV level (Low, Mid, High, Top)
            
        Returns:
            Mismatch category
        """
        high_rfm_segments = ['Champions', 'Loyal Customers']
        low_rfm_segments = ['At Risk', 'Lost']
        
        if segment in high_rfm_segments and clv_level == 'Low':
            return 'Overrated (High RFM, Low CLV)'
        elif segment in low_rfm_segments and clv_level in ['High', 'Top']:
            return 'Underrated (Low RFM, High CLV)'
        else:
            return 'Aligned'
    
    def get_latest_channel_per_customer(
        self,
        df: pd.DataFrame,
        customer_id_column: str = 'CustomerID',
        date_column: str = 'Arrival Date',
        channel_column: str = 'Channel'
    ) -> pd.DataFrame:
        """
        Get the most recent channel for each customer.
        
        Args:
            df: DataFrame with transactions
            customer_id_column: Name of the customer ID column
            date_column: Name of the date column
            channel_column: Name of the channel column
            
        Returns:
            DataFrame with CustomerID, Channel, Channel_Group
        """
        latest = (
            df.sort_values([customer_id_column, date_column])
            .groupby(customer_id_column)
            .tail(1)[[customer_id_column, channel_column, 'Channel_Group']]
            .reset_index(drop=True)
        )
        return latest
    
    def preprocess_for_training(
        self,
        filepath: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete preprocessing pipeline for model training.
        
        Args:
            filepath: Path to the transaction CSV file
            
        Returns:
            Dictionary with:
            - 'transactions': Cleaned transaction data
            - 'customer_data': Aggregated customer-level data
            - 'rfm': RFM features and segments
            - 'one_timers': One-timer customer data
            - 'repeaters': Repeater customer data
        """
        # Load and clean
        df_txn = self.load_and_clean(filepath)
        
        # Add channel groups
        df_txn = self.add_channel_groups(df_txn)
        
        # Identify customer types
        df_txn = self.identify_customer_types(df_txn)
        
        # Create customer-level aggregates
        df_cust = self.aggregate_customer_data(df_txn)
        
        # Create RFM features
        df_rfm = self.create_rfm_features(df_txn)
        df_rfm = self.calculate_rfm_scores(df_rfm)
        
        # Split by customer type
        df_one_timers, df_repeaters = self.split_customers(df_txn)
        
        # Get latest channel
        latest_channel = self.get_latest_channel_per_customer(df_txn)
        
        return {
            'transactions': df_txn,
            'customer_data': df_cust,
            'rfm': df_rfm,
            'one_timers': df_one_timers,
            'repeaters': df_repeaters,
            'latest_channel': latest_channel
        }
    
    def preprocess_for_inference(
        self,
        customer_data: List[Dict[str, Any]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Preprocess customer data for inference (API request).
        
        Args:
            customer_data: List of customer dictionaries with transactions
                Example:
                [
                    {
                        "CustomerID": "C001",
                        "transactions": [
                            {"Arrival Date": "2024-01-15", "Total Revenue USD": 500, "Channel": "WEB"},
                            {"Arrival Date": "2024-06-20", "Total Revenue USD": 750, "Channel": "WEB"}
                        ]
                    }
                ]
                
        Returns:
            Dictionary with processed DataFrames ready for prediction
        """
        # Convert to DataFrame
        all_transactions = []
        for customer in customer_data:
            customer_id = customer['CustomerID']
            for txn in customer['transactions']:
                txn_record = {
                    'CustomerID': customer_id,
                    'Arrival Date': pd.to_datetime(txn['Arrival Date']),
                    'Total Revenue USD': txn['Total Revenue USD'],
                    'Channel': txn.get('Channel', 'Unknown')
                }
                all_transactions.append(txn_record)
        
        df_txn = pd.DataFrame(all_transactions)
        
        # Add channel groups
        df_txn = self.add_channel_groups(df_txn)
        
        # Identify customer types
        df_txn = self.identify_customer_types(df_txn)
        
        # Create customer-level aggregates
        df_cust = self.aggregate_customer_data(df_txn)
        
        # Create RFM features
        df_rfm = self.create_rfm_features(df_txn)
        df_rfm = self.calculate_rfm_scores(df_rfm)
        
        # Split by customer type
        df_one_timers_txn, df_repeaters_txn = self.split_customers(df_txn)
        
        # Get one-timer and repeater customer data
        one_timer_ids = df_one_timers_txn['CustomerID'].unique()
        repeater_ids = df_repeaters_txn['CustomerID'].unique()
        
        df_one_timers = df_cust[df_cust['CustomerID'].isin(one_timer_ids)]
        df_repeaters = df_cust[df_cust['CustomerID'].isin(repeater_ids)]
        
        return {
            'transactions': df_txn,
            'customer_data': df_cust,
            'rfm': df_rfm,
            'one_timers': df_one_timers,
            'repeaters': df_repeaters,
            'repeaters_txn': df_repeaters_txn
        }
