"""
End-to-End Pipeline Test Script

Tests the complete CLV pipeline:
1. Load data
2. Preprocess
3. Train models
4. Save models
5. Load models
6. Make predictions

Usage:
    python scripts/test_pipeline.py --data-file path/to/data.csv --snapshot-date 2025-06-04
"""

import os
import sys
import json
import argparse
import tempfile
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.data_processor import DataProcessor
from src.training.train_all import CLVModelPipeline
from src.inference.inference_handler import CLVInferenceHandler


def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_with_sample_data():
    """Test with sample synthetic data."""
    print_section("TESTING WITH SAMPLE DATA (No CSV needed)")
    
    # Create sample customer data for inference
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
        },
        {
            "CustomerID": "TEST003",
            "transactions": [
                {"Arrival Date": "2024-02-01", "Total Revenue USD": 1000, "Channel": "CHOPRA"},
                {"Arrival Date": "2024-05-15", "Total Revenue USD": 800, "Channel": "CHOPRA"},
                {"Arrival Date": "2024-08-20", "Total Revenue USD": 1200, "Channel": "CHOPRA"}
            ]
        },
        {
            "CustomerID": "TEST004",
            "transactions": [
                {"Arrival Date": "2024-04-01", "Total Revenue USD": 2500, "Channel": "WEB"}
            ]
        },
        {
            "CustomerID": "TEST005",
            "transactions": [
                {"Arrival Date": "2023-06-15", "Total Revenue USD": 400, "Channel": "RL"},
                {"Arrival Date": "2024-01-10", "Total Revenue USD": 450, "Channel": "RL"},
                {"Arrival Date": "2024-07-20", "Total Revenue USD": 500, "Channel": "RL"},
                {"Arrival Date": "2025-01-05", "Total Revenue USD": 550, "Channel": "RL"}
            ]
        }
    ]
    
    # Test preprocessing
    print_section("1. PREPROCESSING TEST")
    processor = DataProcessor(snapshot_date='2025-06-04', xdays=365)
    data = processor.preprocess_for_inference(sample_customers)
    
    print(f"[OK] Processed {len(sample_customers)} customers")
    print(f"  - Total transactions: {len(data['transactions'])}")
    print(f"  - One-timers: {len(data['one_timers'])}")
    print(f"  - Repeaters: {len(data['repeaters'])}")
    
    # Test RFM
    print("\n  RFM Analysis:")
    for _, row in data['rfm'].iterrows():
        print(f"    {row['CustomerID']}: R={row['Recency']:.0f}, F={row['Frequency']}, M=${row['Monetary']:.0f} -> {row['Segment']}")
    
    # Test inference (without trained models - uses fallback)
    print_section("2. INFERENCE TEST (Fallback mode - no trained models)")
    handler = CLVInferenceHandler()
    handler.config = {'snapshot_date': '2025-06-04', 'xdays': 365}
    handler.processor = processor
    
    predictions = handler.predict(sample_customers)
    
    print(f"[OK] Generated {len(predictions)} predictions\n")
    
    # Print predictions table
    print(f"{'CustomerID':<12} {'Group':<10} {'CLV':>10} {'Level':<6} {'Segment':<20} {'Mismatch':<30}")
    print("-" * 100)
    
    for pred in predictions:
        print(f"{pred['CustomerID']:<12} {pred['group']:<10} ${pred['CLV']:>8.2f} {pred['CLV_Level']:<6} {pred['Segment']:<20} {pred['Mismatch']:<30}")
    
    # Test JSON serialization (API response format)
    print_section("3. API RESPONSE FORMAT TEST")
    response = {
        'predictions': predictions,
        'count': len(predictions),
        'snapshot_date': '2025-06-04',
        'xdays': 365
    }
    
    json_response = json.dumps(response, indent=2)
    print(json_response[:500] + "..." if len(json_response) > 500 else json_response)
    
    print("\n[OK] JSON serialization successful")
    
    return True


def test_with_csv(data_file: str, snapshot_date: str, xdays: int = 365):
    """Test with real CSV data including training."""
    print_section(f"TESTING WITH CSV: {data_file}")
    
    # Create temp directory for models
    model_dir = tempfile.mkdtemp(prefix='clv_models_')
    print(f"Model directory: {model_dir}")
    
    try:
        # 1. Train models
        print_section("1. TRAINING MODELS")
        pipeline = CLVModelPipeline(
            snapshot_date=snapshot_date,
            xdays=xdays,
            penalizer=0.0
        )
        
        pipeline.load_data(data_file)
        pipeline.train_all()
        pipeline.save_models(model_dir)
        
        print("[OK] All models trained and saved")
        
        # 2. Load models for inference
        print_section("2. LOADING MODELS FOR INFERENCE")
        handler = CLVInferenceHandler(model_dir=model_dir)
        print("[OK] Models loaded")
        
        # 3. Test with sample customers
        print_section("3. INFERENCE TEST")
        sample_customers = [
            {
                "CustomerID": "SAMPLE001",
                "transactions": [
                    {"Arrival Date": "2024-06-15", "Total Revenue USD": 800, "Channel": "WEB"}
                ]
            },
            {
                "CustomerID": "SAMPLE002",
                "transactions": [
                    {"Arrival Date": "2024-01-10", "Total Revenue USD": 500, "Channel": "WEB"},
                    {"Arrival Date": "2024-08-20", "Total Revenue USD": 600, "Channel": "WEB"}
                ]
            }
        ]
        
        predictions = handler.predict(sample_customers, snapshot_date=snapshot_date, xdays=xdays)
        
        print(f"[OK] Generated {len(predictions)} predictions\n")
        for pred in predictions:
            print(f"  {pred['CustomerID']}: CLV=${pred['CLV']:.2f}, {pred['group']}, {pred['Segment']}")
        
        print_section("TEST COMPLETE - SUCCESS!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"\nCleaned up temp directory: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test CLV Pipeline')
    parser.add_argument('--data-file', type=str, help='Path to transaction CSV file')
    parser.add_argument('--snapshot-date', type=str, default='2025-06-04',
                        help='Snapshot date (YYYY-MM-DD)')
    parser.add_argument('--xdays', type=int, default=365,
                        help='Prediction horizon in days')
    parser.add_argument('--sample-only', action='store_true',
                        help='Only test with sample data (no CSV needed)')
    
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("   CLV PIPELINE END-TO-END TEST")
    print("#"*60)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Snapshot Date: {args.snapshot_date}")
    print(f"Prediction Horizon: {args.xdays} days")
    
    # Always run sample test
    success = test_with_sample_data()
    
    # Optionally run CSV test
    if args.data_file and not args.sample_only:
        if os.path.exists(args.data_file):
            success = test_with_csv(args.data_file, args.snapshot_date, args.xdays) and success
        else:
            print(f"\n[WARNING] Data file not found: {args.data_file}")
            print("   Run with --sample-only to skip CSV testing")
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("[SUCCESS] ALL TESTS PASSED")
    else:
        print("[FAILED] SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
