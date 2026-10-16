"""
SpaceSync — Reproducible ML Demand Forecasting & Benchmark Script
DATATHON 2026 | VIT-AP University | Education Track

Task: Forecast campus room request volume per hourly slot (requests_per_slot).
Split: Chronological 80% train (Aug 1 - Nov 4) / 20% test (Nov 5 - Nov 28). Zero future leakage.
"""

from engine import load_and_validate_datasets, evaluate_demand_forecasting
import pandas as pd

def main():
    print("=" * 70)
    print(" SpaceSync — Machine Learning Demand Forecasting Evaluation")
    print("=" * 70)
    
    bundle = load_and_validate_datasets()
    res = evaluate_demand_forecasting(bundle)
    
    print(f"\n[Target]: {res['target']}")
    print(f"[Features]: {', '.join(res['features'])}")
    print(f"[Split]: {res['split_method']}")
    print(f"[Train Period]: {res['train_period']}")
    print(f"[Test Period]:  {res['test_period']}\n")
    
    print("-" * 70)
    print(" MODEL PERFORMANCE COMPARISON (Primary Metric: MAE)")
    print("-" * 70)
    df = pd.DataFrame(res['metrics_table'])
    print(df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"[Conclusion]:\n{res['takeaway']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
