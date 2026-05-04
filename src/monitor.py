"""
monitor.py — Data Drift Monitoring with Evidently AI
=====================================================
Generates data drift reports comparing training data distribution
vs a simulated "new batch" of production transactions.
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')


def generate_drift_report():
    """
    Generate an Evidently AI drift report.
    Compares training data (reference) with a simulated production batch (current).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Generating data drift report...")

    # Load training data as reference
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))

    # Use test data as "current" production batch
    reference_data = X_train.sample(n=min(5000, len(X_train)), random_state=42)
    current_data = X_test.sample(n=min(2000, len(X_test)), random_state=42)

    try:
        # Try newer Evidently API (v0.5+)
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataQualityPreset
    except ImportError:
        try:
            # Try older Evidently API (v0.4.x)
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        except ImportError:
            print("   Evidently import failed. Generating basic drift report manually...")
            _generate_manual_drift_report(reference_data, current_data)
            return

    # Generate Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    drift_path = os.path.join(REPORTS_DIR, 'drift_report.html')
    drift_report.save_html(drift_path)
    print(f"   Drift report saved to {drift_path}")

    # Generate Data Quality Report
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=reference_data, current_data=current_data)
    quality_path = os.path.join(REPORTS_DIR, 'data_quality_report.html')
    quality_report.save_html(quality_path)
    print(f"   Data quality report saved to {quality_path}")

    print("\nAll monitoring reports generated!")


def _generate_manual_drift_report(reference, current):
    """Fallback: generate a simple HTML drift comparison report."""
    import json

    stats = []
    for col in reference.columns[:20]:  # Top 20 features
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()
        drift_pct = abs(ref_mean - cur_mean) / (abs(ref_mean) + 1e-8) * 100
        stats.append({
            'feature': col,
            'ref_mean': round(ref_mean, 4),
            'cur_mean': round(cur_mean, 4),
            'drift_pct': round(drift_pct, 2),
            'drifted': drift_pct > 10
        })

    html = """<!DOCTYPE html>
<html><head><title>Data Drift Report</title>
<style>
body { font-family: 'Inter', sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
h1 { color: #a5b4fc; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #1e293b; color: #94a3b8; }
.drift { color: #ef4444; font-weight: bold; }
.ok { color: #22c55e; }
</style></head><body>
<h1>Data Drift Report</h1>
<p>Reference: Training set (5000 samples) vs Current: Test set (2000 samples)</p>
<table><tr><th>Feature</th><th>Ref Mean</th><th>Current Mean</th><th>Drift %</th><th>Status</th></tr>"""

    for s in stats:
        cls = 'drift' if s['drifted'] else 'ok'
        status = 'DRIFT' if s['drifted'] else 'OK'
        html += f"<tr><td>{s['feature']}</td><td>{s['ref_mean']}</td><td>{s['cur_mean']}</td><td>{s['drift_pct']}%</td><td class='{cls}'>{status}</td></tr>"

    html += "</table></body></html>"

    drift_path = os.path.join(REPORTS_DIR, 'drift_report.html')
    with open(drift_path, 'w') as f:
        f.write(html)
    print(f"   Manual drift report saved to {drift_path}")


if __name__ == '__main__':
    generate_drift_report()
