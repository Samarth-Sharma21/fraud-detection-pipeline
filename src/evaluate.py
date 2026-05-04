"""
evaluate.py — Model Evaluation & Visualization
================================================
Generates evaluation plots: ROC curve, precision-recall curve,
confusion matrix heatmap, and feature importance chart.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')


def generate_evaluation_plots():
    """Generate and save all evaluation visualizations."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load predictions
    preds = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_predictions.csv'))
    y_true = preds['y_true'].values
    y_pred = preds['y_pred'].values
    y_prob = preds['y_prob'].values

    # Load model metadata
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Set dark theme for professional look
    plt.style.use('dark_background')
    sns.set_theme(style='darkgrid')

    # ─── 1. ROC Curve ────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#00D4FF', lw=2.5, label=f'{metadata["model_name"]} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#FF6B6B', lw=1.5, linestyle='--', alpha=0.7, label='Random Baseline')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#00D4FF')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Fraud Detection Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ ROC Curve saved")

    # ─── 2. Precision-Recall Curve ───────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#FFD93D', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    ax.fill_between(recall, precision, alpha=0.15, color='#FFD93D')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'precision_recall_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Precision-Recall Curve saved")

    # ─── 3. Confusion Matrix ────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud'],
        ax=ax, annot_kws={'size': 16}
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Confusion Matrix saved")

    # ─── 4. Feature Importance (Top 20) ──────────────────────────────
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))

    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.95, len(importance)))
        ax.barh(importance['feature'], importance['importance'], color=colors)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Feature Importance chart saved")

    print(f"\n📊 All evaluation plots saved to {REPORTS_DIR}")


if __name__ == '__main__':
    generate_evaluation_plots()
