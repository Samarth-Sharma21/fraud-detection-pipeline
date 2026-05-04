"""
preprocess.py — Data Loading, Merging, Cleaning & Feature Engineering
=====================================================================
Loads the IEEE-CIS Fraud Detection dataset, merges transaction + identity
tables, engineers interpretable features, and saves train/test splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ───────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Features we keep — chosen for interpretability + predictive power
NUMERIC_FEATURES = [
    'TransactionAmt', 'dist1', 'dist2',
    'C1', 'C2', 'C5', 'C6', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D15',
    'V12', 'V13', 'V14', 'V53', 'V54', 'V56', 'V75', 'V76', 'V78',
    'V82', 'V83', 'V87', 'V258', 'V261', 'V279', 'V280', 'V282',
    'V283', 'V285', 'V294', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313',
    'V314', 'V315', 'V317', 'V318', 'V323', 'V324', 'V331', 'V332'
]

CATEGORICAL_FEATURES = [
    'ProductCD', 'card4', 'card6',
    'P_emaildomain', 'R_emaildomain',
    'DeviceType', 'DeviceInfo_cleaned'
]

TARGET = 'isFraud'


def load_and_merge(sample_frac=0.3, random_state=42):
    """Load raw CSVs, merge on TransactionID, and optionally subsample."""
    print("📂 Loading raw datasets...")

    txn_path = os.path.join(RAW_DIR, 'train_transaction.csv')
    id_path = os.path.join(RAW_DIR, 'train_identity.csv')

    # Load transaction data
    df_txn = pd.read_csv(txn_path)
    print(f"   Transactions loaded: {df_txn.shape}")

    # Load identity data
    df_id = pd.read_csv(id_path)
    print(f"   Identity loaded: {df_id.shape}")

    # Merge on TransactionID (left join — not all transactions have identity)
    df = pd.merge(df_txn, df_id, on='TransactionID', how='left')
    print(f"   Merged dataset: {df.shape}")

    # Subsample to keep training fast on a regular laptop
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state, ignore_index=True)
        print(f"   Subsampled to {sample_frac*100:.0f}%: {df.shape}")

    fraud_rate = df[TARGET].mean() * 100
    print(f"   Fraud rate: {fraud_rate:.2f}%")
    return df


def clean_device_info(device):
    """Simplify DeviceInfo into clean categories."""
    if pd.isna(device):
        return 'Unknown'
    device = str(device).lower()
    if 'windows' in device:
        return 'Windows'
    elif 'ios' in device or 'iphone' in device or 'ipad' in device:
        return 'iOS'
    elif 'mac' in device:
        return 'MacOS'
    elif 'linux' in device:
        return 'Linux'
    elif 'samsung' in device:
        return 'Samsung'
    elif 'huawei' in device or 'hi6210sft' in device:
        return 'Huawei'
    elif any(brand in device for brand in ['lg', 'moto', 'pixel', 'nexus', 'oneplus', 'xiaomi', 'redmi', 'oppo', 'vivo']):
        return 'Android_Other'
    elif 'android' in device or 'rv:' in device:
        return 'Android_Other'
    else:
        return 'Other'


def clean_email_domain(email):
    """Group email domains into interpretable categories."""
    if pd.isna(email):
        return 'Unknown'
    email = str(email).lower()
    if 'gmail' in email:
        return 'Gmail'
    elif 'yahoo' in email or 'ymail' in email:
        return 'Yahoo'
    elif 'hotmail' in email or 'outlook' in email or 'live' in email or 'msn' in email:
        return 'Microsoft'
    elif 'aol' in email:
        return 'AOL'
    elif 'icloud' in email or 'mac.com' in email or 'me.com' in email:
        return 'Apple'
    elif 'protonmail' in email or 'proton' in email:
        return 'Protonmail'
    else:
        return 'Other'


def engineer_features(df):
    """Create new features and clean existing ones for modeling."""
    print("⚙️  Engineering features...")

    # Clean DeviceInfo
    df['DeviceInfo_cleaned'] = df['DeviceInfo'].apply(clean_device_info)

    # Clean email domains
    df['P_emaildomain'] = df['P_emaildomain'].apply(clean_email_domain)
    df['R_emaildomain'] = df['R_emaildomain'].apply(clean_email_domain)

    # Log-transform TransactionAmt (heavy right skew)
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])

    # Transaction hour from TransactionDT (seconds from reference)
    df['Transaction_hour'] = (df['TransactionDT'] // 3600) % 24
    df['Transaction_dayofweek'] = (df['TransactionDT'] // 86400) % 7

    print("   ✅ Features engineered")
    return df


def encode_and_prepare(df):
    """Encode categoricals, handle NaN, and select final feature set."""
    print("🔧 Encoding categorical features...")

    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        # Fill NaN with 'Unknown' before encoding
        df[col] = df[col].fillna('Unknown').astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Add engineered numeric features
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['TransactionAmt_log', 'Transaction_hour', 'Transaction_dayofweek']

    # Keep only features that exist in the dataframe
    available_features = [f for f in all_features if f in df.columns]
    print(f"   Using {len(available_features)} features")

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Fill remaining NaN with median for numeric columns
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    print("   ✅ Encoding complete")
    return X, y, label_encoders, available_features


def run_preprocessing(sample_frac=0.3):
    """Full preprocessing pipeline: load → clean → engineer → split → save."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 1: Load and merge
    df = load_and_merge(sample_frac=sample_frac)

    # Step 2: Engineer features
    df = engineer_features(df)

    # Step 3: Encode and prepare
    X, y, label_encoders, feature_names = encode_and_prepare(df)

    # Step 4: Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Split results:")
    print(f"   Train: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% fraud)")
    print(f"   Test:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% fraud)")

    # Step 5: Scale numeric features
    scaler = StandardScaler()
    numeric_cols = [c for c in NUMERIC_FEATURES + ['TransactionAmt_log', 'Transaction_hour', 'Transaction_dayofweek'] if c in X_train.columns]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Step 6: Save everything
    X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.joblib'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))

    print(f"\n💾 Saved processed data to {PROCESSED_DIR}")
    print(f"💾 Saved scaler & encoders to {MODELS_DIR}")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    run_preprocessing(sample_frac=0.5)
