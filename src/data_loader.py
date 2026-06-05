"""
NSL-KDD Dataset Loader
Downloads and preprocesses the NSL-KDD dataset for network intrusion detection.
"""

import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# NSL-KDD column names (41 features + label + difficulty)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Map specific attack names to attack categories
ATTACK_MAP = {
    'normal': 'normal',
    # DoS attacks
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'worm': 'DoS', 'mailbomb': 'DoS',
    # Probe attacks
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L attacks
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
}

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"


def download_file(url, filepath):
    """Download a file if it doesn't already exist."""
    if os.path.exists(filepath):
        print(f"  [OK] {os.path.basename(filepath)} already exists.")
        return
    print(f"  Downloading {os.path.basename(filepath)}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"  [OK] Downloaded {os.path.basename(filepath)}")


def load_nsl_kdd(data_dir="data"):
    """Download and load the NSL-KDD train and test sets."""
    print("\n[1/4] Loading NSL-KDD dataset...")

    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    download_file(TRAIN_URL, train_path)
    download_file(TEST_URL, test_path)

    df_train = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)

    print(f"  Train: {len(df_train)} records, Test: {len(df_test)} records")
    return df_train, df_test


def preprocess(df_train, df_test):
    """Encode categorical features, create labels, and scale numeric features."""
    print("\n[2/4] Preprocessing data...")

    # Map attack labels to categories
    df_train['attack_category'] = df_train['label'].map(ATTACK_MAP).fillna('unknown')
    df_test['attack_category'] = df_test['label'].map(ATTACK_MAP).fillna('unknown')

    # Binary label: normal (0) vs attack (1)
    df_train['binary_label'] = (df_train['attack_category'] != 'normal').astype(int)
    df_test['binary_label'] = (df_test['attack_category'] != 'normal').astype(int)

    # Remove rows with unknown attack types
    df_train = df_train[df_train['attack_category'] != 'unknown']
    df_test = df_test[df_test['attack_category'] != 'unknown']

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        label_encoders[col] = le

    # Feature columns (exclude label, difficulty, attack_category, binary_label)
    feature_cols = [c for c in COLUMN_NAMES if c not in ['label', 'difficulty']]

    X_train = df_train[feature_cols].values.astype(np.float64)
    X_test = df_test[feature_cols].values.astype(np.float64)

    # Scale features to [0, 1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Binary labels
    y_train_binary = df_train['binary_label'].values
    y_test_binary = df_test['binary_label'].values

    # Multi-class labels (attack category)
    cat_encoder = LabelEncoder()
    y_train_multi = cat_encoder.fit_transform(df_train['attack_category'].values)
    y_test_multi = cat_encoder.transform(df_test['attack_category'].values)

    class_names = cat_encoder.classes_.tolist()

    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train binary: {np.bincount(y_train_binary)} (normal/attack)")
    print(f"  Attack categories: {class_names}")
    print(f"  Train multi-class distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(y_train_multi == i)
        print(f"    {name}: {count}")

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train_binary': y_train_binary, 'y_test_binary': y_test_binary,
        'y_train_multi': y_train_multi, 'y_test_multi': y_test_multi,
        'feature_cols': feature_cols, 'class_names': class_names,
        'scaler': scaler, 'label_encoders': label_encoders,
        'cat_encoder': cat_encoder,
    }
