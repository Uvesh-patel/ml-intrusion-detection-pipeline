"""
Data loader for the OCPP 1.6 WebSocket Intrusion Detection Dataset.

This dataset contains network traffic from EV (Electric Vehicle) charging
stations communicating over WebSocket using the OCPP 1.6 protocol. It was
collected in a federated setting with 3 separate charging station clients.

Attack types:
    - normal: legitimate OCPP communication
    - cyberattack_ocpp16_dos_flooding_heartbeat: DoS via heartbeat flooding
    - cyberattack_ocpp16_unauthorized_access: unauthorized OCPP commands
    - cyberattack_ocpp16_fdi_chargingprofile: false data injection in profiles
    - cyberattack_ocpp16_doc_idtag: denial of charge via invalid ID tags

Two feature layers available:
    - TCP/IP layer: 87 network flow features (CICFlowMeter)
    - Application layer: 49 OCPP/WebSocket-specific features (OCPPFlowMeter)

The 3-client structure provides a natural domain adaptation scenario:
train on Client A, test on Client B (different station = different distribution).

Dataset source:
    Dalamagkas et al. (2025) "Federated Detection of Open Charge Point
    Protocol 1.6 Cyberattacks" arXiv:2502.01569
    Zenodo: https://zenodo.org/records/14887131

Author: Uvesh Patel
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


OCPP_DATA_DIR = "data/ocpp"

# Short labels for readability
ATTACK_SHORT = {
    "normal": "Normal",
    "cyberattack_ocpp16_dos_flooding_heartbeat": "DoS",
    "cyberattack_ocpp16_unauthorized_access": "UnauthorizedAccess",
    "cyberattack_ocpp16_fdi_chargingprofile": "FDI",
    "cyberattack_ocpp16_doc_idtag": "DenialOfCharge",
}

# Columns to drop (identifiers, not features)
DROP_COLS_TCP = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port",
                 "Protocol", "Timestamp", "Label"]
DROP_COLS_APP = ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
                 "flow_start_timestamp", "flow_end_timestamp", "label"]


def load_ocpp_client(client_id, layer="tcp", data_dir=OCPP_DATA_DIR):
    """
    Load train/test data for a specific client (charging station).

    Args:
        client_id: 1, 2, or 3
        layer: "tcp" for TCP/IP features, "app" for application-layer features
        data_dir: base directory for OCPP data

    Returns:
        (train_df, test_df) with raw features and labels
    """
    if layer == "tcp":
        base = os.path.join(data_dir, "Balanced_OCPP16_TCP-IP_Layer",
                            f"Client_{client_id}")
    else:
        base = os.path.join(data_dir, "Balanced_OCPP16_APP_Layer",
                            f"Client_{client_id}")

    train_df = pd.read_csv(os.path.join(base, "Train.csv"))
    test_df = pd.read_csv(os.path.join(base, "Test.csv"))
    return train_df, test_df


def load_ocpp_combined(layer="tcp", data_dir=OCPP_DATA_DIR):
    """Load the combined (all clients) train/test split."""
    if layer == "tcp":
        base = os.path.join(data_dir, "Balanced_OCPP16_TCP-IP_Layer", "Combined")
    else:
        base = os.path.join(data_dir, "Balanced_OCPP16_APP_Layer", "Combined")

    train_df = pd.read_csv(os.path.join(base, "Train.csv"))
    test_df = pd.read_csv(os.path.join(base, "Test.csv"))
    return train_df, test_df


def preprocess_ocpp(df, layer="tcp", scaler=None, label_encoder=None,
                    feature_columns=None):
    """
    Preprocess a single OCPP DataFrame.

    Steps:
        1. Extract label column
        2. Drop identifier columns
        3. Convert to numeric, handle inf/NaN
        4. Scale features

    Args:
        df: raw DataFrame
        layer: "tcp" or "app" (determines which columns to drop)
        scaler: fitted StandardScaler (None = fit new one)
        label_encoder: fitted LabelEncoder (None = fit new one)
        feature_columns: fixed list of feature columns to use (for consistency
                         between source and target)

    Returns:
        X: scaled feature array
        y_binary: 0=normal, 1=attack
        y_multi: integer-encoded attack type
        feature_names: list of feature column names
        scaler: the fitted scaler (for reuse on test data)
        label_encoder: the fitted encoder
    """
    # Get labels
    label_col = "Label" if "Label" in df.columns else "label"
    labels = df[label_col].str.strip().str.lower()

    # Drop non-feature columns
    drop_cols = DROP_COLS_TCP if layer == "tcp" else DROP_COLS_APP
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Numeric conversion
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Use fixed feature columns if provided, otherwise select non-constant
    if feature_columns is not None:
        # Use only the columns that exist in this dataframe
        valid_cols = [c for c in feature_columns if c in X.columns]
        X = X[valid_cols]
        # Add any missing columns as zeros
        for c in feature_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_columns]
    else:
        non_constant = X.columns[X.std() > 0]
        X = X[non_constant]

    feature_names = list(X.columns)

    # Scale
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = scaler.transform(X.values)

    # Binary labels: normal=0, attack=1
    y_binary = (labels != "normal").astype(int).values

    # Multi-class labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_multi = label_encoder.fit_transform(labels.values)
    else:
        y_multi = label_encoder.transform(labels.values)

    return X_scaled, y_binary, y_multi, feature_names, scaler, label_encoder


def load_cross_station(source_client, target_client, layer="tcp",
                       data_dir=OCPP_DATA_DIR):
    """
    Load data for cross-station domain adaptation experiment.

    Train on source_client, test on target_client. This creates a natural
    domain shift because different charging stations have different traffic
    patterns even under the same attack types.

    Args:
        source_client: client ID for training (1, 2, or 3)
        target_client: client ID for testing (1, 2, or 3)
        layer: "tcp" or "app"

    Returns:
        dict with source/target train/test arrays and metadata
    """
    # Load source client
    src_train_df, src_test_df = load_ocpp_client(source_client, layer, data_dir)
    src_all = pd.concat([src_train_df, src_test_df], ignore_index=True)

    # Load target client
    tgt_train_df, tgt_test_df = load_ocpp_client(target_client, layer, data_dir)
    tgt_all = pd.concat([tgt_train_df, tgt_test_df], ignore_index=True)

    # Preprocess source (fit scaler and encoder here)
    X_source, y_src_bin, y_src_multi, feat_names, scaler, le = \
        preprocess_ocpp(src_all, layer=layer)

    # Preprocess target with source scaler and same feature columns
    X_target, y_tgt_bin, y_tgt_multi, _, _, _ = \
        preprocess_ocpp(tgt_all, layer=layer, scaler=scaler,
                        label_encoder=le, feature_columns=feat_names)

    class_names = list(le.classes_)

    print(f"  Source (Client {source_client}): {len(X_source)} samples")
    print(f"  Target (Client {target_client}): {len(X_target)} samples")
    print(f"  Features: {len(feat_names)} ({layer} layer)")
    print(f"  Classes: {class_names}")

    return {
        "X_source": X_source,
        "y_source_binary": y_src_bin,
        "y_source_multi": y_src_multi,
        "X_target": X_target,
        "y_target_binary": y_tgt_bin,
        "y_target_multi": y_tgt_multi,
        "feature_names": feat_names,
        "class_names": class_names,
        "scaler": scaler,
        "label_encoder": le,
    }


def get_shared_features_with_cicids():
    """
    Return the list of feature names shared between OCPP (TCP/IP layer)
    and CICIDS2017. Both datasets use CICFlowMeter, so many features overlap.
    This enables cross-dataset domain adaptation experiments.
    """
    shared = [
        "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
        "Total Length of Fwd Packet", "Total Length of Bwd Packet",
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Length", "Bwd Header Length",
        "Fwd Packets/s", "Bwd Packets/s",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "Down/Up Ratio", "Average Packet Size",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ]
    return shared
