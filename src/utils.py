import pandas as pd
import numpy as np
import re

def extract_r_peak_features(r_peaks_list):
    if isinstance(r_peaks_list, str):
        r_peaks_list = list(map(int, re.findall(r'\d+', r_peaks_list)))
    else:
        r_peaks_list = list(r_peaks_list)

    rr_intervals = np.diff(r_peaks_list)
    return pd.Series({
        'std_rr': np.std(rr_intervals)
    })


def get_data_split(ecgs: pd.DataFrame, labels: np.ndarray):
    """Assigns binary labels for healthy (0) and AF (1), preprocesses features, splits into train/test."""
    ecgs = ecgs.copy()
    ecgs['label'] = None

    healthy_indices = (labels[:, 7] == 1) & (labels[:, 15] == 1)
    af_indices = labels[:, 2] == 1  # AF in column 2

    ecgs.loc[healthy_indices, 'label'] = 0
    ecgs.loc[af_indices, 'label'] = 1

    ecgs.dropna(inplace=True)
    ecgs.drop(columns=['sample_idx', 'lead', 'r_peaks'], inplace=True, errors='ignore')
    ecgs['label'] = ecgs['label'].astype(int)

    X = ecgs.drop('label', axis=1)
    y = ecgs['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, pd.DataFrame(X_scaled, columns=X.columns)