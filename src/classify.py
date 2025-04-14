import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import extract_r_peak_features, get_data_split
from plotting import plot_corr_heatmap, plot_feature_importance


def classify_real_synth_healthy_af(label: str, use_peaks=True):
    assert label in ['healthy', 'af'], "Label must be either 'healthy' or 'af'"

    path = f'data/processed_data/features/real_vs_synth_separate/real_synth_{label}.csv'
    df = pd.read_csv(path)

    # Separate real and synthetic ECGs
    real = df[df['label'] == 0].dropna()
    synth = df[df['label'] == 1].dropna()

    # Balance classes
    synth = synth.sample(n=real.shape[0], random_state=42)

    # Concatenate real and synthetic for classification
    df_combined = pd.concat([real, synth], axis=0).reset_index(drop=True)

    # Extract RR interval features
    r_peak_features = df_combined['r_peaks'].apply(extract_r_peak_features)
    df = pd.concat([df_combined, r_peak_features], axis=1)

    drop_cols = ['sample_idx', 'lead', 'r_peaks']
    if not use_peaks:
        drop_cols += ['p_peaks', 't_peaks']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Prepare for classification
    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, stratify=y, random_state=42
    )

    models = [LogisticRegression, SVC, KNeighborsClassifier, RandomForestClassifier]

    for model in models:
        if model == SVC:
            clf = model(kernel="linear", probability=True, random_state=42)
        elif model == KNeighborsClassifier:
            clf = model(n_neighbors=5)
        else:
            clf = model(random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"Classification Report for {model.__name__}:\n")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.title(f"Confusion Matrix - {model.__name__}")
        plt.show()

        plot_corr_heatmap(pd.DataFrame(X_scaled, columns=X.columns), model.__name__)

        if model == LogisticRegression:
            plot_feature_importance(clf, X.columns)


def classify_healthy_af(
    data='real',
    real_csv_path=None,
    real_labels_path=None,
    synth_csv_path=None,
    synth_labels_path=None,
    X_train_manual=None,
    y_train_manual=None,
):
    """
    Classifies healthy vs AF ECGs using real or synthetic data.

    Parameters:
        data (str): 'real' or 'synth'
        real_csv_path (str): Path to real ECG features CSV
        real_labels_path (str): Path to real ECG labels (.npy)
        synth_csv_path (str): Path to synthetic ECG features CSV
        synth_labels_path (str): Path to synthetic ECG labels (.npy)
        X_train_manual (ndarray): Optional manual training features (required for 'synth')
        y_train_manual (ndarray): Optional manual training labels (required for 'synth')
    """
    np.random.seed(0)
    random.seed(0)

    if data == 'real':
        assert real_csv_path and real_labels_path, "Must provide real data paths."
        df = pd.read_csv(real_csv_path)
        labels = np.load(real_labels_path)

        X_train, X_test, y_train, y_test, X_full = get_data_split(df, labels)

    elif data == 'synth':
        assert synth_csv_path and synth_labels_path, "Must provide synthetic data paths."
        assert X_train_manual is not None and y_train_manual is not None, "Must provide real train data manually for synthetic eval."

        df = pd.read_csv(synth_csv_path)
        labels = np.load(synth_labels_path)

        _, X_test, _, y_test, X_full = get_data_split(df, labels)

        X_train = X_train_manual
        y_train = y_train_manual

        # Optional: randomly drop a few samples from synthetic test set
        remove_idx = random.sample(range(0, X_test.shape[0]), min(8, X_test.shape[0]))
        X_test = np.delete(X_test, remove_idx, axis=0)
        y_test = np.delete(y_test, remove_idx, axis=0)

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    else:
        raise ValueError("Invalid `data` argument. Use 'real' or 'synth'.")

    clf = RandomForestClassifier(random_state=42)

    print("\nTraining RandomForestClassifier...")
    clf.fit(X_train, y_train)

    # Correlation matrix from full feature set (X_full)
    corr_matrix = X_full.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Heatmap - RandomForestClassifier")
    plt.show()

    # Confusion matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.title(f"Confusion Matrix - RandomForestClassifier")
    plt.show()

    # Classification report
    print(f"Classification Report - RandomForestClassifier:\n")
    print(classification_report(y_test, y_pred))


def main():
    parser = argparse.ArgumentParser(description="ECG Classification for Healthy vs AF and Real vs Synthetic")

    # Add arguments for classification
    parser.add_argument(
        "--task", choices=["real_vs_synth", "healthy_vs_af"], required=True,
        help="Task to perform: 'real_vs_synth' or 'healthy_vs_af'"
    )

    parser.add_argument(
        "--label", choices=["healthy", "af"],
        help="Label for real vs synthetic classification (only required for 'real_vs_synth')"
    )

    parser.add_argument(
        "--use_peaks", type=bool, default=True,
        help="Whether to include peak-based features (True/False)"
    )

    parser.add_argument(
        "--data", choices=["real", "synth"], default="real",
        help="Type of data for 'healthy_vs_af' classification: 'real' or 'synth'"
    )

    parser.add_argument(
        "--real_csv_path", type=str, help="Path to the real ECG features CSV"
    )

    parser.add_argument(
        "--real_labels_path", type=str, help="Path to the real ECG labels (.npy)"
    )

    parser.add_argument(
        "--synth_csv_path", type=str, help="Path to the synthetic ECG features CSV"
    )

    parser.add_argument(
        "--synth_labels_path", type=str, help="Path to the synthetic ECG labels (.npy)"
    )

    args = parser.parse_args()

    if args.task == "real_vs_synth":
        if args.label is None:
            print("Label must be specified for 'real_vs_synth' task.")
        else:
            classify_real_synth_healthy_af(args.label, args.use_peaks)
    elif args.task == "healthy_vs_af":
        classify_healthy_af(
            data=args.data,
            real_csv_path=args.real_csv_path,
            real_labels_path=args.real_labels_path,
            synth_csv_path=args.synth_csv_path,
            synth_labels_path=args.synth_labels_path
        )


if __name__ == "__main__":
    main()


