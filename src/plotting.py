import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_corr_heatmap(df, model_name):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Heatmap - {model_name}")
    plt.show()

def plot_feature_importance(clf, feature_names):
    coef = np.abs(clf.coef_[0])
    sorted_idx = np.argsort(coef)[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_coef = coef[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_coef)
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance - Logistic Regression')
    plt.gca().invert_yaxis()
    plt.show()
