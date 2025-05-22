# geochem_classifier_gui/core/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import streamlit as st
import pandas as pd # For feature importance display

# Define your class names as used in training (after label encoding if any)
# e.g., if 'Au-rich' was 0 and 'Cu-rich' was 1
CLASS_NAMES = ['Au-rich', 'Cu-rich'] # Or however you map them

def plot_confusion_matrix_func(y_true, y_pred, class_names=None):
    """Generates and returns a Matplotlib figure for the confusion matrix."""
    if class_names is None:
        class_names = CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    return fig

def plot_roc_curve_func(y_true, y_pred_proba, model_name="Model"):
    """Generates and returns a Matplotlib figure for the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def plot_precision_recall_curve_func(y_true, y_pred_proba, model_name="Model"):
    """Generates and returns a Matplotlib figure for the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig

def get_feature_importances(model, feature_names, model_type):
    """Extracts feature importances from tree-based or SVM models."""
    if model_type in ["Random Forest", "XGBoost"] and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
    elif model_type == "SVM" and hasattr(model, 'coef_') and model.kernel == 'linear':
        # For linear SVM, coefficients can be used as importances
        importances = model.coef_[0] # Assuming binary classification
        return pd.Series(abs(importances), index=feature_names).sort_values(ascending=False)
    return None

def plot_feature_importances_func(importances_series, model_name):
    """Generates a bar chart for feature importances."""
    if importances_series is None or importances_series.empty:
        return None
    
    top_n = min(len(importances_series), 15) # Show top N features
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances_series.head(top_n).values, y=importances_series.head(top_n).index, ax=ax, palette="viridis")
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    plt.tight_layout()
    return fig

# For SHAP plots in Model Insights - these are usually pre-generated images or complex to generate on-the-fly
# Example for loading a pre-generated image:
# def load_static_shap_summary_plot(model_name):
#     try:
#         image_path = f"assets/shap_summary_{model_name.lower().replace(' ', '_')}.png"
#         return image_path # Or load with PIL Image.open(image_path)
#     except FileNotFoundError:
#         return None