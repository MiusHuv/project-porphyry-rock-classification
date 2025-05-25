# geochem_classifier_gui/core/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import streamlit as st
import pandas as pd # For feature importance display
from util.language import T
import matplotlib

# Define your class names as used in training (after label encoding if any)
# e.g., if 'Au-rich' was 0 and 'Cu-rich' was 1
CLASS_NAMES = ['Au-rich', 'Cu-rich'] # Or however you map them

_FONT_SET_FOR_CHINESE = False

def set_matplotlib_font_for_chinese(font_names=None):
    """
    Attempts to set a Matplotlib font that supports Chinese characters.
    Call this once before generating plots if Chinese labels are expected.
    Args:
        font_names (list, optional): A list of font names to try. 
                                     Defaults to ['SimHei', 'Microsoft YaHei', 'sans-serif'].
    """
    global _FONT_SET_FOR_CHINESE
    if _FONT_SET_FOR_CHINESE:
        return

    if font_names is None:
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
    
    try:
        current_lang = st.session_state.get('lang', 'en') # Get current language
        if current_lang == 'zh': # Only set for Chinese
            matplotlib.rcParams['font.family'] = 'sans-serif' # Important to set family first
            matplotlib.rcParams['font.sans-serif'] = font_names
            matplotlib.rcParams['axes.unicode_minus'] = False  # Resolve issues with minus sign display
            # Test if a known Chinese character renders, very basic check
            # fig, ax = plt.subplots()
            # ax.text(0.5, 0.5, "你好")
            # plt.close(fig) 
            # print(f"Attempted to set Matplotlib font for Chinese: {matplotlib.rcParams['font.sans-serif']}") # For debugging
            _FONT_SET_FOR_CHINESE = True # Mark as set to avoid re-running
        else:
            # Optionally reset to default or ensure non-Chinese fonts are prioritized for other languages
            # For simplicity, we only actively change for 'zh'
            pass
            
    except Exception as e:
        # This warning is better shown in the UI by the calling page if font issues are detected
        # print(f"Note: Could not set specific Matplotlib font for Chinese automatically: {e}")
        # A general warning can be added in help_about.py or if a plot fails to render correctly.
        pass


def plot_confusion_matrix_func(y_true, y_pred, class_names=None):
    """Generates and returns a Matplotlib figure for the confusion matrix."""
    set_matplotlib_font_for_chinese() # Attempt to set font if Chinese is expected
    
    if class_names is None:
        # These default class names might need to be translated if used directly.
        # It's better if the calling function (e.g., in performance_visualizer.py)
        # passes already translated class_names.
        # For example, CLASS_NAMES could be [T("class_au_rich"), T("class_cu_rich")]
        # However, for direct use here, we'll use the global CLASS_NAMES or provided ones.
        effective_class_names = CLASS_NAMES 
    else:
        effective_class_names = class_names

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=effective_class_names, yticklabels=effective_class_names)
    
    ax.set_title(T("viz_cm_title", default="Confusion Matrix"))
    ax.set_xlabel(T("viz_cm_xlabel", default="Predicted Label"))
    ax.set_ylabel(T("viz_cm_ylabel", default="True Label"))
    plt.tight_layout()
    return fig

def plot_roc_curve_func(y_true, y_pred_proba, model_name="Model"):
    """Generates and returns a Matplotlib figure for the ROC curve."""
    set_matplotlib_font_for_chinese()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    roc_label = T("viz_roc_label", auc_score=f"{roc_auc:.2f}", default="ROC curve (AUC = {auc_score})")
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=roc_label)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(T("viz_roc_xlabel", default="False Positive Rate"))
    ax.set_ylabel(T("viz_roc_ylabel", default="True Positive Rate"))
    title_key = "viz_roc_title"
    ax.set_title(T(title_key, model_name=model_name, default="Receiver Operating Characteristic (ROC) - {model_name}"))
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def plot_precision_recall_curve_func(y_true, y_pred_proba, model_name="Model"):
    """Generates and returns a Matplotlib figure for the Precision-Recall curve."""
    set_matplotlib_font_for_chinese()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 5))

    pr_label = T("viz_pr_label", ap_score=f"{avg_precision:.2f}", default="PR curve (AP = {ap_score})")
    ax.plot(recall, precision, color='blue', lw=2, label=pr_label)
    ax.set_xlabel(T("viz_pr_xlabel", default="Recall"))
    ax.set_ylabel(T("viz_pr_ylabel", default="Precision"))
    title_key = "viz_pr_title"
    ax.set_title(T(title_key, model_name=model_name, default="Precision-Recall Curve - {model_name}"))
    ax.legend(loc="lower left") # "lower left" is standard, no need to translate the location string itself
    plt.tight_layout()
    return fig

def get_feature_importances(model, feature_names, model_type):
    """Extracts feature importances from tree-based or SVM models."""
    # This function does not produce user-facing text directly that needs translation.
    # Feature names are data-dependent.
    if model_type in ["Random Forest", "XGBoost"] and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
    elif model_type == "SVM" and hasattr(model, 'coef_') and model.kernel == 'linear':
        importances = model.coef_[0] 
        return pd.Series(abs(importances), index=feature_names).sort_values(ascending=False)
    return None

def plot_feature_importances_func(importances_series, model_name):
    """Generates a bar chart for feature importances."""
    set_matplotlib_font_for_chinese()

    if importances_series is None or importances_series.empty:
        return None # No plot to generate
    
    top_n = min(len(importances_series), 15) # Show top N features
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4))) # Adjust height based on N
    
    # Feature names (y-axis labels) are data, not translated here.
    sns.barplot(x=importances_series.head(top_n).values, y=importances_series.head(top_n).index, ax=ax, palette="viridis")
    
    title_key = "viz_fi_title"
    ax.set_title(T(title_key, n_features=top_n, model_name=model_name, default="Top {n_features} Feature Importances - {model_name}"))
    ax.set_xlabel(T("viz_fi_xlabel", default="Importance Score"))
    ax.set_ylabel(T("viz_fi_ylabel", default="Features"))
    plt.tight_layout()
    return fig