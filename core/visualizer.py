import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import os
import torch
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP library not found. Please install with: pip install shap. SHAP plots will be skipped.")
    SHAP_AVAILABLE = False

def plot_loss_vs_trees(train_scores, val_scores, n_estimators_range, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_estimators_range, train_scores, label="Training Score", marker='o', linestyle='-') # T("Training Score")
    ax.plot(n_estimators_range, val_scores, label="Validation Score", marker='o', linestyle='-') # T("Validation Score")
    ax.set_xlabel("Number of Trees") # T("Number of Trees")
    ax.set_ylabel("Accuracy Score") # T("Accuracy Score")
    ax.set_title(f"Score vs. Number of Trees for {model_name}") # T("Score vs. Number of Trees for {model_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    optimal_n_trees_idx = np.argmax(val_scores)
    optimal_n_trees = n_estimators_range[optimal_n_trees_idx]
    print(f"Optimal number of trees for {model_name} based on validation score: {optimal_n_trees} (Validation Score: {val_scores[optimal_n_trees_idx]:.4f})")
    # T("Justification: This is the number of trees where the validation score is maximized...")
    print("Justification: This is the number of trees where the validation score is maximized, "
          "aiming for a balance between model complexity and generalization performance before potential overfitting.")
    return fig, optimal_n_trees # Return fig and optimal_n_trees

def plot_feature_importances(importances, feature_names, model_name, top_n=20):
    if importances is None or len(importances) == 0:
        print(f"No feature importances available or feature_names mismatch for {model_name}.")
        return None
    if len(importances) != len(feature_names):
        print(f"Warning for {model_name}: Mismatch between number of importances ({len(importances)}) and feature_names ({len(feature_names)}). Plotting skipped.")
        return None

    importances_series = pd.Series(importances, index=feature_names)
    sorted_importances = importances_series.sort_values(ascending=False)
    top_n = min(top_n, len(sorted_importances))
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 1.8))) # Adjusted height
    sns.barplot(x=sorted_importances.head(top_n).values, 
                y=sorted_importances.head(top_n).index, 
                hue=sorted_importances.head(top_n).index,
                palette= "viridis", # Use a color palette for better visibility
                orient='h',
                legend=False,
                ax=ax)
    ax.set_xlabel("Importance") # T("Importance")
    ax.set_ylabel("Feature") # T("Feature")
    ax.set_title(f"Top {top_n} Feature Importances for {model_name}") # T("Top {top_n} Feature Importances for {model_name}")
    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names_list, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list, ax=ax)
    ax.set_xlabel("Predicted Label") # T("Predicted Label")
    ax.set_ylabel("True Label") # T("True Label")
    ax.set_title(f"Confusion Matrix for {model_name}") # T("Confusion Matrix for {model_name}")
    plt.tight_layout()
    return fig

def generate_and_save_shap_plots(model, X_sample_df_orig, feature_names_list, model_type_str,
                                 num_classes_model, class_names_list=None,
                                 plot_dir="output_plots/shap_plots", # Consider Path(plot_dir)
                                 X_background_df_orig=None,
                                 device_for_dnn=None):
    if not SHAP_AVAILABLE:
        print(f"SHAP ({model_type_str}): Library not available. Skipping.")
        return

    plot_dir_path = Path(plot_dir) # Use Pathlib
    X_sample_df = X_sample_df_orig.copy()
    if X_sample_df.empty:
        print(f"SHAP ({model_type_str}): X_sample_df is empty. Skipping.")
        return

    try:
        X_sample_df = X_sample_df[feature_names_list]
    except KeyError as e:
        print(f"SHAP ({model_type_str}) Error: Columns in X_sample_df do not precisely match feature_names_list. {e}")
        print(f"Sample DF columns: {X_sample_df.columns.tolist()}")
        print(f"Expected features: {feature_names_list}")
        return

    feature_names_list_str = [str(fn) for fn in feature_names_list]
    X_sample_df.columns = feature_names_list_str

    # --- Check X_sample_df for common issues ---
    if X_sample_df.isnull().values.any():
        print(f"SHAP ({model_type_str}) WARNING: X_sample_df contains NaN values. This may cause issues.")
        # Consider imputing here if it's a recurring problem, though ideally data is clean.
        # X_sample_df = X_sample_df.fillna(X_sample_df.mean()) # Example: mean imputation

    print(f"\nGenerating SHAP plots for {model_type_str} (filename key: {model_type_str})...") # Use model_type_str directly if it's already filename safe
    plot_dir_path.mkdir(parents=True, exist_ok=True) # Use Pathlib

    explainer = None
    shap_values_raw = None # This is the variable you identified as problematic

    try:
        if model_type_str.lower() in ['random_forest', 'xgboost']: # Assuming model_type_str is filename_key
            print(f"SHAP ({model_type_str}): Model type is {type(model)}")
            print(f"SHAP ({model_type_str}): X_sample_df NaNs: {X_sample_df.isnull().sum().sum()}")
            # Ensure all data is float, TreeExplainer can be picky
            X_sample_df_tree = X_sample_df.astype(float)
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_sample_df_tree, check_additivity=False)
            if shap_values_raw is None:
                print(f"SHAP ({model_type_str}) CRITICAL ERROR: TreeExplainer.shap_values() returned None.")

        elif model_type_str.lower() == 'svm' and hasattr(model, 'predict_proba'):
            print(f"SHAP ({model_type_str}): Preparing KernelExplainer...")
            background_svm = X_background_df_orig if X_background_df_orig is not None else X_sample_df
            # Ensure background_svm has string column names if X_sample_df does
            background_svm.columns = feature_names_list_str
            if len(background_svm) > 50:
                background_svm_sample = shap.sample(background_svm, 50, random_state=42)
            else:
                background_svm_sample = background_svm.copy()

            def svm_predict_proba_for_shap(data_for_svm_np):
                data_for_svm_df = pd.DataFrame(data_for_svm_np, columns=feature_names_list_str)
                return model.predict_proba(data_for_svm_df)
            explainer = shap.KernelExplainer(svm_predict_proba_for_shap, background_svm_sample)
            print(f"SHAP ({model_type_str}): Calculating SHAP values with KernelExplainer...")
            shap_values_raw = explainer.shap_values(X_sample_df, nsamples='auto', check_additivity=False)
            print(f"SHAP ({model_type_str}): SHAP values calculated by KernelExplainer.")
            if shap_values_raw is None:
                print(f"SHAP ({model_type_str}) CRITICAL ERROR: KernelExplainer.shap_values() returned None.")


        elif model_type_str.lower() == 'pytorch_dnn': # Match filename key
            if device_for_dnn is None:
                device_for_dnn = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval().to(device_for_dnn)

            if X_background_df_orig is None:
                print(f"SHAP ({model_type_str}): No background data for DeepExplainer, sampling from X_sample_df.")
                X_background_df = shap.sample(X_sample_df, min(50, len(X_sample_df)), random_state=42)
            else:
                X_background_df = X_background_df_orig.copy()
            X_background_df.columns = feature_names_list_str # Ensure string columns

            background_tensor = torch.tensor(X_background_df.values, dtype=torch.float32).to(device_for_dnn)
            X_sample_tensor = torch.tensor(X_sample_df.values, dtype=torch.float32).to(device_for_dnn)

            print(f"SHAP ({model_type_str}): Preparing DeepExplainer with background shape {background_tensor.shape}...")
            explainer = shap.DeepExplainer(model, background_tensor)
            print(f"SHAP ({model_type_str}): Calculating SHAP values with DeepExplainer for sample shape {X_sample_tensor.shape}...")
            shap_output_from_explainer = explainer.shap_values(X_sample_tensor)
            print(f"SHAP ({model_type_str}): SHAP values calculated by DeepExplainer. Type: {type(shap_output_from_explainer)}")

            if shap_output_from_explainer is None:
                print(f"SHAP ({model_type_str}) CRITICAL ERROR: DeepExplainer.shap_values() returned None.")
                shap_values_raw = None # Explicitly set to None
            elif isinstance(shap_output_from_explainer, list):
                shap_values_raw = [s_val.cpu().numpy() if torch.is_tensor(s_val) else s_val for s_val in shap_output_from_explainer]
            elif torch.is_tensor(shap_output_from_explainer):
                shap_values_raw = shap_output_from_explainer.cpu().numpy()
            else: # Already numpy or other
                shap_values_raw = shap_output_from_explainer
        else:
            print(f"SHAP ({model_type_str}): Model type not configured for SHAP in this function. Skipping.")
            return

    except Exception as e_explainer:
        print(f"SHAP ({model_type_str}) EXCEPTION during explainer instantiation or shap_values call: {e_explainer}")
        import traceback
        traceback.print_exc()
        shap_values_raw = None # Ensure it's None if any exception occurs here

    # --- Standardize shap_values structure for plotting ---
    # (This section is from your provided code, ensure it correctly handles shap_values_raw being None)
    if shap_values_raw is None: # Check added from your log
        print(f"SHAP ({model_type_str}) Error: Unexpected type for shap_values_raw: <class 'NoneType'>. Cannot proceed with plotting.")
        return
    # ... (rest of your shap_values_for_plotting and plotting logic using plot_dir_path for saving) ...
    # Make sure to use plot_dir_path when saving:
    # Example: save_path = plot_dir_path / f"{model_type_str...}.png"
    # plt.savefig(save_path, bbox_inches='tight'); plt.close()

    # --- (Your existing logic for standardizing shap_values_raw to shap_values_for_plotting) ---
    # This section seems complex and specific to how different explainers return values.
    # The key is that if shap_values_raw is None here, the following logic will fail.
    # The check above `if shap_values_raw is None: return` handles this.
    
    shap_values_for_plotting = None
    is_multi_class_for_loops = False

    if isinstance(shap_values_raw, list):
        if num_classes_model == 2 and len(shap_values_raw) == 2:
            shap_values_for_plotting = shap_values_raw[1] # Prob of positive class
            print(f"SHAP ({model_type_str}): Binary (from list[2]), using SHAP for positive class (index 1).")
        elif len(shap_values_raw) == num_classes_model:
            shap_values_for_plotting = shap_values_raw
            is_multi_class_for_loops = True
            print(f"SHAP ({model_type_str}): Multi-class (from list[{num_classes_model}]).")
        else:
            print(f"SHAP ({model_type_str}) Error: shap_values_raw is list len {len(shap_values_raw)}, but num_classes is {num_classes_model}. Mismatch."); return
    elif isinstance(shap_values_raw, np.ndarray):
        if num_classes_model == 2: # Binary Classification Cases
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[-1] == 2: # e.g. (samples, features, 2 classes)
                shap_values_for_plotting = shap_values_raw[:, :, 1]
                print(f"SHAP ({model_type_str}): Binary (from 3D array with 2 class outputs), using SHAP for positive class (index 1). Shape: {shap_values_for_plotting.shape}.")
            elif shap_values_raw.ndim == 3 and shap_values_raw.shape[-1] == 1: # e.g. (samples, features, 1 class output for PyTorch binary)
                shap_values_for_plotting = np.squeeze(shap_values_raw, axis=-1)
                print(f"SHAP ({model_type_str}): Binary (from 3D array with 1 class output, e.g., PyTorch DNN), using squeezed SHAP. Shape: {shap_values_for_plotting.shape}.")
            elif shap_values_raw.ndim == 2: # e.g. (samples, features) for XGBoost binary
                shap_values_for_plotting = shap_values_raw
                print(f"SHAP ({model_type_str}): Binary (from 2D array, e.g. XGBoost).")
            else:
                print(f"SHAP ({model_type_str}) Error: Binary case, but ndarray shape {shap_values_raw.shape} not recognized."); return
        elif num_classes_model > 2: # Multi-class
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[-1] == num_classes_model: # (samples, features, num_classes)
                shap_values_for_plotting = [shap_values_raw[:, :, i] for i in range(num_classes_model)]
                is_multi_class_for_loops = True
                print(f"SHAP ({model_type_str}): Multi-class (from 3D ndarray converted to list).")
            else:
                print(f"SHAP ({model_type_str}) Error: Multi-class, but ndarray shape {shap_values_raw.shape} not recognized for num_classes {num_classes_model}."); return
        else: # num_classes_model < 2, should not happen
            print(f"SHAP ({model_type_str}) Error: num_classes_model is {num_classes_model}. Not handled for ndarray."); return
    else:
        # This was already checked by `if shap_values_raw is None:`
        print(f"SHAP ({model_type_str}) Error: Unexpected type for shap_values_raw after explainer: {type(shap_values_raw)}"); return


    # --- Summary Plot & Dependence Plots ---
    X_sample_df_for_plot = X_sample_df.copy() # Has string columns from earlier

    if is_multi_class_for_loops and isinstance(shap_values_for_plotting, list):
        for i in range(len(shap_values_for_plotting)):
            plt.figure()
            class_label_safe = (class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}").replace(' ', '_').lower()
            shap.summary_plot(shap_values_for_plotting[i], X_sample_df_for_plot, feature_names=feature_names_list_str, show=False, plot_type="dot")
            plt.title(f"SHAP Summary for {class_label_safe} - {model_type_str}")
            save_path = plot_dir_path / f"{model_type_str}_shap_summary_{class_label_safe}.png" # Use Pathlib
            plt.savefig(save_path, bbox_inches='tight'); plt.close()
            print(f"  Saved: {save_path}")
    elif shap_values_for_plotting is not None and hasattr(shap_values_for_plotting, 'ndim') and shap_values_for_plotting.ndim == 2:
        plt.figure()
        shap.summary_plot(shap_values_for_plotting, X_sample_df_for_plot, feature_names=feature_names_list_str, show=False, plot_type="dot")
        plt.title(f"SHAP Summary Plot - {model_type_str}")
        save_path = plot_dir_path / f"{model_type_str}_shap_summary.png" # Use Pathlib
        plt.savefig(save_path, bbox_inches='tight'); plt.close()
        print(f"  Saved: {save_path}")
    else:
        print(f"SHAP ({model_type_str}): shap_values_for_plotting not suitable for summary plot. Skipping.");
        if not (is_multi_class_for_loops and isinstance(shap_values_for_plotting, list)): return

    # ... (rest of dependence plot logic, ensuring to use plot_dir_path) ...
    # Example for dependence plot save path:
    # save_path = plot_dir_path / f"{model_type_str.replace(' ', '_').lower()}_shap_dependence_{feature_name_safe_filename}_{class_label_safe}.png"

    # (The rest of your dependence plot logic from the prompt should follow,
    #  making sure to use plot_dir_path for saving figures.)

    mean_abs_shap_per_feature = None
    if is_multi_class_for_loops and isinstance(shap_values_for_plotting, list):
        abs_shap_mean_across_classes = np.mean([np.abs(s) for s in shap_values_for_plotting], axis=0)
        if abs_shap_mean_across_classes.ndim == 2: mean_abs_shap_per_feature = np.mean(abs_shap_mean_across_classes, axis=0)
    elif shap_values_for_plotting is not None and hasattr(shap_values_for_plotting, 'ndim') and shap_values_for_plotting.ndim == 2:
        mean_abs_shap_per_feature = np.abs(shap_values_for_plotting).mean(0)

    if mean_abs_shap_per_feature is None:
        print(f"SHAP ({model_type_str}) Warning: Could not calculate mean_abs_shap_per_feature for dependence plots. Skipping.")
        return

    if len(mean_abs_shap_per_feature) != len(feature_names_list_str):
        print(f"SHAP ({model_type_str}) Warning: Mismatch in length of mean_abs_shap ({len(mean_abs_shap_per_feature)}) and feature_names ({len(feature_names_list_str)}). Skipping dependence plots.")
    else:
        top_feature_indices = np.argsort(mean_abs_shap_per_feature)[::-1][:min(3, len(feature_names_list_str))]
        for feature_idx_int in top_feature_indices:
            feature_name_as_string = feature_names_list_str[feature_idx_int]
            feature_name_safe_filename = feature_name_as_string.replace('/', '_').replace(':', '_').lower()

            if is_multi_class_for_loops and isinstance(shap_values_for_plotting, list):
                for i in range(len(shap_values_for_plotting)):
                    plt.figure()
                    class_label_safe = (class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}").replace(' ', '_').lower()
                    try:
                        shap.dependence_plot(feature_name_as_string, shap_values_for_plotting[i], X_sample_df_for_plot, show=False, interaction_index=None)
                        plt.title(f"SHAP Dependence: {feature_name_as_string} ({class_label_safe}) - {model_type_str}")
                        save_path = plot_dir_path / f"{model_type_str}_shap_dependence_{feature_name_safe_filename}_{class_label_safe}.png"
                        plt.savefig(save_path, bbox_inches='tight'); plt.close()
                        print(f"  Saved: {save_path}")
                    except Exception as e_dep: print(f"  SHAP ({model_type_str}): Error in dependence plot for {feature_name_as_string}, class {class_label_safe}: {e_dep}"); plt.close()
            elif shap_values_for_plotting is not None and hasattr(shap_values_for_plotting, 'ndim') and shap_values_for_plotting.ndim == 2:
                plt.figure()
                try:
                    shap.dependence_plot(feature_name_as_string, shap_values_for_plotting, X_sample_df_for_plot, show=False, interaction_index=None)
                    plt.title(f"SHAP Dependence Plot: {feature_name_as_string} - {model_type_str}")
                    save_path = plot_dir_path / f"{model_type_str}_shap_dependence_{feature_name_safe_filename}.png"
                    plt.savefig(save_path, bbox_inches='tight'); plt.close()
                    print(f"  Saved: {save_path}")
                except Exception as e_dep: print(f"  SHAP ({model_type_str}): Error in dependence plot for {feature_name_as_string}: {e_dep}"); plt.close()
        print(f"SHAP dependence plots generated for top features of {model_type_str}.")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def plot_roc_curves(y_true, y_pred_probas_dict, class_names_list, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_pred_proba in y_pred_probas_dict.items():
        if num_classes == 2:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                 y_scores = y_pred_proba[:, 1]
            elif y_pred_proba.ndim == 1:
                 y_scores = y_pred_proba
            else:
                print(f"Error: y_pred_proba for binary ROC for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != num_classes:
                print(f"Error: y_pred_proba for multi-class ROC for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linestyle='--', label=f'{model_name} - Class {class_names_list[i]} (AUC = {roc_auc:.3f})')
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            ax.plot(fpr_micro, tpr_micro, label=f'{model_name} - Micro Avg (AUC = {roc_auc_micro:.3f})', color='deeppink', linestyle=':', linewidth=3)

    ax.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)') # T('Chance Level (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate') # T('False Positive Rate')
    ax.set_ylabel('True Positive Rate') # T('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves') # T('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_precision_recall_curves(y_true, y_pred_probas_dict, class_names_list, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_pred_proba in y_pred_probas_dict.items():
        if num_classes == 2:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                 y_scores = y_pred_proba[:, 1]
            elif y_pred_proba.ndim == 1:
                 y_scores = y_pred_proba
            else:
                print(f"Error: y_pred_proba for binary PR for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ax.plot(recall, precision, label=f'{model_name}')
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != num_classes:
                print(f"Error: y_pred_proba for multi-class PR for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                ax.plot(recall, precision, linestyle='--', label=f'{model_name} - Class {class_names_list[i]}')
            # Micro-average PR curve
            # precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            # ax.plot(recall_micro, precision_micro, label=f'{model_name} - Micro Avg', color='deeppink', linestyle=':', linewidth=3)


    ax.set_xlabel('Recall') # T('Recall')
    ax.set_ylabel('Precision') # T('Precision')
    ax.set_title('Precision-Recall Curves') # T('Precision-Recall Curves')
    ax.legend(loc="best") # Consider loc="lower left" or "center left" for PR curves
    ax.grid(True)
    plt.tight_layout()
    return fig