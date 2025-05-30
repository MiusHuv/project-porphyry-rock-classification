from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import torch # Add torch import

from skbio.stats.composition import clr 
from sklearn.preprocessing import StandardScaler

import joblib
import json

from core.data_handler import load_and_prepare_data, RAW_EXPECTED_GEOCHEMICAL_FEATURES
from core.visualizer import (
    plot_feature_importances, 
    plot_confusion_matrix_heatmap, 
    plot_roc_curves, 
    plot_precision_recall_curves,
    generate_and_save_shap_plots
)
from util.eda_plot import (
    plot_pair_scatter_matrix,
    plot_correlation_heatmap,
    plot_pca_biplot,
    plot_k2o_na2o_vs_sio2,
    plot_sr_y_vs_y
)
from core.train_tree_model import train_random_forest
from core.train_svm_model import train_svm
from core.train_dnn_model import train_pytorch_dnn, SimpleDNN, DEVICE # Import SimpleDNN and DEVICE for prediction
from core.train_xgboost_model import train_xgboost
from core.data_handler import CLASS_NAMES, LABEL_TO_INT_MAPPING, INT_TO_LABEL_MAPPING


OUTPUT_PLOT_DIR = "output_plots"
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

MODELS_DIR = "models"
ASSETS_DIR = "assets"
EDA_PLOTS_DIR = os.path.join(ASSETS_DIR, "eda_plots")
FEATIMPORT_PLOTS_DIR = os.path.join(ASSETS_DIR, "importance_plots") # For RF/XGBoost feature importance plots
SHAP_PLOTS_DIR = os.path.join(ASSETS_DIR, "shap_plots") # For GUI consistency
MODEL_SPECIFIC_PLOTS_DIR = os.path.join(EDA_PLOTS_DIR, "model_specific") # For RF/XGBoost n_estimators plots

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_SPECIFIC_PLOTS_DIR, exist_ok=True)

# Function to save figures
def save_figure(fig, filename_prefix, plot_dir=OUTPUT_PLOT_DIR):
    if fig is None:
        return
    try:
        filepath = os.path.join(plot_dir, f"{filename_prefix.replace(' ', '_').lower()}.png")
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")
        plt.close(fig) # Important to close the figure to free memory
    except Exception as e:
        print(f"Error saving figure {filepath}: {e}")
        if fig: plt.close(fig)


def create_dummy_data_if_not_exists(filepath, target_col_name, all_expected_cols):
    """Creates a dummy CSV if the specified filepath doesn't exist, including expected columns."""
    if not os.path.exists(filepath):
        print(f"WARNING: Data file '{filepath}' not found.")
        print(f"Creating a dummy CSV for demonstration: 'data/dummy_data.csv' with columns based on image.")
        
        data_dir = os.path.dirname(filepath)
        if not os.path.exists(data_dir) and data_dir:
            os.makedirs(data_dir)
        
        num_samples = 150
        
        # Create DataFrame with all expected columns
        dummy_df_data = {}
        # Add Label and Deposit
        possible_labels = ['Cu-rich PCDs', 'Barren PCDs', 'Intermediate PCDs'] # Example labels
        dummy_df_data[target_col_name] = np.random.choice(possible_labels, size=num_samples)
        if 'Deposit' in all_expected_cols:
            possible_deposits = ['Aktogai Cu', 'Deposit B', 'Deposit C'] # Example deposits
            dummy_df_data['Deposit'] = np.random.choice(possible_deposits, size=num_samples)
        
        # Add numerical features (oxides and trace elements)
        numerical_cols_from_image = [
            'SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Th', 'U'
        ]
        for col in numerical_cols_from_image:
            if col in all_expected_cols:
                # Simulate some plausible data ranges
                if 'SiO2' in col: dummy_df_data[col] = np.random.uniform(40, 80, num_samples)
                elif 'O' in col : dummy_df_data[col] = np.random.uniform(0, 20, num_samples) # Other oxides
                else: dummy_df_data[col] = np.random.uniform(0, 1000, num_samples) # Trace elements
        
        dummy_df = pd.DataFrame(dummy_df_data)

        # Ensure all expected columns are present, fill any missing with 0 for simplicity in dummy
        for col in all_expected_cols:
            if col not in dummy_df.columns and col != target_col_name : # target is already there
                 dummy_df[col] = 0


        # Add some NaNs to test imputation
        for col in dummy_df.columns:
            if col != target_col_name and np.random.rand() > 0.8: # 20% chance to add NaNs
                 dummy_df.loc[dummy_df.sample(frac=0.1).index, col] = np.nan
        
        actual_dummy_filepath = os.path.join(data_dir, "dummy_data.csv")
        dummy_df.to_csv(actual_dummy_filepath, index=False)
        print(f"Created '{actual_dummy_filepath}'.")
        print(f"IMPORTANT: For actual analysis, replace this with your real data file at '{filepath}' "
              f"and ensure TARGET_COLUMN is correctly set to '{target_col_name}'.")
        return actual_dummy_filepath
    return filepath


def run_training_pipeline():
    # --- Configuration ---
    TARGET_COLUMN = "Label"  # Updated based on image
    DATA_FILE_PATH = "data/2025-Project-Data(ESM Table 1).csv"

    # Define all expected columns based on the image for dummy data generation and EDA
    # (Order might not be identical to image but names should match)
    ALL_EXPECTED_COLS_FROM_IMAGE = [
        'Label', 'Deposit', 'SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 
        'Na2O', 'K2O', 'P2O5', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 
        'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
        'Hf', 'Ta', 'Th', 'U'
    ]

    # For Geochemical Ratio Plots - Column names from your CSV
    GEOCHEM_K2O_COL = "K2O" 
    GEOCHEM_NA2O_COL = "Na2O"
    GEOCHEM_SIO2_COL = "SiO2"
    GEOCHEM_SR_COL = "Sr"  
    GEOCHEM_Y_COL = "Y"   

    RANDOM_STATE = 42
    
    actual_data_file_path = create_dummy_data_if_not_exists(DATA_FILE_PATH, TARGET_COLUMN, ALL_EXPECTED_COLS_FROM_IMAGE)
    if "dummy_data.csv" in actual_data_file_path:
        print("\nWARNING: USING DUMMY DATA. RESULTS WILL NOT BE MEANINGFUL FOR YOUR PROJECT.\n")
        # If dummy data is used, ensure geochem plot cols exist, if not, they will be skipped by plot functions
        # No need to change GEOCHEM_..._COL here, as plot functions check for column existence.

    # --- 1. Load and Preprocess Data ---
    print("\n--- Stage 1: Data Loading and Preprocessing ---")
    prepared_data_gui = load_and_prepare_data(
        actual_data_file_path,
        TARGET_COLUMN,
        test_size=0.25,
        random_state=RANDOM_STATE,
        apply_outlier_capping_main=True,
        cols_for_log_transform_main=None, # Specify if any non-CLR numeric cols need log
        gui_model_preparation=True # THIS IS KEY: ensures 'Deposit' is dropped
    )
    
    if prepared_data_gui is None:
        print("Pipeline halted: Data preparation for GUI models failed.")
        return
        
    (X_train, X_test, y_train, y_test,
     final_feature_names_for_model, class_names_from_encoder, num_classes,
     label_encoder_obj, scaler_obj,
     X_processed_for_eda_plots, df_original_for_ratio_plots) = prepared_data_gui

    # --- Save Preprocessing Artifacts for GUI ---
    print("\n--- Saving Preprocessing Artifacts for GUI ---")
    joblib.dump(scaler_obj, os.path.join(MODELS_DIR, "scaler.joblib"))
    print(f"Scaler saved to {os.path.join(MODELS_DIR, 'scaler.joblib')}")

    with open(os.path.join(MODELS_DIR, "label_encoder_classes.json"), 'w') as f:
        json.dump(class_names_from_encoder, f)
    print(f"Label encoder classes saved to {os.path.join(MODELS_DIR, 'label_encoder_classes.json')}: {class_names_from_encoder}")

    with open(os.path.join(MODELS_DIR, "final_feature_names.json"), 'w') as f:
        json.dump(final_feature_names_for_model, f)
    print(f"Final feature names for model input saved to {os.path.join(MODELS_DIR, 'final_feature_names.json')}")

    # --- 2. Exploratory Data Analysis (Using data processed appropriately for EDA) ---
    print("\n--- Stage 2: Exploratory Data Analysis ---")
    # X_processed_for_eda_plots is from load_and_prepare_data,
    # which is X after CLR/log but before OHE (if gui_model_preparation=True, categoricals like Deposit are already dropped)
    # and before train/test split or scaling. Add target for hue.
    if TARGET_COLUMN in df_original_for_ratio_plots.columns:
         # Use original target for hue, aligning indices carefully
        temp_eda_df = X_processed_for_eda_plots.copy()
        # Ensure index alignment for adding target column
        temp_eda_df['EDA_Target_Hue'] = df_original_for_ratio_plots.loc[temp_eda_df.index, TARGET_COLUMN]

    else:
        temp_eda_df = X_processed_for_eda_plots.copy()
        # If target column isn't in original df (shouldn't happen with create_dummy), handle gracefully
        print("Warning: Target column not found in df_original_for_ratio_plots for EDA hue.")


    # Scatter Matrix: focus on CLR transformed features if available, or other key numeric features
    key_elements_for_scatter = [col for col in temp_eda_df.columns if '_clr' in col][:10]
    if not key_elements_for_scatter: # Fallback if no CLR features (e.g. skbio failed or not used)
        key_elements_for_scatter = [col for col in temp_eda_df.select_dtypes(include=np.number).columns if col != 'EDA_Target_Hue'][:10]

    if key_elements_for_scatter:
        eda_scatter_fig = plot_pair_scatter_matrix(temp_eda_df,
                                                   key_elements=key_elements_for_scatter,
                                                   target_column_name='EDA_Target_Hue' if 'EDA_Target_Hue' in temp_eda_df else None)
        save_figure(eda_scatter_fig, "eda_pair_scatter_matrix_after_transforms.png", EDA_PLOTS_DIR)
    else:
        print("Skipping EDA pair scatter matrix due to lack of suitable key elements.")

    # Correlation Heatmap
    eda_corr_fig = plot_correlation_heatmap(temp_eda_df.drop(columns=['EDA_Target_Hue'], errors='ignore').select_dtypes(include=np.number))
    save_figure(eda_corr_fig, "eda_correlation_heatmap_after_transforms.png", EDA_PLOTS_DIR)

    # PCA Biplot: Use the same X_processed_for_eda_plots but scale it specifically for PCA
    X_pca_candidate_df = temp_eda_df.drop(columns=['EDA_Target_Hue'], errors='ignore').select_dtypes(include=np.number)
    if not X_pca_candidate_df.empty and X_pca_candidate_df.shape[1] >=2:
        pca_feature_names_for_plot = X_pca_candidate_df.columns.tolist()
        X_pca_input_scaled = StandardScaler().fit_transform(X_pca_candidate_df) # Scale for PCA

        # Get y_labels corresponding to X_processed_for_eda (full dataset before split)
        # This requires careful index alignment.
        y_full_encoded_for_pca = label_encoder_obj.transform(df_original_for_ratio_plots.loc[X_pca_candidate_df.index, TARGET_COLUMN])

        pca_fig = plot_pca_biplot(X_pca_input_scaled, y_full_encoded_for_pca, class_names_from_encoder, pca_feature_names_for_plot)
        save_figure(pca_fig, "eda_pca_biplot_after_transforms.png", EDA_PLOTS_DIR)
    else:
        print("Skipping PCA plot as not enough numeric features were found or data is empty.")

    # Ratio plots use df_original_for_ratio_plots (which has minimal imputation for these specific columns if needed)
    k2o_ratio_fig = plot_k2o_na2o_vs_sio2(df_original_for_ratio_plots, GEOCHEM_K2O_COL, GEOCHEM_NA2O_COL, GEOCHEM_SIO2_COL, target_col=TARGET_COLUMN)
    save_figure(k2o_ratio_fig, "eda_k2o_na2o_vs_sio2_ratio.png", EDA_PLOTS_DIR)

    sr_y_ratio_fig = plot_sr_y_vs_y(df_original_for_ratio_plots, GEOCHEM_SR_COL, GEOCHEM_Y_COL, target_col=TARGET_COLUMN)
    save_figure(sr_y_ratio_fig, "eda_sr_y_vs_y_ratio.png", EDA_PLOTS_DIR)


     # --- 3. Train and Evaluate Models ---
    print("\n--- Stage 3: Model Training and Evaluation ---")
    models_predictions_proba = {}
    trained_models = {} # To store trained model objects for SHAP

    print("\n-- Training Random Forest --")
    # Pass MODEL_SPECIFIC_PLOTS_DIR to train_random_forest if it saves plots internally
    rf_model, rf_importances = train_random_forest(
        X_train.copy(), y_train.copy(), final_feature_names_for_model, # Use names model was trained on
        random_state=RANDOM_STATE, plot_curves=True,
        model_filename="random_forest_model.joblib", # Will be saved in "models/"
        plot_save_dir=MODEL_SPECIFIC_PLOTS_DIR # Pass the correct plot dir
    )
    if rf_model:
        trained_models['Random Forest'] = rf_model
        y_pred_rf = rf_model.predict(X_test)
        y_pred_proba_rf = rf_model.predict_proba(X_test)
        models_predictions_proba['Random Forest'] = y_pred_proba_rf
        print("\nRandom Forest - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_rf, target_names=class_names_from_encoder, zero_division=0))
        rf_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_rf, class_names_from_encoder, "Random Forest")
        save_figure(rf_cm_fig, "rf_confusion_matrix.png", EDA_PLOTS_DIR)
        if rf_importances is not None and rf_importances.any():
            rf_fi_fig = plot_feature_importances(rf_importances, final_feature_names_for_model, "Random Forest")
            save_figure(rf_fi_fig, "random_forest_feature_importances", FEATIMPORT_PLOTS_DIR)

    print("\n-- Training SVM --")
    svm_model, svm_importances = train_svm(
        X_train.copy(), y_train.copy(), final_feature_names_for_model, random_state=RANDOM_STATE,
        model_filename="svm_model.joblib" # Will be saved in "models/"
    )
    if svm_model:
        trained_models['SVM'] = svm_model
        y_pred_svm = svm_model.predict(X_test)
        y_pred_proba_svm = svm_model.predict_proba(X_test)
        models_predictions_proba['SVM'] = y_pred_proba_svm
        print("\nSVM - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_svm, target_names=class_names_from_encoder, zero_division=0))
        svm_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_svm, class_names_from_encoder, "SVM")
        save_figure(svm_cm_fig, "svm_confusion_matrix.png", EDA_PLOTS_DIR)
        if svm_importances is not None and svm_importances.any(): # svm_importances is permutation importance
            svm_fi_fig = plot_feature_importances(svm_importances, final_feature_names_for_model, "SVM (Permutation)")
            save_figure(svm_fi_fig, "svm_feature_importances", FEATIMPORT_PLOTS_DIR)


    print("\n-- Training PyTorch DNN --")
    # Split training data further for DNN validation if needed (already done if test_size > 0 in main split)
    # X_train already scaled. y_train is encoded.
    # The train_pytorch_dnn function might do its own further split for optuna validation
    input_dim_dnn = X_train.shape[1]
    dnn_epochs = 50 # As per your current code, adjust if needed
    dnn_optuna_trials = 15 # As per your current code

    # Create a dedicated validation set for Optuna and early stopping from the training set
    # Ensure X_train_dnn_main and X_val_dnn are numpy arrays for PyTorch function
    if len(X_train) > 50 and len(np.unique(y_train)) > 1 :
        # Stratify if possible
        stratify_dnn_val = y_train if len(np.unique(y_train)) > 1 and len(np.unique(y_train)) < len(y_train) else None
        X_train_dnn_main_df, X_val_dnn_df, y_train_dnn_main_series, y_val_dnn_series = train_test_split(
            X_train, pd.Series(y_train, index=X_train.index), test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_dnn_val
        )
        X_train_dnn_main = X_train_dnn_main_df.values
        y_train_dnn_main = y_train_dnn_main_series.values
        X_val_dnn = X_val_dnn_df.values
        y_val_dnn = y_val_dnn_series.values
    else: # Fallback for very small datasets
        print("Warning: Small training set, using full X_train for DNN training and X_test for DNN validation (less ideal).")
        X_train_dnn_main = X_train.values
        y_train_dnn_main = y_train
        X_val_dnn = X_test.values # Using test set for validation here is not ideal but a fallback
        y_val_dnn = y_test


    dnn_model_pytorch_trained, dnn_best_params = train_pytorch_dnn( # Modified to return best_params too
        X_train_dnn_main, y_train_dnn_main, X_val_dnn, y_val_dnn,
        input_dim=input_dim_dnn,
        num_classes=num_classes, # ensure num_classes is correctly defined
        epochs=dnn_epochs,
        n_optuna_trials=dnn_optuna_trials,
        model_filename="pytorch_dnn_model.pth", # Saved in "models/" by train_pytorch_dnn
        model_save_dir=MODELS_DIR # Pass the correct save dir
    )

    if dnn_model_pytorch_trained:
        trained_models['PyTorch DNN'] = dnn_model_pytorch_trained # Store for SHAP
        # Save DNN config
        actual_hidden_configs = []
        actual_dropout_configs = []
        if not dnn_best_params:
            num_actual_layers = 2
            actual_hidden_configs = [128, 64]
            actual_dropout_configs = [0.3, 0.2]
            if not dnn_best_params: dnn_best_params = {} 
        else:
            num_actual_layers = dnn_best_params.get('n_layers', 2)
            for i in range(num_actual_layers):
                actual_hidden_configs.append(dnn_best_params.get(f'n_units_l{i}', 64))
            for i in range(num_actual_layers):
                actual_dropout_configs.append(dnn_best_params.get(f'dropout_l{i}', 0.2))
        

        dnn_config = {
            'input_dim': input_dim_dnn,
            'num_classes': num_classes,
            'hidden_layers_config': actual_hidden_configs,
            'dropout_rates': actual_dropout_configs,
            'model_filename': "pytorch_dnn_model.pth",
            'lr': dnn_best_params.get('lr'),
            'optimizer': dnn_best_params.get('optimizer'),
            'batch_size': dnn_best_params.get('batch_size')
        }
        with open(os.path.join(MODELS_DIR, "pytorch_dnn_model_config.json"), 'w') as f:
            json.dump(dnn_config, f, indent=4)
        print(f"PyTorch DNN config saved to {os.path.join(MODELS_DIR, 'pytorch_dnn_model_config.json')}")

        dnn_model_pytorch_trained.eval()
        with torch.no_grad():
            # Ensure X_test is a tensor for PyTorch
            X_test_tensor = torch.tensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32).to(DEVICE)
            outputs_test_dnn = dnn_model_pytorch_trained(X_test_tensor)

            if num_classes == 2: # Binary
                # BCEWithLogitsLoss means raw logits, apply sigmoid
                y_pred_proba_dnn_raw = torch.sigmoid(outputs_test_dnn).cpu().numpy()
                # Ensure it's 2D [:, 1] for consistency if it's (n_samples, 1)
                # If it's already (n_samples,) then it's prob of positive class
                if y_pred_proba_dnn_raw.ndim == 2 and y_pred_proba_dnn_raw.shape[1] == 1:
                    y_pred_proba_dnn = np.hstack((1 - y_pred_proba_dnn_raw, y_pred_proba_dnn_raw))
                elif y_pred_proba_dnn_raw.ndim == 1: # Assumed to be prob of positive class
                     y_pred_proba_dnn = np.vstack((1 - y_pred_proba_dnn_raw, y_pred_proba_dnn_raw)).T
                else: # Should not happen
                    print(f"Unexpected DNN output shape: {y_pred_proba_dnn_raw.shape}")
                    y_pred_proba_dnn = y_pred_proba_dnn_raw # Fallback

                y_pred_dnn = (y_pred_proba_dnn[:, 1] > 0.5).astype(int)
            else: # Multi-class
                y_pred_proba_dnn = torch.softmax(outputs_test_dnn, dim=1).cpu().numpy()
                y_pred_dnn = np.argmax(y_pred_proba_dnn, axis=1)

            models_predictions_proba['PyTorch DNN'] = y_pred_proba_dnn
            print("\nPyTorch DNN - Test Set Evaluation:")
            print(classification_report(y_test, y_pred_dnn, target_names=class_names_from_encoder, zero_division=0))
            dnn_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_dnn, class_names_from_encoder, "PyTorch DNN")
            save_figure(dnn_cm_fig, "dnn_pytorch_confusion_matrix.png", EDA_PLOTS_DIR)
    else:
        print("PyTorch DNN training failed or was skipped.")


    print("\n-- Training XGBoost --")
    xgb_model, xgb_importances = train_xgboost(
        X_train.copy(), y_train.copy(), num_classes=num_classes, feature_names=final_feature_names_for_model,
        random_state=RANDOM_STATE,
        model_filename="xgboost_model.joblib", # CHANGED to .joblib
        model_save_dir=MODELS_DIR, # Pass the main models directory
        plot_save_dir=MODEL_SPECIFIC_PLOTS_DIR
    )
    if xgb_model:
        trained_models['XGBoost'] = xgb_model
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)
        models_predictions_proba['XGBoost'] = y_pred_proba_xgb
        print("\nXGBoost - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_xgb, target_names=class_names_from_encoder, zero_division=0))
        
        xgb_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_xgb, class_names_from_encoder, "XGBoost")
        save_figure(xgb_cm_fig, "xgb_confusion_matrix.png", EDA_PLOTS_DIR)
        
        if xgb_importances is not None and xgb_importances.any():
            xgb_fi_fig = plot_feature_importances(xgb_importances, final_feature_names_for_model, "XGBoost")
            save_figure(xgb_fi_fig, "xgboost_feature_importances", FEATIMPORT_PLOTS_DIR)

# --- 4. Combined ROC and PR Curves ---
    if models_predictions_proba:
        print("\n--- Plotting Combined ROC and Precision-Recall Curves ---")
        # Ensure y_test is 1D array for roc/pr curve functions
        y_test_for_curves = np.array(y_test).ravel()

        roc_fig = plot_roc_curves(y_test_for_curves, models_predictions_proba, class_names_from_encoder, num_classes)
        save_figure(roc_fig, "combined_roc_curves.png", EDA_PLOTS_DIR)

        pr_fig = plot_precision_recall_curves(y_test_for_curves, models_predictions_proba, class_names_from_encoder, num_classes)
        save_figure(pr_fig, "combined_precision_recall_curves.png", EDA_PLOTS_DIR)

    # # # --- 5. SHAP Value Analysis (after all models are trained) ---
    # print("\n--- Stage 5: SHAP Value Analysis ---")
    
    if X_test.empty:
        print("X_test is empty, skipping SHAP plot generation.")
    else:
        sample_size_shap = min(100, len(X_test))
        top_shap_features_by_model = {}  # To store top SHAP features by model
        if sample_size_shap > 0:
            # X_test is already a DataFrame with final_feature_names_for_model
            X_test_sample_shap = X_test.sample(n=sample_size_shap, random_state=RANDOM_STATE)

            for model_key_name, model_instance in trained_models.items():
                if model_instance:
                    # Map display name to a filename-safe version
                    model_filename_key = model_key_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
                    print(f"\nGenerating SHAP plots for {model_key_name} (filename key: {model_filename_key})...")
                    generated_dependence_features  = generate_and_save_shap_plots(
                        model_instance,
                        X_test_sample_shap, # This is the scaled data
                        final_feature_names_for_model,
                        model_filename_key, # Use this for filename consistency
                        num_classes_model=num_classes,
                        class_names_list=class_names_from_encoder,
                        plot_dir=SHAP_PLOTS_DIR
                    )
                    if generated_dependence_features:
                        top_shap_features_by_model[model_key_name] = generated_dependence_features
                        print(f"SHAP plots for {model_key_name} saved successfully.")
                    else:
                        print(f"Warning: No SHAP plots generated for {model_key_name}.")
        else:
            print("Not enough samples in X_test for SHAP analysis.")
    if top_shap_features_by_model:
        top_features_save_path = Path(MODELS_DIR) / "top_shap_features_for_dependence_plots.json"
        try:
            with open(top_features_save_path, 'w') as f:
                json.dump(top_shap_features_by_model, f, indent=4)
            print(f"\nTop SHAP features for dependence plots saved to: {top_features_save_path}")
        except Exception as e:
            print(f"Error saving top SHAP features JSON: {e}")

    print("\n--- Training and Evaluation Pipeline Complete ---")


if __name__ == '__main__':
    print("Starting the Porphyry Rock Classification Training Pipeline...")
    run_training_pipeline()