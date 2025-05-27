import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

from core.data_handler import load_and_prepare_data
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
import torch # Add torch import

from skbio.stats.composition import clr 
from sklearn.preprocessing import StandardScaler

import joblib
import json

OUTPUT_PLOT_DIR = "output_plots"
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

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
        print(f"Error saving figure {filename_prefix}: {e}")
        plt.close(fig) # Still try to close


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
    prepared_data = load_and_prepare_data(
        actual_data_file_path, 
        TARGET_COLUMN, 
        test_size=0.25, 
        random_state=RANDOM_STATE,
        apply_outlier_capping_main=True, # Enable outlier capping
        cols_for_log_transform_main=None # No other specific cols for log transform for now
    )
    
    if prepared_data is None:
        print("Pipeline halted due to data loading/preparation errors.")
        return
        
    (X_train, X_test, y_train, y_test, 
     feature_names, class_names, num_classes, 
     label_encoder, scaler, X_processed_for_eda, df_original_for_ratio_plots) = prepared_data

    # --- 2. Exploratory Data Analysis ---
    print("\n--- Stage 2: Exploratory Data Analysis ---")
    
    # For EDA plots that need CLR-transformed data (scatter matrix, correlation, PCA):
    # X_processed_for_eda is X after CLR, log (if any), and OHE, but before scaling.
    # Add the original target label back for hue.
    if TARGET_COLUMN in df_original_for_ratio_plots.columns:
        # Use original target for hue where possible for better interpretability
        X_processed_for_eda['EDA_Target_Hue'] = df_original_for_ratio_plots[TARGET_COLUMN].loc[X_processed_for_eda.index]
    else: # Fallback
        # This alignment is tricky if indices don't match perfectly after processing
        # It's safer to use the encoded y if direct mapping of original target is complex
        # For simplicity, assuming index alignment or using encoded y.
        # y_full_encoded = label_encoder.transform(df_original_for_ratio_plots[TARGET_COLUMN])
        # X_processed_for_eda['EDA_Target_Hue_Encoded'] = y_full_encoded # if using encoded
        pass


    # Key elements for scatter matrix should now be the transformed names (_clr, _log)
    key_elements_for_scatter_clr = [col for col in X_processed_for_eda.columns if '_clr' in col][:10] # Focus on CLR
    if not key_elements_for_scatter_clr and feature_names:
         key_elements_for_scatter_clr = [col for col in feature_names if '_log' in col or 'Deposit' in col][:10]


    eda_scatter_fig = plot_pair_scatter_matrix(X_processed_for_eda, 
                                               key_elements=key_elements_for_scatter_clr, 
                                               target_column_name='EDA_Target_Hue' if 'EDA_Target_Hue' in X_processed_for_eda else None)
    save_figure(eda_scatter_fig, "eda_pair_scatter_matrix_after_clr")

    eda_corr_fig = plot_correlation_heatmap(X_processed_for_eda.select_dtypes(include=np.number), method='pearson')
    save_figure(eda_corr_fig, "eda_correlation_heatmap_after_clr")

    # For PCA: Use X_processed_for_eda (CLR/log/OHE, but unscaled), then scale it for PCA.
    X_pca_candidate_df = X_processed_for_eda.select_dtypes(include=np.number)
    if not X_pca_candidate_df.empty:
        pca_feature_names_viz = X_pca_candidate_df.columns.tolist()
        X_pca_input_scaled = StandardScaler().fit_transform(X_pca_candidate_df)
        
        # Get y_labels corresponding to X_processed_for_eda (full dataset before split)
        y_full_encoded_for_pca = label_encoder.transform(df_original_for_ratio_plots.loc[X_processed_for_eda.index, TARGET_COLUMN])

        pca_fig = plot_pca_biplot(X_pca_input_scaled, y_full_encoded_for_pca, class_names, pca_feature_names_viz)
        save_figure(pca_fig, "eda_pca_biplot_on_clr_data")
    else:
        print("Skipping PCA plot as no numeric features were found in X_processed_for_eda.")

    # Ratio plots use df_original_for_ratio_plots
    k2o_ratio_fig = plot_k2o_na2o_vs_sio2(df_original_for_ratio_plots, GEOCHEM_K2O_COL, GEOCHEM_NA2O_COL, GEOCHEM_SIO2_COL, target_col=TARGET_COLUMN)
    save_figure(k2o_ratio_fig, "eda_k2o_na2o_vs_sio2_ratio")

    # --- 3. Train and Evaluate Models ---
    # (Model training and evaluation section remains largely the same as previous version)
    print("\n--- Stage 3: Model Training and Evaluation ---")
    models_predictions = {} 

    print("\n-- Training Random Forest --")
    rf_model, rf_importances = train_random_forest(
        X_train.copy(), y_train.copy(), feature_names, 
        random_state=RANDOM_STATE, plot_curves=True
    )
    if rf_model:
        y_pred_rf = rf_model.predict(X_test)
        y_pred_proba_rf = rf_model.predict_proba(X_test)
        models_predictions['Random Forest'] = y_pred_proba_rf
        print("\nRandom Forest - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_rf, target_names=class_names, zero_division=0))
        plot_confusion_matrix_heatmap(y_test, y_pred_rf, class_names, "Random Forest")
        rf_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_rf, class_names, "Random Forest")
        save_figure(rf_cm_fig, "rf_confusion_matrix")

        if rf_importances is not None and rf_importances.any():
            rf_fi_fig = plot_feature_importances(rf_importances, feature_names, "Random Forest")
            save_figure(rf_fi_fig, "rf_feature_importances")

    print("\n-- Training SVM --")
    svm_model, svm_importances = train_svm(
        X_train.copy(), y_train.copy(), feature_names, random_state=RANDOM_STATE
    )
    if svm_model:
        y_pred_svm = svm_model.predict(X_test)
        y_pred_proba_svm = svm_model.predict_proba(X_test)
        models_predictions['SVM'] = y_pred_proba_svm
        print("\nSVM - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_svm, target_names=class_names, zero_division=0))
        plot_confusion_matrix_heatmap(y_test, y_pred_svm, class_names, "SVM")
        svm_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_svm, class_names, "SVM")
        save_figure(svm_cm_fig, "svm_confusion_matrix")
        # SVM feature importances are typically not directly available, but if using permutation importance:
        if hasattr(svm_model, 'feature_importances_'):
            svm_importances = svm_model.feature_importances_
        else:
            svm_importances = None
        if svm_importances is not None and svm_importances.any():
            svm_fi_fig = plot_feature_importances(svm_importances, feature_names, "SVM (Permutation)")
            save_figure(svm_fi_fig, "svm_feature_importances")  

    # Model 3: Deep Neural Network (PyTorch)
    print("\n-- Training DNN (PyTorch with Optuna) --")
    if len(X_train) > 50 and len(np.unique(y_train)) > 1:
        X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
    else: 
        X_train_dnn, y_train_dnn = X_train, y_train
        X_val_dnn, y_val_dnn = X_test, y_test 
        print("Warning: Using test set as validation for DNN due to small training/stratification constraints.")

    input_dim_dnn = X_train_dnn.shape[1]

    # Ensure data is numpy before passing to PyTorch function if it's pandas
    if isinstance(X_train_dnn, pd.DataFrame): X_train_dnn_np = X_train_dnn.values
    else: X_train_dnn_np = X_train_dnn
    if isinstance(y_train_dnn, pd.Series): y_train_dnn_np = y_train_dnn.values
    else: y_train_dnn_np = y_train_dnn
    if isinstance(X_val_dnn, pd.DataFrame): X_val_dnn_np = X_val_dnn.values
    else: X_val_dnn_np = X_val_dnn
    if isinstance(y_val_dnn, pd.Series): y_val_dnn_np = y_val_dnn.values
    else: y_val_dnn_np = y_val_dnn

    dnn_model_pytorch = train_pytorch_dnn(
        X_train_dnn_np, y_train_dnn_np, X_val_dnn_np, y_val_dnn_np,
        input_dim=input_dim_dnn, 
        num_classes=num_classes,
        epochs=30, # Epochs for Optuna trials and final training phase
        n_optuna_trials=15, # Number of Optuna trials (adjust based on time)
        project_name=f'{TARGET_COLUMN.lower().replace(" ", "_").replace("-","_")}_pytorch_dnn_tuning'
    )

    if dnn_model_pytorch:
        dnn_model_pytorch.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32).to(DEVICE)
            outputs_test = dnn_model_pytorch(X_test_tensor)
            
            if num_classes == 2:
                # BCEWithLogitsLoss means raw logits, apply sigmoid for probabilities/predictions
                y_pred_proba_dnn_raw = torch.sigmoid(outputs_test).cpu().numpy().flatten()
                y_pred_dnn = (y_pred_proba_dnn_raw > 0.5).astype(int)
                models_predictions['DNN (PyTorch)'] = y_pred_proba_dnn_raw # Probabilities for positive class
            else:
                # CrossEntropyLoss means raw logits, apply softmax for probabilities
                y_pred_proba_dnn_raw = torch.softmax(outputs_test, dim=1).cpu().numpy()
                y_pred_dnn = np.argmax(y_pred_proba_dnn_raw, axis=1)
                models_predictions['DNN (PyTorch)'] = y_pred_proba_dnn_raw # All class probabilities
            
        print("\nDNN (PyTorch) - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_dnn, target_names=class_names, zero_division=0))
        dnn_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_dnn, class_names, "DNN (PyTorch)")
        save_figure(dnn_cm_fig, "dnn_pytorch_confusion_matrix")
        print("DNN Feature importances (e.g., SHAP, Captum) are typically more complex for PyTorch and not plotted by default.")


    # Model 4: XGBoost (New)
    print("\n-- Training XGBoost --")
    xgb_model, xgb_importances = train_xgboost(
        X_train.copy(), y_train.copy(), num_classes=num_classes, feature_names=feature_names,
        random_state=RANDOM_STATE
    )
    if xgb_model:
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)
        models_predictions['XGBoost'] = y_pred_proba_xgb # Add to combined plots
        
        print("\nXGBoost - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_xgb, target_names=class_names, zero_division=0))
        
        xgb_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_xgb, class_names, "XGBoost")
        save_figure(xgb_cm_fig, "xgb_confusion_matrix")
        
        if xgb_importances is not None and xgb_importances.any():
            xgb_fi_fig = plot_feature_importances(xgb_importances, feature_names, "XGBoost")
            save_figure(xgb_fi_fig, "xgb_feature_importances")

    if models_predictions:
        print("\n--- Plotting Combined ROC and Precision-Recall Curves ---")
        roc_fig = plot_roc_curves(y_test, models_predictions, class_names, num_classes)
        save_figure(roc_fig, "combined_roc_curves")
        
        pr_fig = plot_precision_recall_curves(y_test, models_predictions, class_names, num_classes)
        save_figure(pr_fig, "combined_precision_recall_curves")
            
    print("\n--- Training and Evaluation Pipeline Complete ---")

# --- Prepare background data for SHAP DeepExplainer (sample from X_train) ---
    X_train_background_sample_df = None
    if isinstance(X_train, pd.DataFrame) and not X_train.empty:
        num_background_samples = min(100, len(X_train)) # Use up to 100 samples from training set for background
        X_train_background_sample_df = X_train.sample(num_background_samples, random_state=RANDOM_STATE)
    elif isinstance(X_train, np.ndarray) and X_train.shape[0] > 0 : # If X_train is numpy
        num_background_samples = min(100, X_train.shape[0])
        indices = np.random.choice(X_train.shape[0], num_background_samples, replace=False)
        X_train_background_sample_df = pd.DataFrame(X_train[indices], columns=feature_names)


    # --- 5. SHAP Value Analysis (after all models are trained) ---
    print("\n--- Stage 5: SHAP Value Analysis ---")
    
    if X_test.empty:
        print("X_test is empty, skipping SHAP plot generation.")
    else:
        sample_size_shap = min(100, len(X_test)) 
        if sample_size_shap > 0:
            if not isinstance(X_test, pd.DataFrame): 
                 X_test_df_for_shap_call = pd.DataFrame(X_test, columns=feature_names)
            else:
                 X_test_df_for_shap_call = X_test
            X_test_sample_shap = X_test_df_for_shap_call.sample(sample_size_shap, random_state=RANDOM_STATE)

            # Pass X_train_background_sample_df to relevant explainers
            if 'rf_model' in locals() and rf_model:
                generate_and_save_shap_plots(rf_model, X_test_sample_shap, feature_names, 
                                             "Random Forest", num_classes, class_names_list=class_names,
                                             X_background_df_orig=X_train_background_sample_df) # TreeExplainer can also use background
            if 'xgb_model' in locals() and xgb_model:
                generate_and_save_shap_plots(xgb_model, X_test_sample_shap, feature_names, 
                                             "XGBoost", num_classes, class_names_list=class_names,
                                             X_background_df_orig=X_train_background_sample_df)
            if 'svm_model' in locals() and svm_model:
                print("\nNote: SHAP for SVM (KernelExplainer) can be very slow.")
                generate_and_save_shap_plots(svm_model, X_test_sample_shap, feature_names, 
                                             "SVM", num_classes, class_names_list=class_names,
                                             X_background_df_orig=X_train_background_sample_df) # Pass background for KernelExplainer

            if 'dnn_model_pytorch' in locals() and dnn_model_pytorch:
                print(f"\nAttempting SHAP for PyTorch DNN (using device: {DEVICE})...") # DEVICE is from train_dnn_model.py
                if X_train_background_sample_df is not None:
                    generate_and_save_shap_plots(
                        dnn_model_pytorch, 
                        X_test_sample_shap, 
                        feature_names, 
                        "PyTorch DNN", 
                        num_classes, 
                        class_names_list=class_names,
                        X_background_df_orig=X_train_background_sample_df, # Pass training sample as background
                        device_for_dnn=DEVICE # Pass the torch device
                    )
                else:
                    print("Skipping PyTorch DNN SHAP: No background data (from X_train) available.")
        else:
            print("Not enough samples in X_test for SHAP analysis.")

    # --- Stage 4: Saving All Assets ---
    print("\n--- Stage 4: Saving Models and Preprocessing Assets ---")
    assets_dir = "trained_models"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # 1. Save the scaler
    if 'scaler' in locals() and scaler:
        scaler_path = os.path.join(assets_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
    else:
        print("Error: Scaler not found in pipeline locals. Cannot save.")

    # 2. Save the final feature names
    if 'feature_names' in locals() and feature_names:
        feature_names_path = os.path.join(assets_dir, "final_feature_names.json")
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"Final feature names saved to {feature_names_path}")
    else:
        print("Error: Final feature names not found in pipeline locals. Cannot save.")

    # 3. Save Label Encoder related mappings (or rely on CLASS_NAMES from data_handler)
    # CLASS_NAMES, LABEL_TO_INT_MAPPING, INT_TO_LABEL_MAPPING are usually sufficient if defined consistently.
    # If label_encoder object itself is needed (e.g. for inverse_transforming unseen labels, though unlikely for prediction GUI):
    if 'label_encoder' in locals() and label_encoder:
        label_encoder_path = os.path.join(assets_dir, "label_encoder.joblib")
        joblib.dump(label_encoder, label_encoder_path)
        print(f"Label encoder saved to {label_encoder_path}")

    # 4. Models (assuming individual train_*.py scripts save them, or save them here)
    # Ensure models are saved with standard names in 'trained_models/'
    # Example for models if not saved within their train functions:
    if 'rf_model' in locals() and rf_model:
        joblib.dump(rf_model, os.path.join(assets_dir, "rf_model.joblib"))
        print("Random Forest model explicitly saved by pipeline.")
    if 'svm_model' in locals() and svm_model:
        joblib.dump(svm_model, os.path.join(assets_dir, "svm_model.joblib"))
        print("SVM model explicitly saved by pipeline.")
    if 'xgb_model' in locals() and xgb_model:
        joblib.dump(xgb_model, os.path.join(assets_dir, "xgb_model.joblib"))
        print("XGBoost model explicitly saved by pipeline.")
    # PyTorch DNN model is saved within train_pytorch_dnn function.

    # SHAP plots and other EDA plots are already saved by their respective functions.
    # Ensure output_plots directory structure is as GUI expects for model_insights.

    print("\n--- All essential assets for GUI should now be saved in 'trained_models/' ---")
    print("--- EDA and model-specific plots should be in 'output_plots/' ---")


if __name__ == '__main__':
    print("Starting the Porphyry Rock Classification Training Pipeline...")
    run_training_pipeline()
