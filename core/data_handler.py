import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
# Attempt to import CLR, if not available, CLR step will be skipped with a warning.
try:
    from skbio.stats.composition import clr
    SKBIO_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-bio (skbio) not installed. CLR transformation will be skipped. "
          "Install with: pip install scikit-bio")
    SKBIO_AVAILABLE = False

# --- Constants for Column Types (based on typical geochemical data) ---
# User should verify these lists match their actual CSV column names
# These are the columns that will undergo unit harmonization and then CLR
MAJOR_OXIDES_WT_PERCENT = ['SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']
TRACE_ELEMENTS_PPM = [
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
    'Hf', 'Ta', 'Th', 'U'
]
# Categorical columns that need specific handling (e.g., One-Hot Encoding)
CATEGORICAL_COLS_APP = ['Deposit']

RAW_EXPECTED_GEOCHEMICAL_FEATURES = [
    'SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'Th', 'U'
]
# RAW_EXPECTED_DEPOSIT_COLUMN = 'Deposit' # If 'Deposit' is consistently present in raw input
# RAW_EXPECTED_LABEL_COLUMN = 'Label'   # If 'Label' is consistently present in raw input for training

# Configuration for preprocessing steps (ensure these align with your actual column names)
MAJOR_OXIDES_WT_PERCENT = ['SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']
TRACE_ELEMENTS_PPM = [
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'Th', 'U'
]
# For GUI models, 'Deposit' will be dropped, so CATEGORICAL_COLS_APP for OHE will be empty or not include 'Deposit'.
# This list is used if keep_categorical=True in preprocess_features.
# For GUI model training, keep_categorical will be False.
CATEGORICAL_COLS_TO_POTENTIALLY_OHE = ['Deposit']

CLASS_NAMES = ['Au-rich', 'Cu-rich'] 
# Mapping for converting string labels to numeric, ensure consistency
LABEL_TO_INT_MAPPING = {label: i for i, label in enumerate(CLASS_NAMES)}
INT_TO_LABEL_MAPPING = {i: label for i, label in enumerate(CLASS_NAMES)}

def load_data_from_file(uploaded_file):
    """Loads data from an uploaded CSV or Excel file."""
    if uploaded_file is None:
        return None, "No file uploaded."
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Please upload CSV or Excel."
        return df, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

def validate_input_data(df, expected_features):
    """
    Validates the DataFrame against expected features.
    Does not perform one-hot encoding for 'Deposit' here.
    """
    if df is None:
        return None, "No data to validate."
    
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}. Expected {len(expected_features)} features."

    # Check for non-numeric data in expected feature columns
    # This should happen BEFORE imputation if imputation is zero-filling
    for col in expected_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Attempt to convert, and if it fails for any value, then it's problematic
            try:
                pd.to_numeric(df[col])
            except ValueError:
                 return None, f"Column '{col}' contains non-numeric data that cannot be converted. Please clean your data."
    
    return df[expected_features], None # Return only the expected features in order

def preprocess_data_for_prediction(raw_df, scaler, feature_names):
    """
    Preprocesses raw user-uploaded DataFrame for prediction.
    - Selects features
    - Handles missing values (e.g., zero imputation as per CLR context)
    - Applies the pre-trained scaler
    - Ensures 'Deposit' is NOT one-hot encoded if models are trained without it.
    """
    if raw_df is None or scaler is None or feature_names is None:
        return None, "Missing raw data, scaler, or feature names for preprocessing."

    # 1. Select only the features used for training
    try:
        processed_df = raw_df[feature_names].copy()
    except KeyError as e:
        missing_input_cols = list(set(feature_names) - set(raw_df.columns))
        return None, f"Input data is missing the following required columns: {missing_input_cols}. Error: {e}"

    # 2. Impute missing values (e.g., with zero, consistent with training for CLR-transformed data)
    # This step is crucial and must match how training data was handled *before* scaling.
    # If CLR was applied, zeros might be appropriate for missing original values.
    # If features are already CLR transformed, then missingness might mean something else.
    # Assuming feature_names are post-CLR, pre-scaling.
    processed_df.fillna(0, inplace=True) 

    # 3. Apply the pre-trained scaler
    try:
        scaled_values = scaler.transform(processed_df)
        scaled_df = pd.DataFrame(scaled_values, columns=feature_names, index=processed_df.index)
    except Exception as e:
        return None, f"Error applying scaler: {str(e)}. Ensure data has correct numeric types and no NaNs after imputation."
        
    # 4. Ensure 'Deposit' column is NOT one-hot encoded
    # This is implicitly handled if 'feature_names' does not include one-hot encoded 'Deposit' columns
    # and the models were trained accordingly.

    return scaled_df, None

def harmonize_units_to_ppm(df, oxide_cols_wt_percent, trace_cols_ppm):
    """
    Harmonizes units of specified columns to PPM.
    Oxides in wt% are multiplied by 10,000. Trace elements in ppm are kept as is.
    Returns a new DataFrame with harmonized columns and a list of all harmonized column names.
    """
    df_harmonized = df.copy()
    harmonized_column_names = []

    print("Harmonizing units to PPM...")
    # Convert major oxides from wt% to ppm
    for col in oxide_cols_wt_percent:
        if col in df_harmonized.columns:
            new_col_name = f"{col}_ppm"
            df_harmonized[new_col_name] = df_harmonized[col] * 10000
            harmonized_column_names.append(new_col_name)
            df_harmonized = df_harmonized.drop(columns=[col]) # Optionally drop original
            print(f"  Converted '{col}' (wt%) to '{new_col_name}' (ppm).")
        else:
            print(f"  Warning: Oxide column '{col}' not found for unit harmonization.")

    # Trace elements are already in ppm (or assumed to be)
    for col in trace_cols_ppm:
        if col in df_harmonized.columns:
            # If we want to ensure they have a _ppm suffix for consistency:
            # new_col_name = f"{col}_ppm" if not col.endswith("_ppm") else col
            # df_harmonized[new_col_name] = df_harmonized[col]
            # harmonized_column_names.append(new_col_name)
            # if new_col_name != col:
            #     df_harmonized = df_harmonized.drop(columns=[col])
            # For now, assume they are fine as is if already in ppm
            harmonized_column_names.append(col) # Add to the list of columns for CLR
            print(f"  Column '{col}' assumed to be in ppm, included for CLR.")
        else:
            print(f"  Warning: Trace element column '{col}' not found.")
            
    return df_harmonized, harmonized_column_names

def apply_clr_transformation(df, compositional_cols_ppm):
    """
    Applies Centered Log-Ratio (CLR) transformation to the specified compositional columns.
    Assumes columns are already harmonized to the same unit (e.g., ppm).
    Returns a new DataFrame with CLR transformed columns.
    """
    if not SKBIO_AVAILABLE:
        print("CLR transformation skipped: scikit-bio (skbio) not installed.")
        return df, [] # Return original df and empty list of clr_cols
    
    if not compositional_cols_ppm:
        print("CLR transformation skipped: No compositional columns provided.")
        return df, []

    df_clr_input = df[compositional_cols_ppm].copy()
    
    # Replace 0s and negative values with a small epsilon for CLR
    small_epsilon = 1e-9 # A very small number, but larger than machine epsilon
    num_zeros_neg = (df_clr_input <= 0).sum().sum()
    if num_zeros_neg > 0:
        print(f"CLR Info: Found {num_zeros_neg} zero or negative values in compositional columns. Replacing with {small_epsilon}.")
        df_clr_input[df_clr_input <= 0] = small_epsilon
    
    print(f"Applying CLR transformation to {len(compositional_cols_ppm)} columns: {compositional_cols_ppm}")
    try:
        clr_transformed_data = clr(df_clr_input.to_numpy()) # skbio's clr expects numpy array
        clr_col_names = [f"{col}_clr" for col in compositional_cols_ppm]
        df_clr_output = pd.DataFrame(clr_transformed_data, columns=clr_col_names, index=df.index)
        
        # Combine with non-compositional columns from original df
        df_remaining_cols = df.drop(columns=compositional_cols_ppm, errors='ignore')
        df_transformed = pd.concat([df_remaining_cols, df_clr_output], axis=1)
        print("CLR transformation applied successfully.")
        return df_transformed, clr_col_names
    except Exception as e:
        print(f"ERROR during CLR transformation: {e}. Returning original data for these columns.")
        return df, [] # Return original df and empty list if CLR fails

def apply_log_transformation(df, cols_to_log):
    """
    Applies Log (np.log1p) transformation to specified columns.
    This might be used for features *not* part of the CLR-transformed compositional block,
    if they are highly skewed and require it (e.g., other measurements, ratios).
    """
    if not cols_to_log:
        # print("Log transformation: No columns specified.") # Optional: be less verbose
        return df, []

    df_logged = df.copy()
    logged_col_names = []
    print(f"Applying Log (np.log1p) transformation to: {cols_to_log}")
    for col in cols_to_log:
        if col in df_logged.columns:
            new_col_name = f"{col}_log"
            df_logged[new_col_name] = np.log1p(df_logged[col]) # log1p handles 0s by log(1+x)
            logged_col_names.append(new_col_name)
            # df_logged = df_logged.drop(columns=[col]) # Optionally drop original
            print(f"  Log transformed '{col}' to '{new_col_name}'.")
        else:
            print(f"  Warning: Column '{col}' not found for log transformation.")
    return df_logged, logged_col_names

def cap_outliers_iqr(df, column_list, factor=1.5):
    """Caps outliers in specified columns using the IQR method."""
    # (Implementation from previous response - ensure it's present)
    df_capped = df.copy()
    for column in column_list:
        if column not in df_capped.columns or not pd.api.types.is_numeric_dtype(df_capped[column]):
            # print(f"Outlier Capping: Column {column} not found or not numeric. Skipping.")
            continue
        Q1 = df_capped[column].quantile(0.25)
        Q3 = df_capped[column].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0: # Avoid division by zero or issues if all values are same after some processing
            # print(f"Outlier Capping: IQR is zero for column {column}. Skipping.")
            continue
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        original_min = df_capped[column].min()
        original_max = df_capped[column].max()
        df_capped[column] = np.clip(df_capped[column], lower_bound, upper_bound)
        if original_min < df_capped[column].min() or original_max > df_capped[column].max():
             print(f"  Outlier Capping: Column '{column}' values capped.")
    return df_capped


def preprocess_features(X_input,
                        oxide_cols_config=MAJOR_OXIDES_WT_PERCENT,
                        trace_cols_config=TRACE_ELEMENTS_PPM,
                        categorical_cols_to_handle=CATEGORICAL_COLS_TO_POTENTIALLY_OHE, # Renamed for clarity
                        apply_outlier_capping=True,
                        cols_for_log_transform=None,
                        keep_categorical_features=False): # Modified parameter name for clarity
    """
    Main feature preprocessing function.
    Handles unit harmonization, outlier capping, CLR, optional log transforms,
    and handling of specified categorical columns (OHE if keep_categorical_features=True, else drop).
    Returns processed X DataFrame and list of final feature names *before scaling*.
    """
    X = X_input.copy()

    # --- Unit Harmonization ---
    X, harmonized_ppm_cols_for_clr = harmonize_units_to_ppm(X, oxide_cols_config, trace_cols_config)

    # --- Outlier Capping (Applied to harmonized PPM columns BEFORE CLR) ---
    if apply_outlier_capping and harmonized_ppm_cols_for_clr:
        print("Applying IQR outlier capping to harmonized PPM columns...")
        X = cap_outliers_iqr(X, harmonized_ppm_cols_for_clr)

    # --- CLR Transformation (Applied to ALL harmonized PPM columns) ---
    X, clr_feature_names = apply_clr_transformation(X, harmonized_ppm_cols_for_clr)

    # --- Optional Log Transformation ---
    if cols_for_log_transform:
        X, logged_feature_names = apply_log_transformation(X, cols_for_log_transform)
    else:
        logged_feature_names = []

    # --- Handle Categorical Features ---
    ohe_feature_names = []
    actual_categorical_cols_in_X = [col for col in categorical_cols_to_handle if col in X.columns]

    if actual_categorical_cols_in_X:
        if keep_categorical_features:
            print(f"One-hot encoding categorical features: {actual_categorical_cols_in_X}")
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X[actual_categorical_cols_in_X] = cat_imputer.fit_transform(X[actual_categorical_cols_in_X])

            # Check if any columns were dropped by previous steps (e.g. if they were in oxide_cols_config)
            # This check ensures we only try to OHE columns that still exist AND are in actual_categorical_cols_in_X
            cols_for_ohe_transformer = [col for col in actual_categorical_cols_in_X if col in X.columns]

            if cols_for_ohe_transformer:
                ct_onehot = ColumnTransformer(
                    [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cols_for_ohe_transformer)],
                    remainder='passthrough' # Keep other columns
                )
                X_encoded_np = ct_onehot.fit_transform(X)
                # Get feature names after OHE
                # For 'passthrough' columns, get_feature_names_out uses original names or 'remainder__{colname}'
                # For 'onehot' columns, it prefixes them like 'onehot__{colname}__{category}'
                raw_ohe_names = ct_onehot.get_feature_names_out()

                # Reconstruct DataFrame with proper names
                X = pd.DataFrame(X_encoded_np, columns=raw_ohe_names, index=X.index)

                # Clean up 'remainder__' prefix if present from passthrough columns
                X.columns = [col.replace('remainder__', '') for col in X.columns]

                # Identify which of these new columns are the OHE features
                ohe_feature_names = [name for name in X.columns if name.startswith("onehot__")]
            else:
                print("No categorical columns left to one-hot encode after previous steps.")

        else: # Drop the categorical features
            print(f"Dropping categorical features as keep_categorical_features is False: {actual_categorical_cols_in_X}")
            columns_to_drop_cat = [col for col in actual_categorical_cols_in_X if col in X.columns]
            if columns_to_drop_cat:
                X.drop(columns=columns_to_drop_cat, inplace=True, errors='ignore')

    final_feature_names_before_scaling = X.columns.tolist()
    print(f"Features after preprocessing (before scaling): {final_feature_names_before_scaling}")

    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"WARNING: Non-numeric columns still present after preprocessing: {non_numeric_cols.tolist()}. Coercing to numeric and imputing.")
        for col in non_numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        # Impute NaNs that might have been introduced by coercion or were already there
        # SimpleImputer for numerical columns
        num_imputer = SimpleImputer(strategy='median')
        X[X.select_dtypes(include=np.number).columns] = num_imputer.fit_transform(X.select_dtypes(include=np.number))


    return X, final_feature_names_before_scaling

def load_and_prepare_data(csv_path, target_column_name,
                           test_size=0.25, random_state=42,
                           apply_outlier_capping_main=True,
                           cols_for_log_transform_main=None,
                           gui_model_preparation=False): # Added flag for GUI models
    """
    Top-level function to load data, preprocess features, and split/scale.
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test,
        final_feature_names (after all transforms, used for scaling and model training),
        class_names (from LabelEncoder),
        num_classes,
        label_encoder (fitted),
        scaler_object (fitted StandardScaler),
        X_for_eda_plots (DataFrame for EDA, typically after CLR/log but before OHE/scaling specific to training splits),
        df_original_for_ratio_plots (original DataFrame with imputations for ratio plots)
    """
    print(f"Starting data loading and preparation from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read data file {csv_path}: {e}")
        return None


    if target_column_name not in df.columns:
        print(f"ERROR: Target column '{target_column_name}' not found. Available: {df.columns.tolist()}")
        return None

    # --- Basic Missing Value Imputation ---
    if df[target_column_name].isnull().any():
        print(f"Warning: Target column '{target_column_name}' has missing values. Imputing with mode.")
        df[target_column_name].fillna(df[target_column_name].mode()[0], inplace=True)

    # For categorical columns like 'Deposit', if they exist and are used by preprocess_features
    # Impute before passing to preprocess_features
    for cat_col in CATEGORICAL_COLS_TO_POTENTIALLY_OHE:
        if cat_col in df.columns and df[cat_col].isnull().any():
            print(f"Imputing missing values in categorical column '{cat_col}' with mode before preprocessing.")
            df[cat_col].fillna(df[cat_col].mode()[0], inplace=True)

    numerical_cols_initial = df.select_dtypes(include=np.number).columns.tolist()
    if numerical_cols_initial:
        print(f"Initial median imputation for numerical columns: {numerical_cols_initial}")
        median_imputer_initial = SimpleImputer(strategy='median')
        df[numerical_cols_initial] = median_imputer_initial.fit_transform(df[numerical_cols_initial])

    df_original_for_plots = df.copy() # For ratio plots that need original-like data

    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_) # Ensure it's a list
    num_classes = len(class_names)
    print(f"Target '{target_column_name}' encoded. Classes: {class_names}, N_classes: {num_classes}")

    # --- Feature Preprocessing ---
    # For GUI models, we set keep_categorical_features=False to drop 'Deposit'
    X_processed, final_feature_names = preprocess_features(
        X,
        apply_outlier_capping=apply_outlier_capping_main,
        cols_for_log_transform=cols_for_log_transform_main,
        keep_categorical_features=not gui_model_preparation # If gui_model_preparation=True, then keep_categorical_features=False
    )

    # X_for_eda_plots should reflect the state after geochemical transforms,
    # potentially before OHE if OHE was part of preprocess_features and gui_model_preparation=False.
    # If gui_model_preparation=True, X_processed already has 'Deposit' (and other categoricals) dropped.
    X_for_eda_plots = X_processed.copy()


    # --- SMOTE Oversampling (Applied only to training data after split) ---
    # This section is moved to after train-test split to prevent data leakage from test set

    # --- Split and Scale ---
    # Stratify option
    num_unique_classes_in_y = len(np.unique(y_encoded))
    stratify_option = y_encoded if num_unique_classes_in_y > 1 and num_unique_classes_in_y < len(y_encoded) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify_option
    )
    print(f"Training set shape before SMOTE: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

    # --- SMOTE Oversampling (Applied only to training data) ---
    unique_classes_train, counts_train = np.unique(y_train, return_counts=True)
    if num_classes > 1 and len(unique_classes_train) > 1 and len(counts_train) > 1 and min(counts_train) > 1 : # SMOTE needs at least 2 samples of minority class
        # Check for significant imbalance before applying SMOTE
        if min(counts_train) < max(counts_train) / 2 : # Example threshold for imbalance
             print("Detected class imbalance in training data. Applying SMOTE oversampling...")
             from imblearn.over_sampling import SMOTE # Local import
             smote = SMOTE(random_state=random_state)
             X_train, y_train = smote.fit_resample(X_train, y_train)
             print(f"SMOTE oversampling complete. Training set shape after SMOTE: X_train {X_train.shape}, y_train {y_train.shape}")
        else:
            print("Class distribution in training data is relatively balanced. Skipping SMOTE.")

    else:
        print("Not enough classes or samples in minority class in training data for SMOTE. Skipping SMOTE.")


    print("Scaling features using StandardScaler.")
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train) # Fit on X_train
    X_test_scaled_np = scaler.transform(X_test)    # Transform X_test

    # Convert scaled arrays back to DataFrames with original indices and correct feature names
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=final_feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=final_feature_names, index=X_test.index)

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            final_feature_names, class_names, num_classes,
            label_encoder, scaler, X_for_eda_plots, df_original_for_plots)


def preprocess_data_for_prediction(raw_df_input, scaler_object, trained_feature_names,
                                    oxide_cols_config=MAJOR_OXIDES_WT_PERCENT,
                                    trace_cols_config=TRACE_ELEMENTS_PPM,
                                    cols_for_log_transform_config=None):
    """
    Preprocesses new raw data for prediction using a fitted scaler and known transformations.
    - raw_df_input: DataFrame of new samples with raw feature values.
    - scaler_object: The fitted StandardScaler instance.
    - trained_feature_names: The list of feature names the model was trained on (after CLR, log, etc., before scaling).
    """
    if not isinstance(raw_df_input, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    X_predict = raw_df_input.copy()

    # 0. Select only the geochemical features expected for preprocessing
    # Assuming RAW_EXPECTED_GEOCHEMICAL_FEATURES are the ones to start with
    missing_geo_cols = [col for col in RAW_EXPECTED_GEOCHEMICAL_FEATURES if col not in X_predict.columns]
    if missing_geo_cols:
        # Attempt to fill with zeros or mean, or raise error. For now, let's note and impute later.
        print(f"Warning: Prediction data is missing expected geochemical columns: {missing_geo_cols}. They will be imputed with 0 before processing.")
        for col in missing_geo_cols:
            X_predict[col] = 0 # Or use a more sophisticated imputation strategy if necessary

    # Ensure only expected geochemical features are carried forward for elemental transformations
    X_predict_geo_features_only = X_predict[RAW_EXPECTED_GEOCHEMICAL_FEATURES].copy()


    # 1. Impute missing values in the raw feature set before transformations
    numerical_cols_to_impute = X_predict_geo_features_only.select_dtypes(include=np.number).columns
    if X_predict_geo_features_only[numerical_cols_to_impute].isnull().any().any():
        print("Imputing missing numerical values with median for prediction data...")
        imputer = SimpleImputer(strategy='median')
        X_predict_geo_features_only[numerical_cols_to_impute] = imputer.fit_transform(X_predict_geo_features_only[numerical_cols_to_impute])


    # 2. Apply the same transformations as in preprocess_features (excluding OHE and dropping 'Deposit' explicitly if not in geo features)
    # Unit Harmonization
    X_processed, harmonized_cols = harmonize_units_to_ppm(X_predict_geo_features_only, oxide_cols_config, trace_cols_config)

    # CLR Transformation
    # Outlier capping is usually not applied at prediction time on single/few samples,
    # but if it was part of training PRE-CLR, it might be needed if distributions are very different.
    # For now, skipping outlier capping at prediction.
    X_processed, clr_cols = apply_clr_transformation(X_processed, harmonized_cols)

    # Log Transformation (if any were applied during training)
    if cols_for_log_transform_config:
        # Identify which of these log_transform_cols are actually present after CLR
        # (CLR might have dropped original columns)
        log_cols_present = [col for col in cols_for_log_transform_config if col in X_processed.columns]
        if log_cols_present:
             X_processed, logged_cols = apply_log_transformation(X_processed, log_cols_present)


    # 3. Align columns with the feature set the model was trained on (trained_feature_names)
    # This is crucial: X_processed must now have the same columns, in the same order, as
    # the data used to fit the scaler during training.

    # Check for missing columns that were expected by the trained model (after transformation)
    missing_model_features = [col for col in trained_feature_names if col not in X_processed.columns]
    if missing_model_features:
        # This indicates a mismatch in preprocessing logic or feature set.
        # For robustness, one might add these columns with a default value (e.g., 0 or mean from training)
        # but this is risky if the features are important.
        print(f"ERROR: Processed prediction data is missing features the model expects: {missing_model_features}. Filling with 0.")
        for col in missing_model_features:
            X_processed[col] = 0 # Or another imputation strategy

    # Ensure correct order and subset of columns
    try:
        X_processed = X_processed[trained_feature_names]
    except KeyError as e:
        print(f"ERROR: Could not align prediction data columns with trained feature names. Missing: {e}")
        print(f"Processed columns: {X_processed.columns.tolist()}")
        print(f"Expected trained features: {trained_feature_names}")
        return None

    # 4. Scale the data using the loaded (fitted) scaler
    X_scaled_np = scaler_object.transform(X_processed)
    X_scaled_df = pd.DataFrame(X_scaled_np, columns=trained_feature_names, index=X_predict.index)

    return X_scaled_df

# Add load_data and validate_data if they are not already present or need slight modification for GUI context
# For now, assuming they are mostly suitable from the existing file.
# The `validate_data` function in run_prediction.py uses EXPECTED_COLUMNS.
# Ensure this is consistent with RAW_EXPECTED_GEOCHEMICAL_FEATURES
# Let's rename EXPECTED_COLUMNS in run_prediction.py to RAW_EXPECTED_GEOCHEMICAL_FEATURES and import it.

# In data_handler.py, ensure skbio warning is clear if it's not found during training/prediction.
# The `SKBIO_AVAILABLE` flag is already used.