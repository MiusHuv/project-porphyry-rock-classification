import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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
    small_epsilon = 1e-10 # A very small number, but larger than machine epsilon
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
                        categorical_cols_config=CATEGORICAL_COLS_APP,
                        apply_outlier_capping=True,
                        cols_for_log_transform=None,
                        keep_categorical=False): # Specify if any non-CLR cols need log
    """
    Main feature preprocessing function.
    Handles unit harmonization, outlier capping, CLR, optional log transforms, and OHE.
    Returns processed X DataFrame and list of final feature names.
    """
    X = X_input.copy()

    # --- Unit Harmonization (Oxides to PPM, Traces stay PPM) ---
    # This defines the set of columns that will form the compositional data for CLR
    X, harmonized_ppm_cols_for_clr = harmonize_units_to_ppm(X, oxide_cols_config, trace_cols_config)
    
    # --- Outlier Capping (Applied to harmonized PPM columns BEFORE CLR) ---
    if apply_outlier_capping and harmonized_ppm_cols_for_clr:
        print("Applying IQR outlier capping to harmonized PPM columns...")
        X = cap_outliers_iqr(X, harmonized_ppm_cols_for_clr)

    # --- CLR Transformation (Applied to ALL harmonized PPM columns) ---
    X, clr_feature_names = apply_clr_transformation(X, harmonized_ppm_cols_for_clr)
    
    # --- Optional Log Transformation (for any *other* numeric features if needed) ---
    # Example: If you had other numeric features not part of CLR that are skewed.
    # For now, we assume major oxides and trace elements are handled by CLR.
    # If you had, say, a 'grain_size' column that was highly skewed:
    # cols_for_log_transform = ['grain_size'] (if it exists in X)
    if cols_for_log_transform:
        X, logged_feature_names = apply_log_transformation(X, cols_for_log_transform)
    else:
        logged_feature_names = []

    # keep_categorical is a flag to control whether to keep categorical columns
    # If True, categorical columns will be one-hot encoded after CLR and log transformations.
    # If False, they will be dropped from the final DataFrame.
    if keep_categorical:
        # --- One-Hot Encode Categorical Features ---
        ohe_feature_names = []
        # Identify categorical columns that are still in X (might have been dropped if part of ppm_cols)
        actual_categorical_cols_to_encode = [col for col in categorical_cols_config if col in X.columns]
        
        if actual_categorical_cols_to_encode:
            print(f"One-hot encoding categorical features: {actual_categorical_cols_to_encode}")
            # Impute missing values in categorical columns before OHE
            cat_imputer = SimpleImputer(strategy='most_frequent', copy=False) # copy=False to modify X inplace temporarily
            X[actual_categorical_cols_to_encode] = cat_imputer.fit_transform(X[actual_categorical_cols_to_encode])

            ct_onehot = ColumnTransformer(
                [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), actual_categorical_cols_to_encode)],
                remainder='passthrough'
            )
            X_encoded_np = ct_onehot.fit_transform(X)
            ohe_feature_names_raw = ct_onehot.get_feature_names_out()
            X = pd.DataFrame(X_encoded_np, columns=ohe_feature_names_raw, index=X.index)
            # Store the OHE feature names (they include prefixes like 'onehot__Deposit_')
            ohe_feature_names = [name for name in ohe_feature_names_raw if name.startswith("onehot__")]
    else:
        # If not keeping categorical, drop them from X
        print("Dropping categorical columns as keep_categorical is False.")
        actual_categorical_cols_to_encode = [col for col in categorical_cols_config if col in X.columns]
        X.drop(columns=actual_categorical_cols_to_encode, inplace=True, errors='ignore')
        ohe_feature_names = []

    # --- Final Feature Names ---

    final_feature_names = X.columns.tolist()
    print(f"Final features after preprocessing: {final_feature_names}")
    
    # Ensure all columns are numeric
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        print(f"ERROR: Non-numeric columns still present: {non_numeric_cols.tolist()}. Coercing to numeric.")
        for col in non_numeric_cols: X[col] = pd.to_numeric(X[col], errors='coerce')
        X.fillna(X.median(), inplace=True) # Impute NaNs introduced by coercion

    return X, final_feature_names


def split_and_scale_data(X_processed, y_encoded, final_feature_names, test_size=0.25, random_state=42):
    """
    Splits processed data into training/testing sets and scales features.
    """
    print(f"Splitting data: test_size={test_size}, random_state={random_state}")
    num_unique_classes_in_y = len(np.unique(y_encoded))
    stratify_option = y_encoded if num_unique_classes_in_y > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify_option
    )
    print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

    print("Scaling all features using StandardScaler.")
    scaler = StandardScaler()
    # X_train and X_test should already be DataFrames with correct columns
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_test_scaled_np = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=final_feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=final_feature_names, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler # Return scaler for inverse_transform or test data


def SMOTE_oversample(X_train, y_train, num_classes, random_state=42):
    """
    Applies SMOTE oversampling to the training data.
    Returns the oversampled X_train and y_train.
    """
    from imblearn.over_sampling import SMOTE

    print("Applying SMOTE oversampling to balance classes in training data...")
    smote = SMOTE(random_state=random_state, n_jobs=-1) # Use all available cores
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"SMOTE oversampling complete: {len(y_train_resampled)} samples after resampling.")
    return X_train_resampled, y_train_resampled

def load_and_prepare_data(csv_path, target_column_name, 
                           test_size=0.25, random_state=42,
                           apply_outlier_capping_main=True,
                           cols_for_log_transform_main=None):
    """
    Top-level function to load data, preprocess features, and split/scale.
    This is the primary function to be called by the training pipeline.
    """
    print(f"Starting data loading and preparation from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {csv_path}")
        return None # Indicate failure clearly

    if target_column_name not in df.columns:
        print(f"ERROR: Target column '{target_column_name}' not found. Available: {df.columns.tolist()}")
        return None
    
    # --- Basic Missing Value Imputation (for target and initial categoricals if needed) ---
    # Impute target if it has NaNs, though usually target shouldn't.
    if df[target_column_name].isnull().any():
        print(f"Warning: Target column '{target_column_name}' has missing values. Imputing with mode.")
        df[target_column_name].fillna(df[target_column_name].mode()[0], inplace=True)

    # Impute other essential categorical columns (like 'Deposit') before they are used or OHE'd
    for cat_col in CATEGORICAL_COLS_APP: # Using the global config
        if cat_col in df.columns and df[cat_col].isnull().any():
            print(f"Imputing missing values in categorical column '{cat_col}' with mode.")
            df[cat_col].fillna(df[cat_col].mode()[0], inplace=True)
    
    # --- Impute numerical columns (oxides and trace elements) that might be used before full preprocess_features ---
    # This is a general imputation for all numericals before specific processing starts.
    numerical_cols_initial_impute = df.select_dtypes(include=np.number).columns.tolist()
    if numerical_cols_initial_impute:
        print(f"Initial median imputation for numerical columns: {numerical_cols_initial_impute}")
        median_imputer = SimpleImputer(strategy='median', copy=False) # copy=False to modify df
        df[numerical_cols_initial_impute] = median_imputer.fit_transform(df[numerical_cols_initial_impute])


    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print(f"Target '{target_column_name}' encoded. Classes: {class_names}, N_classes: {num_classes}")

    # --- Feature Preprocessing ---
    X_processed, final_feature_names = preprocess_features(
        X, 
        apply_outlier_capping=apply_outlier_capping_main,
        cols_for_log_transform=cols_for_log_transform_main
    )
    
    # --- Data for EDA (after geochemical transforms but before OHE of 'Deposit' and scaling) ---
    # This requires careful thought on what 'X_for_eda_plots_intermediate' should contain.
    # If preprocess_features handles OHE internally, we need X *before* that for some EDA.
    # Let's get it before OHE within preprocess_features if possible, or reconstruct.
    # For now, we'll pass the X_processed which contains CLR/log but also OHE features.
    # A more refined EDA df might need X just after harmonize_units and CLR.
    X_for_eda_plots = X_processed.copy() # This X_processed has CLR, log, OHE.
                                        # For EDA on CLR without OHE, would need X before OHE step.

    # --- SMOTE Oversampling (if needed) ---
    if num_classes > 1 and len(np.unique(y_encoded)) < num_classes:
        print("Detected class imbalance. Applying SMOTE oversampling to training data.")
        X_processed, y_encoded = SMOTE_oversample(X_processed, y_encoded, num_classes, random_state)
    else:
        print("No class imbalance detected or only one class present. Skipping SMOTE.")
    # If SMOTE was applied, X_processed and y_encoded will be the resampled data.

    # --- Split and Scale ---
    X_train_scaled, X_test_scaled, y_train, y_test, scaler_object = split_and_scale_data(
        X_processed, y_encoded, final_feature_names, test_size, random_state
    )

    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            final_feature_names, class_names, num_classes, 
            label_encoder, scaler_object, X_for_eda_plots, df.copy()) # Return original df for ratio plots