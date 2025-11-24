"""
Preprocessing Module - Load, Clean, Transform
Uses functional approach (no classes) for simplicity
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler
from typing import Optional, Tuple, List


def load_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:

    try:
        df = pd.read_csv(filepath, encoding=encoding)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def select_numeric_features(
    df: pd.DataFrame, 
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:

    # Select numeric types
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Exclude specific columns if provided
    if exclude_cols:
        df_numeric = df_numeric.drop(columns=exclude_cols, errors='ignore')
    
    print(f"✓ Numeric features selected: {df_numeric.shape[1]} columns")
    print(f"  Features: {list(df_numeric.columns)}")
    
    return df_numeric


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'drop',
    fill_value: Optional[float] = None
) -> pd.DataFrame:

    n_missing = df.isnull().sum().sum()
    
    if n_missing == 0:
        print("✓ No missing values found")
        return df
    
    print(f"⚠ Found {n_missing} missing values")
    
    if strategy == 'drop':
        df_clean = df.dropna()
        print(f"  Dropped rows: {len(df) - len(df_clean)}")
    elif strategy == 'mean':
        df_clean = df.fillna(df.mean())
        print(f"  Filled with mean")
    elif strategy == 'median':
        df_clean = df.fillna(df.median())
        print(f"  Filled with median")
    elif strategy == 'constant':
        df_clean = df.fillna(fill_value)
        print(f"  Filled with constant: {fill_value}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean


def yeo_johnson_transform(
    X: np.ndarray, 
    standardize: bool = False
) -> Tuple[np.ndarray, PowerTransformer]:

    transformer = PowerTransformer(method='yeo-johnson', standardize=standardize)
    X_transformed = transformer.fit_transform(X)
    
    print("✓ Yeo-Johnson transformation applied")
    print(f"  Original skewness: {np.mean([abs(pd.Series(X[:, i]).skew()) for i in range(X.shape[1])]):.3f}")
    print(f"  Transformed skewness: {np.mean([abs(pd.Series(X_transformed[:, i]).skew()) for i in range(X_transformed.shape[1])]):.3f}")
    
    return X_transformed, transformer


def robust_scale(
    X: np.ndarray,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: Tuple[float, float] = (25.0, 75.0)
) -> Tuple[np.ndarray, RobustScaler]:

    scaler = RobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range
    )
    X_scaled = scaler.fit_transform(X)
    
    print("✓ RobustScaler applied")
    print(f"  Median: {np.median(X_scaled, axis=0)[:3]}...")  # Show first 3 features
    print(f"  IQR: {np.percentile(X_scaled, 75, axis=0)[:3] - np.percentile(X_scaled, 25, axis=0)[:3]}...")
    
    return X_scaled, scaler


def preprocess_pipeline(
    filepath: str,
    exclude_cols: Optional[List[str]] = None,
    missing_strategy: str = 'drop',
    apply_yeo_johnson: bool = True,
    apply_robust_scale: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, pd.DataFrame, dict]:

    if random_state is not None:
        np.random.seed(random_state)
    
    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_data(filepath)
    df_original = df.copy()
    
    # Step 2: Select numeric features
    df_numeric = select_numeric_features(df, exclude_cols=exclude_cols)
    
    # Step 3: Handle missing values
    df_clean = handle_missing_values(df_numeric, strategy=missing_strategy)
    
    # Convert to numpy
    X = df_clean.values
    print(f"\n✓ Converted to numpy array: {X.shape}")
    
    transformers = {}
    
    # Step 4: Yeo-Johnson transformation
    if apply_yeo_johnson:
        print("\n" + "-"*60)
        X, yj_transformer = yeo_johnson_transform(X, standardize=False)
        transformers['yeo_johnson'] = yj_transformer
    
    # Step 5: RobustScaler
    if apply_robust_scale:
        print("\n" + "-"*60)
        X_scaled, scaler = robust_scale(X)
        transformers['robust_scaler'] = scaler
    else:
        X_scaled = X
    
    print("\n" + "="*60)
    print(f"✓ PREPROCESSING COMPLETE")
    print(f"  Final shape: {X_scaled.shape}")
    print(f"  Features: {list(df_clean.columns)}")
    print("="*60)
    
    return X_scaled, df_original, transformers


# Example usage
# ============================================================
# OOP Wrapper: DataPreprocessor (compatible with main.ipynb)
# ============================================================

class DataPreprocessor:
    """
    Wrapper class so preprocessing can be called as an object.
    Internally uses the functional pipeline you already wrote.
    """

    def __init__(
        self,
        exclude_cols=None,
        missing_strategy='drop',
        apply_yeo_johnson=True,
        apply_robust_scale=True,
        random_state=None
    ):
        self.exclude_cols = exclude_cols
        self.missing_strategy = missing_strategy
        self.apply_yeo_johnson = apply_yeo_johnson
        self.apply_robust_scale = apply_robust_scale
        self.random_state = random_state

    # ------------------------------------------------------------
    def load_data(self, filepath, encoding='utf-8'):
        """Proxy to the standalone load_data function"""
        return load_data(filepath, encoding=encoding)

    # ------------------------------------------------------------
    def transform(self, df):
        """
        Apply preprocessing steps to an already-loaded DataFrame.
        Acts similarly to preprocess_pipeline but without loading file.
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)

        print("=" * 60)
        print("PREPROCESSING (DataFrame Input)")
        print("=" * 60)

        # Step 1: numeric selection
        df_numeric = select_numeric_features(df, exclude_cols=self.exclude_cols)

        # Step 2: handle missing
        df_clean = handle_missing_values(df_numeric, strategy=self.missing_strategy)

        X = df_clean.values
        transformers = {}

        # Step 3: Yeo-Johnson
        if self.apply_yeo_johnson:
            X, yj = yeo_johnson_transform(X, standardize=False)
            transformers['yeo_johnson'] = yj

        # Step 4: RobustScaler
        if self.apply_robust_scale:
            X_scaled, scaler = robust_scale(X)
            transformers['robust_scaler'] = scaler
        else:
            X_scaled = X

        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETE (DataFrame input)")
        print(f"  Final shape: {X_scaled.shape}")
        print("="*60)

        return X_scaled

    # ------------------------------------------------------------
    def fit_transform(self, filepath):
        """
        Equivalent to preprocess_pipeline() but inside the class.
        """
        X_scaled, df_orig, transformers = preprocess_pipeline(
            filepath=filepath,
            exclude_cols=self.exclude_cols,
            missing_strategy=self.missing_strategy,
            apply_yeo_johnson=self.apply_yeo_johnson,
            apply_robust_scale=self.apply_robust_scale,
            random_state=self.random_state
        )
        self.transformers = transformers
        return X_scaled, df_orig
