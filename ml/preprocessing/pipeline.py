import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import sys
from pathlib import Path

from ml.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, PREPROCESSORS_DIR
from ml.utils import setup_logger

logger = setup_logger("preprocessing")

def build_preprocessor(df, scale_numeric=False):
    """
    Builds a ColumnTransformer for preprocessing.
    If scale_numeric is True, applies StandardScaler to numericals.
    Otherwise, just imputes missing numericals.
    Categoricals are One-Hot Encoded.
    """
    # Filter only features present in the dataset
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    logger.info(f"Building preprocessor. Num cols: {len(num_cols)}, Cat cols: {len(cat_cols)}")
    
    if scale_numeric:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop' # Drop any column not explicitly defined as a feature
    )
    
    return preprocessor

def get_fitted_preprocessor(train_df, scale_numeric=False, target_name=""):
    """
    Returns a fitted preprocessor. Fits on train_df.
    Saves the preprocessor object.
    """
    suffix = "scaled" if scale_numeric else "unscaled"
    name = f"preprocessor_{target_name}_{suffix}.joblib" if target_name else f"preprocessor_{suffix}.joblib"
    path = PREPROCESSORS_DIR / name
    
    preprocessor = build_preprocessor(train_df, scale_numeric=scale_numeric)
    
    logger.info(f"Fitting preprocessor {name} on training data...")
    preprocessor.fit(train_df)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Saved preprocessor to {path}")
    
    return preprocessor
