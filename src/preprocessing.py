import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from src.config import DATA_FILE, RANDOM_STATE, TEST_SIZE

def load_data():
    """Loads the raw credit card dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}. Please download it from Kaggle and place it there.")
    df = pd.read_csv(DATA_FILE)
    return df

def feature_engineering(df):
    """
    Applies feature engineering:
    1. Hour of Day (from Time in seconds)
    2. Amount_Log (Log transformation to handle skewness)
    3. Drops original Time column (optional, but let's keep it or drop if needed).
    """
    df_engineered = df.copy()
    
    # 1. Hour_of_Day
    # Time is seconds elapsed since first transaction. 
    # 3600 seconds in an hour, 24 hours in a day.
    df_engineered['Hour_of_Day'] = (df_engineered['Time'] // 3600) % 24
    
    # 2. Amount_Log
    # Adding a small constant to avoid log(0)
    df_engineered['Amount_Log'] = np.log1p(df_engineered['Amount'])
    
    return df_engineered

def scale_features(df):
    """
    Scales Amount_Log and Time (or other features) using RobustScaler
    since the data has extreme outliers.
    """
    df_scaled = df.copy()
    scaler = RobustScaler()
    
    # We will scale 'Amount_Log' and 'Time'
    # The V1-V28 are already PCA transformed and likely scaled, but we can scale Amount and Time to match.
    df_scaled['Amount_Log_Scaled'] = scaler.fit_transform(df_scaled['Amount_Log'].values.reshape(-1, 1))
    df_scaled['Time_Scaled'] = scaler.fit_transform(df_scaled['Time'].values.reshape(-1, 1))
    df_scaled['Hour_of_Day_Scaled'] = scaler.fit_transform(df_scaled['Hour_of_Day'].values.reshape(-1, 1))
    
    # Drop intermediate columns if desired to keep it clean, or keep them.
    df_scaled.drop(['Amount', 'Amount_Log', 'Time', 'Hour_of_Day'], axis=1, inplace=True)
    return df_scaled

def prepare_data(df, apply_smote=True):
    """
    Splits the data into train/test and optionally applies SMOTE.
    Returns X_train, X_test, y_train, y_test
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    if apply_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    return X_train, X_test, y_train, y_test

def full_preprocessing_pipeline(apply_smote=True):
    """Runs the entire preprocessing pipeline."""
    df = load_data()
    df = feature_engineering(df)
    df = scale_features(df)
    return prepare_data(df, apply_smote=apply_smote)
