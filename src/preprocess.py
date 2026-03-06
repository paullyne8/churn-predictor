"""
preprocess.py
-------------
Data cleaning and feature engineering pipeline for the
Telecom Customer Churn Prediction System.

Dataset: IBM Watson Telco Customer Churn Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, fix data types, and remove inconsistencies.
    """
    df = df.copy()

    # TotalCharges has whitespace strings instead of NaN — fix this
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)

    # Drop customerID — not predictive
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    print(f"After cleaning: {df.shape[0]} rows")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using Label Encoding.
    Binary columns (Yes/No) mapped to 1/0.
    """
    df = df.copy()

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Label encode remaining categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df


def split_data(df: pd.DataFrame, target: str = 'Churn', test_size: float = 0.2):
    """
    Split into train/test sets. Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    print(f"Churn rate in train: {y_train.mean():.2%}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data("data/raw/telco_churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Preprocessing complete.")
