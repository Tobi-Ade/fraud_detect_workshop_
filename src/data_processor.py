"""
Fraud Detection Workshop - Data Processing Module
==================================================

Handles dataset loading, preprocessing, balancing, and feature engineering
for fraud detection training and inference.

Author: Workshop Team
Date: 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import urllib.request


class FraudDataProcessor:
    """
    Handles all data processing operations for fraud detection.

    Optimized for SageMaker training and inference.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the data processor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.feature_columns: List[str] = []
        self.target_column: str = 'Class'

    def download_dataset(
        self,
        url: str = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
        local_path: str = "creditcard.csv"
    ) -> bool:
        """
        Download the credit card fraud dataset from a public URL.

        Args:
            url: URL to download from
            local_path: Local file path to save to

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Downloading dataset from {url}")
            print("This may take 1-2 minutes...")
            urllib.request.urlretrieve(url, local_path)
            print(f"Dataset downloaded successfully: {local_path}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {len(df):,} transactions")
        print(f"Features: {len(df.columns) - 1}")

        # Store feature columns
        self.feature_columns = [col for col in df.columns if col != self.target_column]

        return df

    def analyze_imbalance(self, df: pd.DataFrame) -> dict:
        """
        Analyze class imbalance in the dataset.

        Args:
            df: DataFrame with Class column

        Returns:
            Dictionary with imbalance statistics
        """
        fraud_count = len(df[df[self.target_column] == 1])
        legit_count = len(df[df[self.target_column] == 0])
        fraud_pct = (fraud_count / len(df)) * 100

        stats = {
            'total': len(df),
            'legitimate': legit_count,
            'fraud': fraud_count,
            'fraud_percentage': fraud_pct,
            'imbalance_ratio': legit_count // fraud_count if fraud_count > 0 else 0
        }

        print("\n" + "=" * 60)
        print("Class Distribution Analysis")
        print("=" * 60)
        print(f"Total transactions: {stats['total']:,}")
        print(f"Legitimate: {stats['legitimate']:,} ({100-fraud_pct:.2f}%)")
        print(f"Fraudulent: {stats['fraud']:,} ({fraud_pct:.3f}%)")
        print(f"Imbalance ratio: {stats['imbalance_ratio']}:1")
        print("=" * 60 + "\n")

        return stats

    def balance_dataset(
        self,
        df: pd.DataFrame,
        sampling_strategy: float = 0.3
    ) -> pd.DataFrame:
        """
        Balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique).

        Args:
            df: Imbalanced DataFrame
            sampling_strategy: Ratio of minority to majority class after balancing

        Returns:
            Balanced DataFrame
        """
        print(f"\nApplying SMOTE to balance dataset...")
        print(f"Target fraud ratio: {sampling_strategy:.1%}\n")

        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        print(f"Before SMOTE:")
        print(f"  Legitimate: {sum(y == 0):,}")
        print(f"  Fraud: {sum(y == 1):,}")

        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)

        print(f"\nAfter SMOTE:")
        print(f"  Legitimate: {sum(y_balanced == 0):,}")
        print(f"  Fraud: {sum(y_balanced == 1):,}")
        print(f"  New fraud ratio: {sum(y_balanced == 1) / len(y_balanced):.1%}\n")

        # Create balanced DataFrame
        df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        df_balanced[self.target_column] = y_balanced

        return df_balanced

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.

        Args:
            df: DataFrame to split
            test_size: Proportion of data for testing
            stratify: Whether to stratify split by target class

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]

        stratify_param = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        print("\n" + "=" * 60)
        print("Train/Test Split")
        print("=" * 60)
        print(f"Training set: {len(X_train):,} transactions")
        print(f"Test set: {len(X_test):,} transactions")
        print("=" * 60 + "\n")

        return X_train, X_test, y_train, y_test

    def derive_signals(self, transaction_row: pd.Series) -> List[str]:
        """
        Extract human-readable signals from a transaction for explainability.

        Args:
            transaction_row: pandas Series with transaction features

        Returns:
            List of signal strings describing the transaction
        """
        signals = []

        # Amount-based signals
        amount = float(transaction_row.get('Amount', 0))
        if amount > 500:
            signals.append(f'High amount (${amount:,.2f})')
        elif amount < 5:
            signals.append(f'Micro-transaction (${amount:.2f})')
        else:
            signals.append(f'Normal amount (${amount:,.2f})')

        # Time-based signals
        time_seconds = float(transaction_row.get('Time', 0))
        hour = int((time_seconds / 3600) % 24)

        if 0 <= hour < 6:
            signals.append(f'Off-hours transaction ({hour:02d}:00)')
        elif 6 <= hour < 12:
            signals.append(f'Morning transaction ({hour:02d}:00)')
        elif 12 <= hour < 18:
            signals.append(f'Afternoon transaction ({hour:02d}:00)')
        else:
            signals.append(f'Evening transaction ({hour:02d}:00)')

        # Anomaly detection
        extreme_features = []
        for col in transaction_row.index:
            if col.startswith('V') and abs(transaction_row[col]) > 3:
                extreme_features.append(col)

        if extreme_features:
            signals.append(f'Anomalous patterns detected ({len(extreme_features)} features)')

        if len(extreme_features) > 5:
            signals.append('Multiple anomalies - high risk pattern')

        return signals

    def prepare_for_inference(self, transaction_data: dict) -> pd.DataFrame:
        """
        Prepare a single transaction for model inference.

        Args:
            transaction_data: Dictionary with transaction features

        Returns:
            DataFrame with single row ready for model prediction
        """
        if not self.feature_columns:
            self.feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

        # Create DataFrame with proper column order
        df = pd.DataFrame([transaction_data])

        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder columns to match training data
        df = df[self.feature_columns]

        return df
