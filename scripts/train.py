"""
SageMaker Training Script for Fraud Detection
==============================================

This script is designed to run in AWS SageMaker training jobs.
It handles data loading, model training, and model saving.

Author: Workshop Team
Date: 2026
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import sys
import json

# Add src to path for imports
sys.path.insert(0, '/opt/ml/code/src')
from data_processor import FraudDataProcessor


def train(args):
    """
    Main training function for SageMaker.

    Args:
        args: Command-line arguments
    """
    print("=" * 60)
    print("Starting Fraud Detection Model Training")
    print("=" * 60)

    # Initialize data processor
    processor = FraudDataProcessor(random_state=args.random_state)

    # Load training data
    print(f"\nLoading training data from {args.train}...")
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))

    # Load validation data if provided
    val_df = None
    if os.path.exists(os.path.join(args.validation, 'validation.csv')):
        print(f"Loading validation data from {args.validation}...")
        val_df = pd.read_csv(os.path.join(args.validation, 'validation.csv'))

    # Separate features and target
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Fraud cases: {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"Legitimate cases: {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
    print(f"\nScale pos weight: {scale_pos_weight:.2f}")

    # Initialize and train XGBoost model
    print("\n" + "=" * 60)
    print("Training XGBoost Model")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist',  # Faster for large datasets
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)
    print("\nModel training completed!")

    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    print("\n" + "=" * 60)
    print("Training Set Performance")
    print("=" * 60)
    print(classification_report(y_train, y_train_pred, target_names=['Legitimate', 'Fraud']))
    print(f"ROC-AUC Score: {roc_auc_score(y_train, y_train_proba):.4f}")

    # Evaluate on validation data if available
    if val_df is not None:
        X_val = val_df.drop('Class', axis=1)
        y_val = val_df['Class']

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        print("\n" + "=" * 60)
        print("Validation Set Performance")
        print("=" * 60)
        print(classification_report(y_val, y_val_pred, target_names=['Legitimate', 'Fraud']))

        roc_auc = roc_auc_score(y_val, y_val_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # Calculate confusion matrix for validation set
        cm = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = cm.ravel()

        # Save metrics for SageMaker
        metrics = {
            'validation:roc_auc': float(roc_auc),
            'validation:precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'validation:recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'validation:true_positives': int(tp),
            'validation:false_positives': int(fp),
            'validation:true_negatives': int(tn),
            'validation:false_negatives': int(fn)
        }

        print("\nValidation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    # Save the model
    model_path = os.path.join(args.model_dir, 'fraud_detection_model.pkl')
    joblib.dump(model, model_path)

    print(f"\n" + "=" * 60)
    print(f"Model saved to: {model_path}")

    # Get model file size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    print("=" * 60)

    # Save feature names for inference
    feature_names_path = os.path.join(args.model_dir, 'feature_names.json')
    feature_names = list(X_train.columns)
    with open(feature_names_path, 'w') as f:
        json.dump({'features': feature_names}, f)
    print(f"Feature names saved to: {feature_names_path}")

    print("\nTraining job completed successfully!")


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=42)

    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
