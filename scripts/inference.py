"""
SageMaker Inference Script for Fraud Detection
==============================================

This script handles model loading and predictions for SageMaker endpoints.
It implements the required functions for SageMaker hosting.

Author: Workshop Team
Date: 2026
"""

import json
import joblib
import pandas as pd
import numpy as np
import os


def model_fn(model_dir):
    """
    Load the model for inference.

    This function is called once when the endpoint is created.

    Args:
        model_dir: Directory where model artifacts are stored

    Returns:
        Loaded model object
    """
    model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
    model = joblib.load(model_path)

    # Load feature names if available
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_data = json.load(f)
            model.feature_names = feature_data.get('features', [])
    else:
        # Default feature names
        model.feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    print(f"Model loaded successfully from {model_path}")
    return model


def input_fn(request_body, content_type='application/json'):
    """
    Deserialize and prepare the prediction input.

    Args:
        request_body: The request payload
        content_type: The content type of the request

    Returns:
        Input data in the format expected by predict_fn
    """
    if content_type == 'application/json':
        data = json.loads(request_body)

        # Handle both single transaction and batch predictions
        if isinstance(data, dict):
            # Single transaction
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Batch of transactions
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported input format: {type(data)}")

        return df

    elif content_type == 'text/csv':
        # Handle CSV input
        import io
        df = pd.read_csv(io.StringIO(request_body))
        return df

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.

    Args:
        input_data: Input DataFrame from input_fn
        model: Loaded model from model_fn

    Returns:
        Dictionary with predictions and probabilities
    """
    # Ensure all required features are present
    for feature in model.feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0.0

    # Reorder columns to match training data
    input_data = input_data[model.feature_names]

    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    # Derive signals for explainability (simplified version)
    signals_list = []
    for idx, row in input_data.iterrows():
        signals = derive_signals(row)
        signals_list.append(signals)

    # Prepare results
    results = []
    for i in range(len(predictions)):
        result = {
            'prediction': int(predictions[i]),
            'fraud_probability': float(probabilities[i]),
            'risk_level': get_risk_level(probabilities[i]),
            'signals': signals_list[i]
        }
        results.append(result)

    return results


def output_fn(prediction, accept='application/json'):
    """
    Serialize the prediction output.

    Args:
        prediction: Output from predict_fn
        accept: The desired content type of the response

    Returns:
        Serialized prediction
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


def derive_signals(transaction_row):
    """
    Extract human-readable signals from a transaction.

    Args:
        transaction_row: pandas Series with transaction features

    Returns:
        List of signal strings
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
    extreme_features = 0
    for col in transaction_row.index:
        if col.startswith('V') and abs(transaction_row[col]) > 3:
            extreme_features += 1

    if extreme_features > 0:
        signals.append(f'Anomalous patterns detected ({extreme_features} features)')

    if extreme_features > 5:
        signals.append('Multiple anomalies - high risk pattern')

    return signals


def get_risk_level(probability):
    """
    Classify transaction into risk levels.

    Args:
        probability: Fraud probability (0.0 to 1.0)

    Returns:
        Risk level string
    """
    if probability >= 0.9:
        return 'CRITICAL'
    elif probability >= 0.7:
        return 'HIGH'
    elif probability >= 0.3:
        return 'MEDIUM'
    else:
        return 'LOW'
