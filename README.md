# Fraud Detection on AWS SageMaker

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready fraud detection system built with XGBoost and deployed on AWS SageMaker. This workshop demonstrates end-to-end ML operations including data processing, model training, and real-time endpoint deployment.

## Overview

This project provides a complete, hands-on demonstration of:

- **XGBoost ML Model** - High-performance gradient boosting for fraud detection
- **SMOTE Balancing** - Handling severely imbalanced datasets (0.17% fraud rate)
- **SageMaker Training** - Scalable cloud-based model training
- **SageMaker Endpoints** - Real-time inference with auto-scaling
- **Production Code** - Ready-to-use integration examples

### Why This Matters

- Credit card fraud costs billions annually
- Real-time detection is critical for preventing losses
- Explainable AI helps with compliance and customer trust
- SageMaker provides enterprise-grade ML infrastructure

## Project Structure

```
fraud_detection_workshop/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package configuration
├── .gitignore                          # Git ignore rules
│
├── notebooks/
│   └── sagemaker_fraud_detection.ipynb # Main SageMaker notebook
│
├── scripts/
│   ├── train.py                        # SageMaker training script
│   └── inference.py                    # SageMaker inference script
│
├── src/
│   ├── __init__.py
│   └── data_processor.py               # Data processing utilities
│
├── data/                               # Dataset directory (git-ignored)
├── models/                             # Model storage (git-ignored)
└── config/                             # Configuration files
```

## Quick Start

### Prerequisites

- AWS Account with SageMaker access
- IAM role with SageMaker permissions
- Python 3.10 or higher

### Option 1: Run in SageMaker Studio (Recommended)

1. **Open SageMaker Studio**
   ```bash
   # Navigate to AWS Console → SageMaker → Studio
   # Create a Studio domain if you don't have one
   ```

2. **Clone this repository**
   ```bash
   git clone <your-repo-url>
   cd fraud_detection_workshop
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Open and run the notebook**
   ```bash
   # Open notebooks/sagemaker_fraud_detection.ipynb
   # Run all cells sequentially
   ```

### Option 2: Run in Local SageMaker Notebook Instance

1. **Create a Notebook Instance**
   - Navigate to SageMaker → Notebook instances
   - Create new instance (ml.t3.medium recommended)
   - Wait for status: InService

2. **Clone and run**
   - Click "Open JupyterLab"
   - Open terminal and clone this repo
   - Navigate to `notebooks/sagemaker_fraud_detection.ipynb`
   - Run all cells

## Notebook Walkthrough

The main notebook ([sagemaker_fraud_detection.ipynb](notebooks/sagemaker_fraud_detection.ipynb)) covers:

### 1. Setup and Configuration
- AWS SDK initialization
- S3 bucket creation
- IAM role configuration

### 2. Data Preparation
- Download credit card fraud dataset (284,807 transactions)
- Analyze severe class imbalance (0.172% fraud rate)
- Apply SMOTE balancing (30% fraud ratio)
- Split into train/validation sets

### 3. SageMaker Training
- Upload data to S3
- Configure XGBoost estimator
- Train on `ml.m5.xlarge` instance
- Monitor training progress

### 4. Model Deployment
- Deploy to SageMaker endpoint
- Configure `ml.t2.medium` instance
- Enable auto-scaling (optional)

### 5. Testing and Validation
- Single transaction predictions
- Batch predictions
- Risk level classification
- Signal detection for explainability

### 6. Production Integration
- boto3 runtime client examples
- JSON request/response format
- Error handling patterns

### 7. Cleanup
- Delete endpoint to avoid charges
- Clean up S3 resources

## Cost Estimation

Approximate costs for a complete workshop run:

| Resource | Configuration | Duration | Cost |
|----------|--------------|----------|------|
| Training | ml.m5.xlarge | 10 mins | $0.23 |
| Endpoint (testing) | ml.t2.medium | 1 hour | $0.10 |
| S3 Storage | 500 MB | 1 day | $0.01 |
| Data Transfer | Minimal | - | $0.01 |
| **Total** | | | **~$0.35** |

**Important**: Delete the endpoint after testing to avoid ongoing hourly charges!

## Training Script

The training script ([scripts/train.py](scripts/train.py)) is SageMaker-compatible and includes:

- Command-line argument parsing
- Data loading from S3 channels
- XGBoost model training
- Validation metrics
- Model serialization

**Hyperparameters:**
- `n_estimators`: 100 (number of trees)
- `max_depth`: 6 (tree depth)
- `learning_rate`: 0.1
- `scale_pos_weight`: Auto-calculated from class imbalance

## Inference Script

The inference script ([scripts/inference.py](scripts/inference.py)) implements:

- `model_fn()` - Model loading
- `input_fn()` - Request deserialization (JSON/CSV)
- `predict_fn()` - Predictions and signal extraction
- `output_fn()` - Response serialization

**Input Format:**
```json
{
  "Time": 72000,
  "Amount": 149.62,
  "V1": -1.359807,
  "V2": -0.072781,
  ...
  "V28": 0.014724
}
```

**Output Format:**
```json
{
  "prediction": 0,
  "fraud_probability": 0.0234,
  "risk_level": "LOW",
  "signals": [
    "Normal amount ($149.62)",
    "Afternoon transaction (20:00)",
    "Anomalous patterns detected (2 features)"
  ]
}
```

## Integration Examples

### Python with boto3

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

payload = {
    "Time": 72000,
    "Amount": 149.62,
    "V1": -1.359807,
    # ... other features
}

response = runtime.invoke_endpoint(
    EndpointName='fraud-detection-endpoint',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read())
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Predictions

```python
batch_payload = [
    {"Time": 72000, "Amount": 149.62, ...},
    {"Time": 73000, "Amount": 2.99, ...},
    {"Time": 74000, "Amount": 1250.00, ...}
]

response = runtime.invoke_endpoint(
    EndpointName='fraud-detection-endpoint',
    ContentType='application/json',
    Body=json.dumps(batch_payload)
)

results = json.loads(response['Body'].read())
for i, result in enumerate(results):
    print(f"Transaction {i+1}: {result['risk_level']}")
```

## Performance Metrics

Expected model performance on balanced dataset:

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.98+ |
| Precision | 0.95+ |
| Recall | 0.93+ |
| F1-Score | 0.94+ |

## Troubleshooting

### Issue: "ResourceLimitExceeded" during training

**Solution**: Your account may have reached the instance limit. Request a limit increase via AWS Support or use a smaller instance type.

### Issue: "AccessDeniedException" when creating S3 bucket

**Solution**: Ensure your SageMaker execution role has `s3:CreateBucket` and `s3:PutObject` permissions.

### Issue: Endpoint deployment fails

**Solution**: Check CloudWatch logs for detailed error messages:
```python
import boto3
logs = boto3.client('logs')
# Check log group: /aws/sagemaker/Endpoints/fraud-detection-endpoint
```

### Issue: High inference latency

**Solution**:
1. Use larger endpoint instance (e.g., ml.m5.large)
2. Enable auto-scaling for multiple instances
3. Consider SageMaker Batch Transform for non-real-time predictions

## Best Practices

### Security
- Use IAM roles with least privilege
- Enable S3 bucket encryption
- Use VPC endpoints for SageMaker (production)
- Rotate access keys regularly

### Cost Optimization
- Delete endpoints when not in use
- Use Spot instances for training (50-90% savings)
- Enable S3 lifecycle policies to delete old data
- Use SageMaker Savings Plans for production

### Monitoring
- Enable CloudWatch metrics for endpoints
- Set up alarms for latency and error rates
- Track model drift with SageMaker Model Monitor
- Log all predictions for audit trail

## Dataset

This project uses the **Credit Card Fraud Detection** dataset:
- Source: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions: 284,807
- Fraud cases: 492 (0.172%)
- Features: 30 (Time, Amount, V1-V28)
- V1-V28 are PCA-transformed features (anonymized)

## License

MIT License - Free for educational and commercial use.

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review AWS SageMaker [documentation](https://docs.aws.amazon.com/sagemaker/)
3. Open an issue in this repository

## Cleanup Checklist

After completing the workshop:

- [ ] Delete SageMaker endpoint
- [ ] Delete endpoint configuration
- [ ] Delete model
- [ ] Delete S3 bucket contents
- [ ] (Optional) Delete S3 bucket
- [ ] Stop/delete notebook instance (if using one)

```python
# Quick cleanup code (run in notebook)
import boto3

sm = boto3.client('sagemaker')
s3 = boto3.client('s3')

# Delete endpoint
sm.delete_endpoint(EndpointName='fraud-detection-endpoint')
sm.delete_endpoint_config(EndpointConfigName='fraud-detection-endpoint-config')
sm.delete_model(ModelName='fraud-detection-model')

# Delete S3 bucket (use your bucket name)
bucket_name = 'your-bucket-name'
objects = s3.list_objects_v2(Bucket=bucket_name)
if 'Contents' in objects:
    for obj in objects['Contents']:
        s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
s3.delete_bucket(Bucket=bucket_name)
```

---

**Built for Production ML on AWS SageMaker**
**Author**: Workshop Team
**Year**: 2026
