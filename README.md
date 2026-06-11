# Credit Card Fraud Detection

## Overview
Professional credit card fraud detection pipeline with training, evaluation, and inference.

This implementation provides a reusable training pipeline using scikit-learn, model persistence, and a simple CLI for training and batch inference.

## Features
- Python implementation
- Easy to extend
- GitHub-ready structure

## Installation

```bash
pip install -r requirements.txt
```

## Run

Training example (using a CSV with a `Class` column):

```bash
python fraud_detection.py --train path/to/creditcard.csv --model fraud_model.joblib
```

Run batch inference on new transactions (CSV without `Class`):

```bash
python fraud_detection.py --model fraud_model.joblib --infer new_transactions.csv --save-preds out.csv
```

The script saves the trained pipeline (scaler + model) as a joblib file.

## Author
Divya Nimbalkar
