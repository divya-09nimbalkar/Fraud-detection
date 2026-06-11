"""
Credit Card Fraud Detection
Starter portfolio project.
"""
import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DEFAULT_MODEL = "fraud_model.joblib"


def load_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset with 0/1 labels.")
    X = df.drop(columns=["Class"]) 
    y = df["Class"]
    return X, y


def build_and_train(X: pd.DataFrame, y: pd.Series, use_smote: bool = False, plots_dir: str = None) -> Tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)


    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42),
        ),
    ])

    # Simple hyperparameter tuning
    params = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 8, 16],
    }

    gs = GridSearchCV(pipeline, params, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    # Optional plotting
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
            plt.close()

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        plt.close()

    metrics = {"report": report, "roc_auc": auc, "best_params": gs.best_params_}
    return best, metrics


def save_model(pipeline: Pipeline, path: str = DEFAULT_MODEL):
    joblib.dump(pipeline, path)


def load_model(path: str) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def run_inference(model_path: str, input_csv: str, output_csv: str = "predictions.csv"):
    model = load_model(model_path)
    df = pd.read_csv(input_csv)

    # If the input CSV accidentally contains the label column, drop it
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Try to align columns to what the pipeline saw during training
    expected_cols = None
    try:
        # look for a fitted transformer that exposes feature names
        if hasattr(model, "named_steps") and "scaler" in model.named_steps:
            scaler = model.named_steps["scaler"]
            if hasattr(scaler, "feature_names_in_"):
                expected_cols = list(scaler.feature_names_in_)
    except Exception:
        expected_cols = None

    if expected_cols is not None:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features required by the model: {missing}")
        df = df[expected_cols]

    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    out = df.copy()
    out["prediction"] = preds
    if proba is not None:
        out["probability"] = proba
    out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Credit card fraud detection training and inference")
    parser.add_argument("--train", help="Path to training CSV (must include 'Class' column)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to save/load model")
    parser.add_argument("--infer", help="CSV file of transactions for inference (no Class column)")
    parser.add_argument("--save-preds", default="predictions.csv", help="Where to save predictions")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE to training data")
    parser.add_argument("--plots-dir", default="plots", help="Directory to save ROC and confusion matrix plots")

    args = parser.parse_args()

    if args.train:
        print(f"Loading training data from: {args.train}")
        X, y = load_data(args.train)
        print(f"Training samples: {len(y)}, positive class ratio: {y.mean():.4f}")
        model, metrics = build_and_train(X, y, use_smote=args.use_smote, plots_dir=args.plots_dir)
        print("Training completed.")
        print("Best params:", metrics.get("best_params"))
        print("ROC AUC:", metrics.get("roc_auc"))
        save_model(model, args.model)
        print(f"Model saved to {args.model}")
    elif args.infer:
        run_inference(args.model, args.infer, args.save_preds)
    else:
        print("No action specified. Use --train or --infer.")


if __name__ == "__main__":
    main()
