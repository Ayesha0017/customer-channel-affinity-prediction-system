"""
src/models/ml_model.py
-----------------------
LightGBM-based channel affinity predictor.

Architecture:
    Multi-class classification — one model, 7 output classes (one per channel).
    LightGBM handles sparse structural behaviors natively and scales effortlessly 
    without arbitrary transformations.

Changes Implemented:
    - Fixed drop columns and introduced a comprehensive LEAKAGE_COLS guard.
    - Right-sized model complexity (n_estimators=300, num_leaves=31) to curb overfitting.
    - Configured strategic 5-fold Stratified CV with validation-set early stopping.
    - Standardized predictions schema to rank_1, rank_2, rank_3, and margin for 
      seamless interface integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNELS = [
    "Email", "Facebook", "Instagram",
    "Google Ads", "Organic Search", "Direct", "YouTube",
]

CAT_FEATURES = [
    "customer_segment",
    "acquisition_channel",
    "customer_status",
    "gender",
]

DROP_COLS = ["customer_id"]

LEAKAGE_COLS = [
    "label_channel",
    "label_score",
    "runner_up_channel",
    "runner_up_score",
    "margin",
    "low_confidence",
    "n_interactions",
]

LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         7,
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      300,  # Right-sized from 500 to optimize capacity
    "learning_rate":     0.05,
    "num_leaves":        31,   # Adjusted down from 63 to optimize generalization
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "class_weight":      "balanced",
    "random_state":      42,
    "verbose":           -1,
    "n_jobs":            -1,
}


# ---------------------------------------------------------------------------
# Data Preparation Helpers
# ---------------------------------------------------------------------------

def _prepare_data(features: pd.DataFrame, labels: pd.DataFrame, high_confidence_only: bool = True):
    """Merges features and targets, casting categorical arrays safely while dropping leakage risk flags."""
    if high_confidence_only:
        labels = labels[~labels["low_confidence"]].copy()
    
    merged = features.merge(
        labels[["customer_id", "label_channel"]],
        on="customer_id", how="inner",
    )
    
    le = LabelEncoder().fit(CHANNELS)
    y = pd.Series(le.transform(merged["label_channel"]), name="label")
    
    # Isolate targets and wipe out any leakage hazards
    all_drops = DROP_COLS + LEAKAGE_COLS
    drop = [c for c in all_drops if c in merged.columns]
    X = merged.drop(columns=drop)
    
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
            
    return X, y, le


# ---------------------------------------------------------------------------
# Core Model Class
# ---------------------------------------------------------------------------

class ChannelAffinityModel:

    def __init__(self, params: Optional[dict] = None):
        self.params        = {**LGBM_PARAMS, **(params or {})}
        self.model         = None
        self.le            = LabelEncoder().fit(CHANNELS)
        self.feature_names = None
        self.metrics       = {}

    def train(self, features_path: str | Path, labels_path: str | Path, high_confidence_only: bool = True):
        print("Loading training artifacts ...")
        features = pd.read_csv(features_path)
        labels   = pd.read_csv(labels_path)

        X, y, le = _prepare_data(features, labels, high_confidence_only)
        self.le            = le
        self.feature_names = X.columns.tolist()

        print(f"Training on {len(X):,} customers x {X.shape[1]} features")
        print("Label distribution:")
        for ch, cnt in pd.Series(le.inverse_transform(y)).value_counts().items():
            print(f"  {ch:20s}  {cnt:,}")

        print("\nRunning Stratified 5-fold cross-validation with early stopping ...")
        cv_scores = self._cross_validate(X, y, n_splits=5)
        print(f"  CV accuracy: {np.mean(cv_scores)*100:.2f}% +/- {np.std(cv_scores)*100:.2f}%")
        self.metrics["cv_accuracy_mean"] = float(np.mean(cv_scores))
        self.metrics["cv_accuracy_std"]  = float(np.std(cv_scores))

        print("\nTraining final model on full high-confidence set ...")
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X, y,
            categorical_feature=[c for c in CAT_FEATURES if c in X.columns],
        )
        print("Training complete.")
        return self

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[float]:
        skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores   = []
        cat_cols = [c for c in CAT_FEATURES if c in X.columns]
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            m = lgb.LGBMClassifier(**self.params)
            m.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                categorical_feature=cat_cols
            )
            
            preds = m.predict(X_va)
            acc   = accuracy_score(y_va, preds)
            scores.append(acc)
            print(f"    Fold {fold} (Trees: {m.best_iteration_ or self.params['n_estimators']}): {acc*100:.2f}%")
        return scores

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        feat = features.copy()
        if "customer_id" not in feat.columns:
            feat = feat.reset_index()
        customer_ids = feat["customer_id"].values
        
        X     = self._prepare_inference(feat)
        proba = self.model.predict_proba(X)

        # Multi-rank resolution across classes
        sorted_indices = np.argsort(-proba, axis=1)
        
        rank_1 = self.le.inverse_transform(sorted_indices[:, 0])
        rank_2 = self.le.inverse_transform(sorted_indices[:, 1])
        rank_3 = self.le.inverse_transform(sorted_indices[:, 2])
        
        confidence = np.take_along_axis(proba, sorted_indices[:, [0]], axis=1).squeeze().round(4)
        
        # Pull exact localized uncertainty spread via winner vs runner-up margins
        sorted_proba = np.sort(proba, axis=1)
        winner        = sorted_proba[:, -1]
        runner_up     = sorted_proba[:, -2]
        margin        = (winner - runner_up).round(4)

        results = pd.DataFrame({
            "customer_id": customer_ids,
            "rank_1":      rank_1,
            "rank_2":      rank_2,
            "rank_3":      rank_3,
            "confidence":  confidence,
            "margin":      margin,
        })
        
        # Append soft probability allocations for full backend visibility
        for i, ch in enumerate(self.le.classes_):
            results[f"prob_{ch.replace(' ','_')}"] = proba[:, i].round(4)
        return results

    def _prepare_inference(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Strip matching columns, align categorical definitions and guard against missing attributes."""
        all_drops = DROP_COLS + LEAKAGE_COLS
        drop = [c for c in all_drops if c in feat.columns]
        X    = feat.drop(columns=drop, errors="ignore")
        
        for col in CAT_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype("category")
                
        if self.feature_names:
            for col in set(self.feature_names) - set(X.columns):
                X[col] = 0.0
            X = X[self.feature_names]
        return X

    def evaluate(self, features: pd.DataFrame, labels: pd.DataFrame, high_confidence_only: bool = True) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")
            
        lbl = labels[~labels["low_confidence"]].copy() if high_confidence_only else labels.copy()
        feat_merged = features.merge(
            lbl[["customer_id", "label_channel"]], on="customer_id", how="inner"
        )
        
        le_tmp = LabelEncoder().fit(CHANNELS)
        y_true = pd.Series(le_tmp.transform(feat_merged["label_channel"]), name="label")
        X      = self._prepare_inference(feat_merged)
        
        proba  = self.model.predict_proba(X)
        y_pred = proba.argmax(axis=1)

        top3_idx = np.argsort(proba, axis=1)[:, -3:]
        top3_acc = np.mean([y_true.iloc[i] in top3_idx[i] for i in range(len(y_true))])

        acc    = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=self.le.classes_,
                                       output_dict=True,
                                       zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "n_evaluated":            len(y_true),
            "accuracy":               round(acc, 4),
            "top3_accuracy":          round(top3_acc, 4),
            "lift_over_baseline":     round(acc * 7, 2),
            "random_baseline":        round(1 / 7, 4),
            "per_class_f1":           {ch: round(report[ch]["f1-score"], 4) for ch in CHANNELS},
            "macro_f1":               round(report["macro avg"]["f1-score"], 4),
            "weighted_f1":            round(report["weighted avg"]["f1-score"], 4),
            "confusion_matrix":       cm.tolist(),
            "confusion_matrix_labels": CHANNELS,
        }
        
        if "cv_accuracy_mean" in self.metrics:
            metrics["cv_accuracy_mean"] = self.metrics["cv_accuracy_mean"]
            metrics["cv_accuracy_std"]  = self.metrics["cv_accuracy_std"]
            
        self.metrics.update(metrics)
        return metrics

    def feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return pd.DataFrame({
            "feature": self.feature_names,
            "gain":    self.model.booster_.feature_importance(importance_type="gain"),
            "split":   self.model.booster_.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False).head(top_n).reset_index(drop=True)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model":         self.model,
            "le":            self.le,
            "feature_names": self.feature_names,
            "metrics":       self.metrics,
            "params":        self.params,
        }, path)
        print(f"Model saved -> {path}")

    @classmethod
    def load(cls, path: str | Path):
        data = joblib.load(path)
        obj  = cls(params=data["params"])
        obj.model         = data["model"]
        obj.le            = data["le"]
        obj.feature_names = data["feature_names"]
        obj.metrics       = data["metrics"]
        print(f"Model loaded <- {path}")
        return obj


# ---------------------------------------------------------------------------
# CLI Execution Engine
# ---------------------------------------------------------------------------

def _print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("ML MODEL — HOLDOUT EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Evaluated on      : {metrics['n_evaluated']:,} customers")
    print(f"  Top-1 accuracy    : {metrics['accuracy']*100:.1f}%")
    print(f"  Top-3 accuracy    : {metrics['top3_accuracy']*100:.1f}%")
    print(f"  Random baseline   : {100/7:.1f}%")
    print(f"  Lift              : {metrics['lift_over_baseline']:.1f}x")
    print(f"  Macro F1          : {metrics['macro_f1']:.3f}")
    if "cv_accuracy_mean" in metrics:
        print(f"  CV accuracy       : {metrics['cv_accuracy_mean']*100:.2f}%"
              f" +/- {metrics['cv_accuracy_std']*100:.2f}%")
    print(f"\n  Per-channel F1:")
    for ch, f1 in metrics["per_class_f1"].items():
        bar = "█" * int(f1 * 40)
        print(f"  {ch:20s}  {f1:.3f}  {bar}")
    print("\n  Confusion matrix (rows=True, cols=Predicted):")
    cm  = np.array(metrics["confusion_matrix"])
    lbs = metrics["confusion_matrix_labels"]
    print("  " + "".join(f"{l[:8]:>10s}" for l in lbs))
    for i, row in enumerate(cm):
        print(f"  {lbs[i]:18s}" + "".join(f"{v:>10d}" for v in row))
    print("=" * 60)


if __name__ == "__main__":
    TRAIN_FEATURES = "data/processed/features_train.csv"
    TRAIN_LABELS   = "data/processed/labels_train.csv"
    
    TEST_FEATURES  = "data/processed/features_test.csv"
    TEST_LABELS    = "data/processed/labels_test.csv"
    
    MODEL_OUT = "models/lgbm_channel_affinity.joblib"
    PRED_OUT  = "data/processed/ml_predictions.csv"
    METR_OUT  = "data/processed/ml_metrics.csv"

    model = ChannelAffinityModel()
    model.train(
        TRAIN_FEATURES,
        TRAIN_LABELS,
        high_confidence_only=True
    )

    train_features = pd.read_csv(TRAIN_FEATURES)
    
    test_features = pd.read_csv(TEST_FEATURES)
    test_labels   = pd.read_csv(TEST_LABELS)
    
    metrics = model.evaluate(
        test_features,
        test_labels
    )
    _print_metrics(metrics)

    print("\nTop 15 features by gain:")
    imp = model.feature_importance(top_n=15)
    for _, row in imp.iterrows():
        bar = "█" * int(row["gain"] / imp["gain"].max() * 30)
        print(f"  {row['feature']:45s}  {bar}")

    model.save(MODEL_OUT)

    print("\nGenerating unified tracking predictions for all customers ...")
    preds = model.predict(test_features)
    preds.to_csv(PRED_OUT, index=False)
    print(f"Saved predictions -> {PRED_OUT}")

    # Export a tabular tracking row for the comparative evaluation framework
    pd.DataFrame([{
        "accuracy": metrics["accuracy"],
        "top3_accuracy": metrics["top3_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "cv_accuracy": metrics.get("cv_accuracy_mean", np.nan)
    }]).to_csv(METR_OUT, index=False)
    print(f"Saved tracking metrics -> {METR_OUT}")

    print("\nChannel distribution of ML recommendations (rank_1):")
    for ch, cnt in preds["rank_1"].value_counts().items():
        print(f"  {ch:20s}  {cnt:,}  ({cnt/len(preds)*100:.1f}%)")