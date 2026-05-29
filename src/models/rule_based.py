"""
Rule-based channel affinity scorer.
Computes a weighted sum of normalized signals to predict the best channel.

Public API:
    score(features) -> predictions DataFrame
    evaluate(predictions, labels) -> metrics dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Config & Weights
# ---------------------------------------------------------------------------

CHANNELS = [
    "Email", "Facebook", "Instagram",
    "Google Ads", "Organic Search", "Direct", "YouTube"
]

# Quick lookup map for dataframe columns (e.g., 'Google Ads' -> 'Google_Ads')
CHANNEL_KEYS = {ch: ch.replace(" ", "_") for ch in CHANNELS}

# Weights optimized via correlation analysis in 02_feature_exploration
WEIGHTS = {
    "tp_count":    3.0,  # volume signal (noisy)
    "tp_clicks":   4.0,  # strong intent signal
    "sess_count":  3.0,  # volume signal
    "sess_eng":    4.0,  # quality signal (weighted)
    "recency_inv": 2.5,  # bumped from 2.0 to prioritize recent activity
    "momentum":    1.5,  # 30d vs full-window velocity
}

MAX_RAW_SCORE = sum(WEIGHTS.values())  # 18.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_raw_signals(features: pd.DataFrame, ch_key: str) -> pd.DataFrame:
    """Extracts the 6 baseline signals for a given channel key."""
    col_map = {
        "tp_count":    f"tp_{ch_key}_count",
        "tp_clicks":   f"tp_{ch_key}_clicks",
        "sess_count":  f"sess_{ch_key}_count",
        "sess_eng":    f"sess_{ch_key}_weighted_eng",
        "recency_inv": f"recency_{ch_key}_tp_inv",
        "momentum":    f"trend_{ch_key}_momentum",
    }
    
    out = pd.DataFrame(index=features.index)
    for signal, col in col_map.items():
        # Fallback to 0.0 if column doesn't exist (handles partial feature tables safely)
        out[signal] = features[col].values if col in features.columns else 0.0
    return out


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scaling to [0, 1]. Safe against zero-division."""
    out = df.copy()
    for col in out.columns:
        mx = out[col].max()
        if mx > 0:
            out[col] = out[col] / mx
    return out


def _compute_score_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Generates an (n_customers x n_channels) matrix of raw weighted scores."""
    scores = {}
    for ch in CHANNELS:
        key = CHANNEL_KEYS[ch]
        raw = _get_raw_signals(features, key)
        norm = _normalise(raw)
        scores[ch] = sum(norm[sig] * w for sig, w in WEIGHTS.items())

    return pd.DataFrame(scores, index=features.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(features: pd.DataFrame) -> pd.DataFrame:
    """Scores all customers and ranks channel recommendations."""
    df = features.copy()
    if "customer_id" not in df.columns:
        df = df.reset_index()
    cust_ids = df["customer_id"].values

    # Drop ID to avoid feeding it into matrix math
    score_in = df.drop(columns=["customer_id"], errors="ignore")
    matrix = _compute_score_matrix(score_in)

    # Bring scores down to a clean 0-1 range
    matrix_norm = matrix / MAX_RAW_SCORE

    # Row-by-row sort to get ranked channels (best first)
    ranked = matrix_norm.apply(
        lambda r: pd.Series(
            r.sort_values(ascending=False).index.tolist(),
            index=[f"rank_{i}" for i in range(1, 8)]
        ),
        axis=1
    )

    # Confidence calculation (winner / total weight across all channels)
    total_scores = matrix_norm.sum(axis=1).replace(0, np.nan)
    winner_score = matrix_norm.max(axis=1)
    confidence = (winner_score / total_scores).fillna(1 / len(CHANNELS))

    # Margin metrics for downstream UI stability checks (winner vs runner-up)
    sorted_arr = np.sort(matrix_norm.values, axis=1)
    margin = sorted_arr[:, -1] - sorted_arr[:, -2]

    # Assemble output payload
    res = pd.DataFrame({"customer_id": cust_ids})
    res = pd.concat([res, ranked.reset_index(drop=True)], axis=1)
    res["confidence"] = confidence.values.round(4)
    res["margin"] = margin.round(4)

    # Track absolute per-channel scores for explainability engines
    for ch in CHANNELS:
        key = CHANNEL_KEYS[ch]
        res[f"score_{key}"] = matrix_norm[ch].values.round(4)

    return res


def evaluate(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    top_k: int = 3,
    high_confidence_only: bool = True,
) -> dict:
    """Evaluates rule-based model against target ground-truth data."""
    # Filter out noisy, low-confidence interactions if flag is set
    lbls = labels[~labels["low_confidence"]].copy() if high_confidence_only else labels.copy()

    merged = predictions.merge(
        lbls[["customer_id", "label_channel"]],
        on="customer_id",
        how="inner"
    )

    if len(merged) == 0:
        raise ValueError("No matching customer IDs found between predictions and labels.")

    y_true = merged["label_channel"].values
    y_pred = merged["rank_1"].values

    # Overall metrics
    acc = (y_true == y_pred).mean()

    # Top-K check
    top_k_cols = [f"rank_{i}" for i in range(1, top_k + 1)]
    in_top_k = merged.apply(lambda r: r["label_channel"] in r[top_k_cols].values, axis=1)
    top_k_acc = in_top_k.mean()

    # Slice recall by channel
    per_ch = {}
    for ch in CHANNELS:
        mask = y_true == ch
        if mask.sum() == 0:
            continue
        per_ch[ch] = float((y_pred[mask] == ch).mean())

    cm = confusion_matrix(y_true, y_pred, labels=CHANNELS)
    cm_df = pd.DataFrame(
        cm, 
        index=pd.Index(CHANNELS, name="True"),
        columns=pd.Index(CHANNELS, name="Predicted")
    )

    report = classification_report(
        y_true, y_pred,
        labels=CHANNELS,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": float(acc),
        "top_k_accuracy": float(top_k_acc),
        "top_k": top_k,
        "per_channel": per_ch,
        "confusion_matrix": cm_df,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_results(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("RULE-BASED SCORER — EVALUATION")
    print("=" * 60)
    print(f"\nTop-1 accuracy  : {metrics['accuracy']*100:.1f}%")
    print(f"Top-{metrics['top_k']} accuracy  : {metrics['top_k_accuracy']*100:.1f}%")
    print(f"Random baseline : {100/7:.1f}%")
    print(f"Lift             : {metrics['accuracy']*7:.1f}x")
    print(f"\nPer-channel recall:")
    for ch, acc in metrics["per_channel"].items():
        bar = chr(9608) * int(acc * 50)
        flag = "✓" if acc > 1/7 else "✗"
        print(f"  [{flag}] {ch:20s}  {acc*100:5.1f}%  {bar}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    features = pd.read_csv("data/processed/features_test.csv")
    labels = pd.read_csv("data/processed/labels_test.csv")

    print(f"Scoring {len(features)} customers across {len(CHANNELS)} channels ...")
    predictions = score(features)

    print(f"Predictions shape : {predictions.shape}")
    print(f"\nChannel distribution (rank_1):")
    dist = predictions["rank_1"].value_counts()
    for ch, cnt in dist.items():
        print(f"  {ch:20s}  {cnt}  ({cnt/len(predictions)*100:.1f}%)")

    print("\nUsing temporal holdout:")
    print("Features: Jul-Nov 2023")
    print("Labels:   Dec 2023")

    print("\nEvaluating against holdout labels ...")
    metrics = evaluate(predictions, labels)
    _print_results(metrics)

    # Save outputs
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_out = out_dir / "rule_based_predictions_test.csv"
    predictions.to_csv(preds_out, index=False)
    print(f"\nSaved predictions -> {preds_out}")

    metrics_df = pd.DataFrame([{
        "accuracy": metrics["accuracy"],
        f"top{metrics['top_k']}_accuracy": metrics["top_k_accuracy"]
    }])
    metrics_out = out_dir / "rule_based_metrics_test.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved baseline metrics -> {metrics_out}")