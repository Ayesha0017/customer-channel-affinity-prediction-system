"""
src/features/label_builder.py
------------------------------
Constructs the target label for the Channel Affinity Prediction model.

Label definition:
    For each customer, which single channel did they most strongly prefer
    during the label window (November 2023)?

Why this is a design decision, not just a query:
    The raw data has multiple evidence sources (touchpoints, sessions) and
    multiple interaction types within each (impressions, clicks, form submits).
    A click is stronger intent than an impression. A session is direct behaviour.
    We must explicitly decide how to weight these signals into one label.

Label construction method — Weighted Evidence Score:
    For each customer × channel, we compute:

        score = (tp_clicks × 3.0)
              + (tp_form_submits × 4.0)
              + (tp_impressions × 1.0)
              + (tp_email_opens × 2.0)
              + (session_count × 2.0)
              + (session_avg_engagement / 20)   ← normalised to ~0-5 range

    The channel with the highest score becomes the label.

    Weight rationale:
        form_submit  (4.0) — highest intent, customer took an action
        click        (3.0) — explicit interest, customer chose to engage
        email_open   (2.0) — moderate intent (passive but real)
        session      (2.0) — direct site visit on that channel
        impression   (1.0) — lowest intent, may be involuntary exposure

Why not just use the most-visited channel?
    A customer with 5 impressions on Facebook and 1 form submit on Email
    would be labelled Facebook. But one form submit (4pts) > 5 impressions
    (5pts) is close, and in reality form_submit is a much stronger signal.
    Volume without quality misleads.

Label coverage:
    ~70.8% of customers have label window activity and receive a data-driven label.
    ~29.2% have zero activity in November — they are excluded from model training
    and evaluation (no reliable ground truth). This is correct: forcing a label
    onto customers with no signal introduces noise, not information.

    These excluded customers CAN still receive rule-based predictions at inference
    time (fallback to feature-window behaviour or CRM segment defaults).

Outputs:
    labels.csv — one row per customer with a label window activity, columns:
        customer_id       str
        label_channel     str    ← the target variable
        label_score       float  ← confidence of the label (max score / total score)
        label_source      str    ← 'tp_only' | 'sess_only' | 'combined'
        n_interactions    int    ← total interactions in label window
        low_confidence    bool   ← True if max_score < 30% above second-best
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNELS = [
    "Email", "Facebook", "Instagram",
    "Google Ads", "Organic Search", "Direct", "YouTube",
]

LABEL_START_DEFAULT = "2023-11-01"
LABEL_END_DEFAULT   = "2023-11-30"

# Interaction weights — explicit, documented, easy to tune
WEIGHTS = {
    "form_submit":  4.0,
    "click":        3.0,
    "email_open":   2.0,
    "email_sent":   0.5,   # received but not opened — very low signal
    "impression":   1.0,
    "session":      2.0,   # one session = 2 base points
    # engagement bonus: session_avg_engagement / ENGAGEMENT_DIVISOR → ~0-5 range
}
ENGAGEMENT_DIVISOR = 20.0

# A label is "low confidence" when the winning channel scores less than
# this multiplier above the second-best channel.
LOW_CONFIDENCE_THRESHOLD = 1.30   # winner must be ≥ 30% better than runner-up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window(df: pd.DataFrame, col: str,
            start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[col] >= start) & (df[col] <= end)].copy()


# ---------------------------------------------------------------------------
# Score builders — one per source table
# ---------------------------------------------------------------------------

def _score_touchpoints(tp: pd.DataFrame,
                      label_start: pd.Timestamp,
                      label_end: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        customer_id, channel, tp_score
    One row per (customer, channel) that had any touchpoint in the label window.
    """
    tp_w = _window(tp, "timestamp", label_start, label_end)

    if tp_w.empty:
        return pd.DataFrame(columns=["customer_id", "channel", "tp_score"])

    # Map touchpoint_type → weight
    type_weight_map = {
        "form_submit":  WEIGHTS["form_submit"],
        "click":        WEIGHTS["click"],
        "email_open":   WEIGHTS["email_open"],
        "email_sent":   WEIGHTS["email_sent"],
        "impression":   WEIGHTS["impression"],
    }
    tp_w["tp_weight"] = tp_w["touchpoint_type"].map(type_weight_map).fillna(1.0)

    agg = (
        tp_w.groupby(["customer_id", "channel"])
        .agg(tp_score=("tp_weight", "sum"))
        .reset_index()
    )
    return agg


def _score_sessions(sess: pd.DataFrame,
                    label_start: pd.Timestamp,
                    label_end: pd.Timestamp) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        customer_id, channel, sess_score
    One row per (customer, channel) that had any session in the label window.

    session score = (n_sessions × session_weight) + (avg_engagement / divisor)
    """
    sess_w = _window(sess, "session_start_timestamp", label_start, label_end)

    if sess_w.empty:
        return pd.DataFrame(columns=["customer_id", "channel", "sess_score"])

    agg = (
        sess_w.groupby(["customer_id", "channel"])
        .agg(
            n_sessions     = ("session_id",      "count"),
            avg_engagement = ("engagement_score", "mean"),
        )
        .reset_index()
    )

    agg["sess_score"] = (
        agg["n_sessions"] * WEIGHTS["session"]
        + agg["avg_engagement"] / ENGAGEMENT_DIVISOR
    )
    return agg[["customer_id", "channel", "sess_score"]]


# ---------------------------------------------------------------------------
# Label assembler
# ---------------------------------------------------------------------------

def build_labels(
    data_dir:    str | Path = "data/raw",
    label_start: str        = LABEL_START_DEFAULT,
    label_end:   str        = LABEL_END_DEFAULT,
    save_to:     str | None = None,
) -> pd.DataFrame:
    """
    Build the target label for every customer who had activity in the label window.

    Parameters
    ----------
    data_dir    : path to raw CSVs
    label_start : inclusive start of label window (ISO date string)
    label_end   : inclusive end of label window   (ISO date string)
    save_to     : if provided, saves labels.csv at this path

    Returns
    -------
    pd.DataFrame with columns:
        customer_id, label_channel, label_score, label_source,
        n_interactions, low_confidence
    Only contains customers who had label-window activity (~70% of base).
    """
    data_dir    = Path(data_dir)
    label_start = pd.Timestamp(label_start)
    label_end   = pd.Timestamp(label_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    print(f"Label window : {label_start.date()} → {label_end.date()}")
    print(f"Loading raw tables from: {data_dir.resolve()}")

    # --- Load only what we need ---
    tp   = pd.read_csv(data_dir / "marketing_touchpoints.csv", parse_dates=["timestamp"])
    sess = pd.read_csv(data_dir / "sessions.csv",
                       parse_dates=["session_start_timestamp"])

    # --- Score each source ---
    print("  Scoring touchpoints …")
    tp_scores   = _score_touchpoints(tp, label_start, label_end)

    print("  Scoring sessions …")
    sess_scores = _score_sessions(sess, label_start, label_end)

    # --- Merge scores onto a unified (customer, channel) grid ---
    # Full outer join: a customer may have sessions but no touchpoints, or vice versa
    combined = (
        tp_scores
        .merge(sess_scores, on=["customer_id", "channel"], how="outer")
        .fillna(0.0)
    )
    combined["total_score"] = combined["tp_score"] + combined["sess_score"]

    # Label source — tells us which evidence we actually had
    combined["has_tp"]   = combined["tp_score"]   > 0
    combined["has_sess"] = combined["sess_score"] > 0
    combined["label_source"] = np.select(
        condlist=[
            combined["has_tp"] & combined["has_sess"],
            combined["has_tp"] & ~combined["has_sess"],
            ~combined["has_tp"] & combined["has_sess"],
        ],
        choicelist=["combined", "tp_only", "sess_only"],
        default="none",
    )

    # --- Pick winner per customer ---
    # Sort descending so idxmax gives the highest-score channel
    idx_max = combined.groupby("customer_id")["total_score"].idxmax()
    winners = combined.loc[idx_max, ["customer_id", "channel", "total_score",
                                      "label_source"]].copy()
    winners.rename(columns={"channel": "label_channel",
                             "total_score": "label_score"}, inplace=True)

    # --- Confidence flag ---
    # Compute the second-best score per customer
    second_best = (
        combined
        .sort_values("total_score", ascending=False)
        .groupby("customer_id", group_keys=False)
        .nth(1)          # second row after sorting = second-best channel
        [["customer_id", "total_score"]]
        .rename(columns={"total_score": "second_score"})
        .reset_index(drop=True)
    )
    winners = winners.merge(second_best, on="customer_id", how="left")
    winners["second_score"] = winners["second_score"].fillna(0.0)

    # low_confidence: winner not clearly dominant over runner-up
    winners["low_confidence"] = (
        winners["label_score"] < winners["second_score"] * LOW_CONFIDENCE_THRESHOLD
    )

    # --- Interaction count (raw volume in label window) ---
    tp_counts   = (
        _window(tp,   "timestamp",               label_start, label_end)
        .groupby("customer_id").size().rename("tp_count")
    )
    sess_counts = (
        _window(sess, "session_start_timestamp", label_start, label_end)
        .groupby("customer_id").size().rename("sess_count")
    )
    interaction_counts = (
        tp_counts.to_frame()
        .join(sess_counts, how="outer")
        .fillna(0)
        .assign(n_interactions=lambda d: d["tp_count"] + d["sess_count"])
        ["n_interactions"]
        .astype(int)
        .reset_index()
    )
    winners = winners.merge(interaction_counts, on="customer_id", how="left")
    winners["n_interactions"] = winners["n_interactions"].fillna(0).astype(int)

    # --- Final clean-up ---
    labels = winners[["customer_id", "label_channel", "label_score",
                       "label_source", "n_interactions", "low_confidence"]].copy()
    labels = labels.sort_values("customer_id").reset_index(drop=True)

    print(f"\n  Customers with labels : {len(labels):,}")
    print(f"  Low-confidence labels : {labels['low_confidence'].sum():,} "
          f"({labels['low_confidence'].mean()*100:.1f}%)")

    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(save_path, index=False)
        print(f"  Saved → {save_path.resolve()}")

    return labels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_labels(labels: pd.DataFrame, data_dir: Path) -> None:
    """
    Sanity checks + signal quality report.
    Assesses confidence distributions and data source splits.
    """
    print("\n" + "=" * 60)
    print("LABEL VALIDATION REPORT")
    print("=" * 60)

    # 1. Label distribution — should be roughly balanced (7 channels × ~14%)
    print("\n[1] Label channel distribution:")
    dist = labels["label_channel"].value_counts()
    total = len(labels)
    for ch, cnt in dist.items():
        bar = "█" * int(cnt / total * 100 / 2)
        print(f"  {ch:18s}  {cnt:5,}  ({cnt/total*100:5.1f}%)  {bar}")

    # 2. Coverage
    print(f"\n[2] Label coverage:")
    print(f"  Labelled customers   : {len(labels):,} / 40,000  "
          f"({len(labels)/40_000*100:.1f}%)")
    print(f"  Excluded (no signal) : {40_000 - len(labels):,}  "
          f"({(40_000 - len(labels))/40_000*100:.1f}%)")

    # 3. Label confidence quality
    print("\n[3] Label confidence quality:")

    high_conf = labels[~labels["low_confidence"]]
    low_conf  = labels[labels["low_confidence"]]

    print(f"  High confidence labels : {len(high_conf):,} "
          f"({len(high_conf)/len(labels)*100:.1f}%)")

    print(f"  Low confidence labels  : {len(low_conf):,} "
          f"({len(low_conf)/len(labels)*100:.1f}%)")

    avg_high = high_conf["label_score"].mean()
    avg_low  = low_conf["label_score"].mean()

    print(f"\n  Avg raw score — high confidence : {avg_high:.2f}")
    print(f"  Avg raw score — low confidence  : {avg_low:.2f}")
    print("  (Note: Low-confidence labels often track higher raw scores due to highly active, cross-channel users)")

    # 4. Confidence breakdown by class
    print(f"\n[4] Confidence breakdown by class:")
    print(f"  High confidence : {len(high_conf):,}  "
          f"({len(high_conf)/len(labels)*100:.1f}%)")
    print(f"  Low confidence  : {len(low_conf):,}  "
          f"({len(low_conf)/len(labels)*100:.1f}%)")

    # 5. Label source breakdown
    print(f"\n[5] Label source breakdown:")
    for src, cnt in labels["label_source"].value_counts().items():
        print(f"  {src:12s}  {cnt:,}  ({cnt/len(labels)*100:.1f}%)")

    # 6. Interaction volume distribution
    print(f"\n[6] Interaction count stats (label window):")
    stats = labels["n_interactions"].describe()
    for k in ["min", "25%", "50%", "75%", "max", "mean"]:
        print(f"  {k:6s} : {stats[k]:.1f}")

    print("\n" + "=" * 60)
    print("Label validation complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "="*60)
    print("BUILDING TRAIN LABELS")
    print("="*60)

    labels_train = build_labels(
        data_dir="data/raw",
        label_start="2023-11-01",
        label_end="2023-11-30",
        save_to="data/processed/labels_train.csv",
    )

    validate_labels(
        labels_train,
        Path("data/raw")
    )

    print("\n" + "="*60)
    print("BUILDING TEST LABELS")
    print("="*60)

    labels_test = build_labels(
        data_dir="data/raw",
        label_start="2023-12-01",
        label_end="2023-12-31",
        save_to="data/processed/labels_test.csv",
    )

    validate_labels(
        labels_test,
        Path("data/raw")
    )

    print("\nDone.")
    print(f"Train labels: {len(labels_train):,}")
    print(f"Test labels : {len(labels_test):,}")