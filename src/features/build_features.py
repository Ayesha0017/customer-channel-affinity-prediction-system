"""
src/features/build_features.py
-------------------------------
Builds the customer-level feature table used for both rule-based scoring
and ML model training.

Design principles:
  1. Single responsibility — this file ONLY builds features. No modelling,
     no label logic, no plotting.

  2. Strict temporal isolation — every aggregation is filtered to the
     feature window BEFORE any join. No future data can leak in.

  3. One flat output — a single DataFrame indexed by customer_id where
     every row is one customer and every column is one feature.
     Downstream code never touches the raw tables again.

  4. Explicit naming — every column is named <source>_<channel/platform>_<metric>
     so the origin is always obvious. No double-prefix accidents.

  5. Sparse-safe — customers who have zero activity in a source get 0,
     not NaN. Missing values are a modelling choice, not an oversight.

Feature groups produced:
  A. customer_profile      — demographics + CRM fields (no temporal filter needed)
  B. touchpoint_* — per-channel counts, clicks, impressions, revenue (feature window)
  C. session_* — per-channel session counts + avg engagement score (feature window)
  D. email_* — open rate, click rate, time-to-open (feature window)
  E. social_* — per-platform engagement counts + avg sentiment (feature window)
  F. content_* — video watch time, completion %, content type mix (feature window)
  G. recency_* — days since last touchpoint per channel (feature window)
  H. trend_* — recent 30d count / full-window count ratio per channel
                             (captures momentum without leaking label window data)

Usage:
    from src.features.build_features import build_feature_table

    features = build_feature_table(
        data_dir   = "data/raw",
        feat_start = "2023-06-01",
        feat_end   = "2023-10-31",   # inclusive — no overlap with label window
    )
    # features: DataFrame, shape (n_customers, n_features), index = customer_id
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

SOCIAL_PLATFORMS = ["Facebook", "Instagram", "Twitter", "YouTube", "LinkedIn"]

CONTENT_TYPES = ["blog", "ebook", "case_study", "webinar", "video"]

# ---------------------------------------------------------------------------
# Explicit categorical columns
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "customer_segment",
    "acquisition_channel",
    "customer_status",
    "gender",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load(data_dir: Path, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV from data/raw/, parse timestamp columns."""
    fpath = data_dir / f"{name}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Expected raw file not found: {fpath}")
    return pd.read_csv(fpath, parse_dates=parse_dates or [])


def _window(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows where df[col] falls in [start, end] inclusive."""
    return df[(df[col] >= start) & (df[col] <= end)].copy()


# ---------------------------------------------------------------------------
# Feature group builders
# Each function returns a DataFrame indexed by customer_id.
# ---------------------------------------------------------------------------

def _build_customer_profile(customers: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Group A — static CRM fields.

    NOTE: primary_channel is intentionally EXCLUDED — it must not be a model
    feature. The model must infer preference from behaviour, not read it directly.

    recency_days: days since last purchase as of the feature window end date.
    Customers who never purchased get a large recency (999 + days since joining)
    to clearly distinguish them from recent buyers.
    """
    cust = customers.copy()
    cust["acquisition_date"]   = pd.to_datetime(cust["acquisition_date"])
    cust["last_purchase_date"] = pd.to_datetime(cust["last_purchase_date"])

    cust["recency_days"] = (as_of - cust["last_purchase_date"]).dt.days
    never_bought = cust["recency_days"].isna()
    cust.loc[never_bought, "recency_days"] = (
        (as_of - cust.loc[never_bought, "acquisition_date"]).dt.days + 999
    )
    cust["recency_days"] = cust["recency_days"].clip(lower=0)

    # tenure: days since acquisition (more tenure → more behavioural signal available)
    cust["tenure_days"] = (as_of - cust["acquisition_date"]).dt.days.clip(lower=0)

    profile_cols = [
        "customer_id",
        "age",
        "total_revenue",
        "total_orders",
        "avg_order_value",
        "lifetime_value",
        "recency_days",
        "tenure_days",
        # categorical — kept for one-hot encoding downstream
        "customer_segment",
        "acquisition_channel",
        "customer_status",
        "gender",
    ]
    return cust[profile_cols].set_index("customer_id")


def _build_touchpoint_features(
    tp: pd.DataFrame,
    all_customer_ids: pd.Index,
    feat_start: pd.Timestamp,
    feat_end: pd.Timestamp,
    recent_days: int = 30,
) -> pd.DataFrame:
    """
    Group B — marketing_touchpoints, feature window only.

    Per channel:
        tp_{ch}_count          total touchpoints
        tp_{ch}_clicks         click-type touchpoints
        tp_{ch}_impressions    impression-type touchpoints
        tp_{ch}_revenue        total revenue attributed
        tp_{ch}_click_rate     clicks / count  (0 if no touchpoints)

    Group H (trend) — also computed here since it uses the same source:
        trend_{ch}_momentum    (last-30d count) / (full-window count)
                               Values > 0.5 mean accelerating activity recently.
                               0 when no full-window activity.
    """
    tp_w = _window(tp, "timestamp", feat_start, feat_end)

    recent_start = feat_end - pd.Timedelta(days=recent_days - 1)
    tp_recent = _window(tp, "timestamp", recent_start, feat_end)

    # Aggregate full window per (customer, channel)
    agg = (
        tp_w.groupby(["customer_id", "channel"])
        .agg(
            count       = ("touchpoint_id",   "count"),
            clicks      = ("touchpoint_type", lambda s: (s == "click").sum()),
            impressions = ("touchpoint_type", lambda s: (s == "impression").sum()),
            revenue     = ("revenue_attributed", "sum"),
        )
        .reset_index()
    )
    agg["click_rate"] = np.where(
        agg["count"] > 0, agg["clicks"] / agg["count"], 0.0
    )

    # Aggregate recent window per (customer, channel) — for momentum
    agg_recent = (
        tp_recent.groupby(["customer_id", "channel"])
        .agg(recent_count=("touchpoint_id", "count"))
        .reset_index()
    )

    agg = agg.merge(agg_recent, on=["customer_id", "channel"], how="left")
    agg["recent_count"] = agg["recent_count"].fillna(0)
    agg["momentum"] = np.where(
        agg["count"] > 0, agg["recent_count"] / agg["count"], 0.0
    )

    # Pivot to wide format — one row per customer, one column per (channel × metric)
    frames = {}
    for ch in CHANNELS:
        ch_key = ch.replace(" ", "_")
        sub = agg[agg["channel"] == ch].set_index("customer_id")
        frames[f"tp_{ch_key}_count"]       = sub["count"]
        frames[f"tp_{ch_key}_clicks"]      = sub["clicks"]
        frames[f"tp_{ch_key}_impressions"] = sub["impressions"]
        frames[f"tp_{ch_key}_revenue"]     = sub["revenue"]
        frames[f"tp_{ch_key}_click_rate"]  = sub["click_rate"]
        frames[f"trend_{ch_key}_momentum"] = sub["momentum"]

    wide = pd.DataFrame(frames, index=all_customer_ids).fillna(0.0)
    return wide


def _build_session_features(
    sess: pd.DataFrame,
    all_customer_ids: pd.Index,
    feat_start: pd.Timestamp,
    feat_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Group C — sessions, feature window only.

    Per channel:
        sess_{ch}_count          number of sessions
        sess_{ch}_avg_engagement average engagement score (0–100)
        sess_{ch}_avg_duration   average session duration in seconds
        sess_{ch}_bounce_rate    fraction of sessions that bounced

    Derived:
        sess_{ch}_weighted_eng   count × avg_engagement  (volume × quality signal)
    """
    sess_w = _window(sess, "session_start_timestamp", feat_start, feat_end)

    agg = (
        sess_w.groupby(["customer_id", "channel"])
        .agg(
            count          = ("session_id",             "count"),
            avg_engagement = ("engagement_score",        "mean"),
            avg_duration   = ("session_duration_seconds","mean"),
            bounce_rate    = ("bounce_rate",             "mean"),
        )
        .reset_index()
    )
    agg["weighted_eng"] = agg["count"] * agg["avg_engagement"]

    frames = {}
    for ch in CHANNELS:
        ch_key = ch.replace(" ", "_")
        sub = agg[agg["channel"] == ch].set_index("customer_id")
        frames[f"sess_{ch_key}_count"]          = sub["count"]
        frames[f"sess_{ch_key}_avg_engagement"] = sub["avg_engagement"]
        frames[f"sess_{ch_key}_avg_duration"]   = sub["avg_duration"]
        frames[f"sess_{ch_key}_bounce_rate"]    = sub["bounce_rate"]
        frames[f"sess_{ch_key}_weighted_eng"]   = sub["weighted_eng"]

    wide = pd.DataFrame(frames, index=all_customer_ids).fillna(0.0)
    return wide


def _build_email_features(
    email: pd.DataFrame,
    all_customer_ids: pd.Index,
    feat_start: pd.Timestamp,
    feat_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Group D — email_engagement, feature window only.

    email_open_rate          opened / sent   (proxy for email channel affinity)
    email_click_rate         clicked / sent
    email_opens_total        raw open count  (volume signal)
    email_clicks_total       raw click count
    email_time_to_open_med   median minutes to open (lower = more eager)
    email_time_to_open_inv   1 / (1 + median_minutes)  (higher = more responsive)

    Why open_rate and click_rate separately:
        open_rate measures passive interest (did they open it?)
        click_rate measures intent (did they act on it?)
        Both matter and they're not perfectly correlated.
    """
    email_w = _window(email, "event_timestamp", feat_start, feat_end)
    email_w["time_to_open_minutes"] = pd.to_numeric(
        email_w["time_to_open_minutes"], errors="coerce"
    )

    def _median_safe(s: pd.Series) -> float:
        valid = s.dropna()
        return float(valid.median()) if len(valid) > 0 else np.nan

    agg = (
        email_w.groupby("customer_id")
        .agg(
            total_sent   = ("event_type",            lambda s: (s == "sent").sum()),
            total_opened = ("event_type",            lambda s: (s == "opened").sum()),
            total_clicks = ("click_count",           "sum"),
            opens_count  = ("open_count",            "sum"),
            median_tto   = ("time_to_open_minutes",  _median_safe),
        )
        .reset_index()
    )

    # Impute missing median_tto with the global median (customers who never opened)
    global_median_tto = agg["median_tto"].median()
    agg["median_tto"] = agg["median_tto"].fillna(global_median_tto).fillna(720)

    agg["email_open_rate"]  = np.where(
        agg["total_sent"] > 0, agg["total_opened"] / agg["total_sent"], 0.0
    )
    agg["email_click_rate"] = np.where(
        agg["total_sent"] > 0, agg["total_clicks"] / agg["total_sent"], 0.0
    )
    # Inverse time-to-open: faster response → higher value → stronger email affinity
    agg["email_time_to_open_inv"] = 1.0 / (1.0 + agg["median_tto"])

    out_cols = {
        "email_open_rate":          "email_open_rate",
        "email_click_rate":         "email_click_rate",
        "opens_count":              "email_opens_total",
        "total_clicks":             "email_clicks_total",
        "median_tto":               "email_time_to_open_med",
        "email_time_to_open_inv":   "email_time_to_open_inv",
    }
    result = agg.set_index("customer_id")[list(out_cols.keys())].rename(columns=out_cols)
    return result.reindex(all_customer_ids).fillna(0.0)


def _build_social_features(
    social: pd.DataFrame,
    all_customer_ids: pd.Index,
    feat_start: pd.Timestamp,
    feat_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Group E — social_media_engagement, feature window only.

    Per platform:
        social_{platform}_count          total engagements
        social_{platform}_avg_sentiment  mean sentiment score (0–1)

    Why both count and sentiment:
        count  = volume of interaction (how often they show up on that platform)
        sentiment = quality of interaction (are they positive, which predicts retention)
    """
    social_w = _window(social, "engagement_timestamp", feat_start, feat_end)

    agg = (
        social_w.groupby(["customer_id", "platform"])
        .agg(
            count         = ("social_engagement_id", "count"),
            avg_sentiment = ("sentiment_score",       "mean"),
        )
        .reset_index()
    )

    frames = {}
    for platform in SOCIAL_PLATFORMS:
        p_key = platform.replace(" ", "_")
        sub = agg[agg["platform"] == platform].set_index("customer_id")
        frames[f"social_{p_key}_count"]         = sub["count"]
        frames[f"social_{p_key}_avg_sentiment"] = sub["avg_sentiment"]

    wide = pd.DataFrame(frames, index=all_customer_ids).fillna(0.0)
    return wide


def _build_content_features(
    content: pd.DataFrame,
    all_customer_ids: pd.Index,
    feat_start: pd.Timestamp,
    feat_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Group F — content_engagement, feature window only.

    content_video_watch_seconds    total seconds spent on video content
    content_video_completion_avg   average video completion % (0–1)
    content_video_count            number of video engagements
    content_{type}_count           count per content type (blog, ebook, etc.)

    Why video gets its own metrics:
        YouTube/Instagram personas have meaningfully different video behaviour.
        Raw count is not enough — a customer who watches 10 videos to 90%
        completion is very different from one who watches 2 to 10%.
    """
    content_w = _window(content, "engagement_timestamp", feat_start, feat_end)

    video_mask = (
        (content_w["content_type"].str.lower() == "video") |
        (content_w["engagement_type"].str.lower() == "watch")
    )
    video_w  = content_w[video_mask]
    other_w  = content_w

    # Video-specific aggregation
    video_agg = (
        video_w.groupby("customer_id")
        .agg(
            video_watch_seconds  = ("time_spent_seconds",     "sum"),
            video_completion_avg = ("completion_percentage",  "mean"),
            video_count          = ("engagement_id",          "count"),
        )
        .reset_index()
        .set_index("customer_id")
        .rename(columns={
            "video_watch_seconds":  "content_video_watch_seconds",
            "video_completion_avg": "content_video_completion_avg",
            "video_count":          "content_video_count",
        })
    )

    # Per content-type counts (all types including video)
    # Name: content_{type}_interactions to avoid collision with content_video_count
    type_agg = (
        other_w.groupby(["customer_id", "content_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CONTENT_TYPES, fill_value=0)
    )
    type_agg.columns = [f"content_{ct}_interactions" for ct in type_agg.columns]

    # Combine
    result = pd.DataFrame(index=all_customer_ids)
    result = result.join(video_agg, how="left").fillna(0.0)
    result = result.join(type_agg, how="left").fillna(0.0)

    return result


def _build_recency_features(
    tp: pd.DataFrame,
    sess: pd.DataFrame,
    all_customer_ids: pd.Index,
    feat_start: pd.Timestamp,
    feat_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Group G — recency per channel (days since last interaction in feature window).

    recency_{ch}_days_since_last_tp     days since last touchpoint on that channel
    recency_{ch}_days_since_last_sess   days since last session on that channel

    Customers with NO activity on a channel get feat_window_length days
    (the maximum possible recency — meaning "haven't seen them there at all").
    This is better than NaN or 0 because it's a meaningful signal:
    a customer who hasn't touched Facebook in 5 months is different from
    one who was there yesterday.

    We invert at the end so higher = more recent (consistent with other features
    where "higher = stronger affinity signal").
    """
    max_recency = (feat_end - feat_start).days + 1  # 153 days for our window

    tp_w    = _window(tp,   "timestamp",               feat_start, feat_end)
    sess_w  = _window(sess, "session_start_timestamp",  feat_start, feat_end)

    # Last touchpoint date per (customer, channel)
    tp_last = (
        tp_w.groupby(["customer_id", "channel"])["timestamp"]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "last_tp"})
    )
    tp_last["days_since"] = (feat_end - tp_last["last_tp"]).dt.days

    # Last session date per (customer, channel)
    sess_last = (
        sess_w.groupby(["customer_id", "channel"])["session_start_timestamp"]
        .max()
        .reset_index()
        .rename(columns={"session_start_timestamp": "last_sess"})
    )
    sess_last["days_since"] = (feat_end - sess_last["last_sess"]).dt.days

    frames = {}
    for ch in CHANNELS:
        ch_key = ch.replace(" ", "_")

        sub_tp = tp_last[tp_last["channel"] == ch].set_index("customer_id")["days_since"]
        sub_ss = sess_last[sess_last["channel"] == ch].set_index("customer_id")["days_since"]

        # Reindex to all customers — missing → max_recency
        days_tp = sub_tp.reindex(all_customer_ids).fillna(max_recency)
        days_ss = sub_ss.reindex(all_customer_ids).fillna(max_recency)

        frames[f"recency_{ch_key}_tp_days"]    = days_tp
        frames[f"recency_{ch_key}_sess_days"]  = days_ss
        # Inverse: 1/(1+days) — decays smoothly, recent = high value
        frames[f"recency_{ch_key}_tp_inv"]     = 1.0 / (1.0 + days_tp)
        frames[f"recency_{ch_key}_sess_inv"]   = 1.0 / (1.0 + days_ss)

    return pd.DataFrame(frames, index=all_customer_ids)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_feature_table(
    data_dir: str | Path = "data/raw",
    feat_start: str       = "2023-06-01",
    feat_end: str         = "2023-10-31",
    save_to: str | None   = None,
) -> pd.DataFrame:
    """
    Load all raw tables, apply the feature window, and return one clean
    customer-level feature DataFrame.

    Parameters
    ----------
    data_dir   : path to the folder containing raw CSVs
    feat_start : inclusive start of the feature window (ISO date string)
    feat_end   : inclusive end of the feature window  (ISO date string)
                 MUST be strictly before the label window start.
    save_to    : if provided, saves the feature table as a CSV at this path

    Returns
    -------
    pd.DataFrame  shape (n_customers, n_features), index = customer_id
    """
    data_dir   = Path(data_dir)
    feat_start = pd.Timestamp(feat_start)
    feat_end   = pd.Timestamp(feat_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    as_of      = feat_end  # "today" for recency calculations

    print(f"Feature window: {feat_start.date()} → {feat_end.date()}")
    print(f"Loading raw tables from: {data_dir.resolve()}")

    # --- Load ---
    customers = _load(data_dir, "customers",
                      parse_dates=["acquisition_date", "last_purchase_date"])
    tp        = _load(data_dir, "marketing_touchpoints", parse_dates=["timestamp"])
    sess = _load(
        data_dir,
        "sessions",
        parse_dates=["session_start_timestamp"]
    )
    email     = _load(data_dir, "email_engagement",     parse_dates=["event_timestamp"])
    social    = _load(data_dir, "social_media_engagement", parse_dates=["engagement_timestamp"])
    content   = _load(data_dir, "content_engagement",   parse_dates=["engagement_timestamp"])

    all_customer_ids = pd.Index(customers["customer_id"].values, name="customer_id")

    print(f"Customers: {len(customers):,}")

    # --- Build each feature group ---
    print("  [A] Building customer profile features …")
    profile = _build_customer_profile(customers, as_of)

    print("  [B] Building touchpoint features …")
    touchpoint = _build_touchpoint_features(tp, all_customer_ids, feat_start, feat_end)

    print("  [C] Building session features …")
    session = _build_session_features(sess, all_customer_ids, feat_start, feat_end)

    print("  [D] Building email features …")
    email_feats = _build_email_features(email, all_customer_ids, feat_start, feat_end)

    print("  [E] Building social features …")
    social_feats = _build_social_features(social, all_customer_ids, feat_start, feat_end)

    print("  [F] Building content features …")
    content_feats = _build_content_features(content, all_customer_ids, feat_start, feat_end)

    print("  [G+H] Building recency + momentum features …")
    recency = _build_recency_features(tp, sess, all_customer_ids, feat_start, feat_end)

    # --- Combine ---
    features = (
        profile
        .join(touchpoint,    how="left")
        .join(session,       how="left")
        .join(email_feats,   how="left")
        .join(social_feats,  how="left")
        .join(content_feats, how="left")
        .join(recency,       how="left")
    )

    # Final safety pass — no infs, no NaNs in numeric cols
    num_cols = features.select_dtypes(include=[np.number]).columns

    features[num_cols] = (
        features[num_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Remove zero-variance numeric columns
    zero_var_cols = [
        col for col in num_cols
        if features[col].nunique() <= 1
    ]

    if len(zero_var_cols) > 0:
        print(f"Removing {len(zero_var_cols)} zero-variance columns")
        features = features.drop(columns=zero_var_cols)

    # Refresh numeric columns after removal
    num_cols = features.select_dtypes(include=[np.number]).columns

    # ---------------------------------------------------------------------------
    # Feature metadata
    # ---------------------------------------------------------------------------

    feature_metadata = pd.DataFrame({
        "feature_name": features.columns,
        "dtype": features.dtypes.astype(str).values,
        "is_categorical": [
            col in CATEGORICAL_COLS
            for col in features.columns
        ],
    })

    metadata_path = Path("data/processed/feature_metadata.csv")

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_metadata.to_csv(
        metadata_path,
        index=False,
    )

    print(f"Feature metadata saved → {metadata_path.resolve()}")

    # --- Optionally save ---
    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        features.reset_index().to_csv(save_path, index=False)
        print(f"  Saved → {save_path.resolve()}")

    return features


# ---------------------------------------------------------------------------
# Smoke test — run directly to sanity-check
# ---------------------------------------------------------------------------

def _smoke_test(features: pd.DataFrame, data_dir: Path) -> None:
    """Quick checks that catch the most common feature engineering bugs."""
    customers = pd.read_csv(data_dir / "customers.csv")

    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # 1. Row count
    assert len(features) == len(customers), \
        f"Row count mismatch: {len(features)} features vs {len(customers)} customers"
    print(f"[✓] Row count matches: {len(features):,} customers")

    # 2. No NaNs in numeric columns
    num_cols = features.select_dtypes(include=[np.number]).columns
    nan_count = features[num_cols].isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in numeric columns"
    print(f"[✓] No NaN values in {len(num_cols)} numeric columns")

    # 3. No infinities
    inf_count = np.isinf(features[num_cols].values).sum()
    assert inf_count == 0, f"Found {inf_count} infinite values"
    print(f"[✓] No infinite values")

    # 4. Recency range sanity — channel recency only (not customer-level recency_days,
    #    which can exceed window length for never-purchased customers by design)
    # Channel recency cols have the pattern recency_{Channel}_tp_days / _sess_days
    ch_recency_cols = [c for c in features.columns
                       if c.startswith("recency_") and ("_tp_days" in c or "_sess_days" in c)]
    max_allowed = (pd.Timestamp("2023-10-31") - pd.Timestamp("2023-06-01")).days + 2
    violations = [f"{col}:{features[col].max()}" for col in ch_recency_cols
                  if features[col].max() > max_allowed + 1]
    assert not violations, f"Channel recency out of range: {violations}"
    print(f"\n[✓] Channel recency values in valid range (0–{max_allowed} days)")

    # 5. Feature group coverage
    groups = {
        "touchpoint": [c for c in features.columns if c.startswith("tp_")],
        "session":    [c for c in features.columns if c.startswith("sess_")],
        "email":      [c for c in features.columns if c.startswith("email_")],
        "social":     [c for c in features.columns if c.startswith("social_")],
        "content":    [c for c in features.columns if c.startswith("content_")],
        "recency":    [c for c in features.columns if c.startswith("recency_")],
        "trend":      [c for c in features.columns if c.startswith("trend_")],
    }
    print(f"\n[✓] Feature group counts:")
    for grp, cols in groups.items():
        print(f"  {grp:12s}   {len(cols):3d} columns")

    print("\n" + "=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)


if __name__ == "__main__":

    print("\n" + "="*60)
    print("BUILDING TRAIN FEATURES")
    print("="*60)

    features_train = build_feature_table(
        data_dir="data/raw",
        feat_start="2023-06-01",
        feat_end="2023-10-31",
        save_to="data/processed/features_train.csv",
    )

    _smoke_test(features_train, Path("data/raw"))

    print("\n" + "="*60)
    print("BUILDING TEST FEATURES")
    print("="*60)

    features_test = build_feature_table(
        data_dir="data/raw",
        feat_start="2023-07-01",
        feat_end="2023-11-30",
        save_to="data/processed/features_test.csv",
    )

    _smoke_test(features_test, Path("data/raw"))

    print("\nDone.")
    print(f"Train shape: {features_train.shape}")
    print(f"Test shape : {features_test.shape}")