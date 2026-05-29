"""
generate_dataset_vectorized.py
FINAL APPROVED VERSION (VECTORIZED)
(Channel Affinity Prediction Project)

OUTPUT TABLES
-------------
customers.csv
campaigns_meta.csv
sessions.csv
email_engagement.csv
social_media_engagement.csv
content_engagement.csv
marketing_touchpoints.csv
orders.csv
order_items.csv
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)

# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = np.array([
    "Email", "Facebook", "Instagram", "Google Ads", 
    "Organic Search", "Direct", "YouTube"
])

SOCIAL_PLATFORMS = np.array([
    "Facebook", "Instagram", "Twitter", "LinkedIn", "YouTube"
])

CONTENT_TYPES = np.array([
    "blog", "ebook", "case_study", "webinar", "video"
])

PRODUCTS = np.array([
    "Laptop", "Phone", "Monitor", "Keyboard", "Mouse", "Headphones", "Camera"
])

DATE_START = pd.Timestamp("2023-06-01")
DATE_END = pd.Timestamp("2024-01-01")

# Pre-generate a pool of cities to avoid calling Faker in large loops
CITY_POOL = np.array([fake.city() for _ in range(250)])

# =============================================================================
# PERSONA-DRIVEN CHANNEL PROBABILITIES
# =============================================================================

PERSONA_CHANNEL_PROBS = {
    "Email": {"Email": 0.40, "Direct": 0.20, "Organic Search": 0.15, "Facebook": 0.10, "Instagram": 0.08, "Google Ads": 0.05, "YouTube": 0.02},
    "Facebook": {"Facebook": 0.38, "Instagram": 0.22, "Direct": 0.15, "Google Ads": 0.10, "Organic Search": 0.08, "Email": 0.05, "YouTube": 0.02},
    "Instagram": {"Instagram": 0.38, "Facebook": 0.22, "YouTube": 0.14, "Direct": 0.12, "Google Ads": 0.08, "Organic Search": 0.04, "Email": 0.02},
    "Google Ads": {"Google Ads": 0.36, "Organic Search": 0.22, "Direct": 0.16, "Email": 0.10, "Facebook": 0.08, "Instagram": 0.05, "YouTube": 0.03},
    "Organic Search": {"Organic Search": 0.38, "Direct": 0.22, "Google Ads": 0.16, "Email": 0.08, "YouTube": 0.08, "Facebook": 0.05, "Instagram": 0.03},
    "Direct": {"Direct": 0.40, "Organic Search": 0.22, "Email": 0.15, "Google Ads": 0.10, "Facebook": 0.07, "Instagram": 0.04, "YouTube": 0.02},
    "YouTube": {"YouTube": 0.38, "Instagram": 0.20, "Facebook": 0.16, "Direct": 0.12, "Organic Search": 0.08, "Google Ads": 0.04, "Email": 0.02},
}

# =============================================================================
# HELPERS
# =============================================================================

def random_datetimes(start: pd.Timestamp, end: pd.Timestamp, n: int) -> pd.Series:
    start_u = start.value // 10**9
    end_u = end.value // 10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, size=n), unit="s")

def vectorized_channel_choice(primary_channels: pd.Series) -> np.ndarray:
    result = np.empty(len(primary_channels), dtype=object)
    for ch in CHANNELS:
        mask = primary_channels == ch
        count = mask.sum()
        if count > 0:
            probs = [PERSONA_CHANNEL_PROBS[ch][c] for c in CHANNELS]
            result[mask] = np.random.choice(CHANNELS, size=count, p=probs)
    return result

# =============================================================================
# PERSONAS & CUSTOMERS
# =============================================================================

def generate_customers(n=40000):
    cids = [f"cust_{i:06d}" for i in range(n)]
    
    primary_channels = np.random.choice(CHANNELS, size=n)
    customer_types = np.random.choice(["stable", "noisy"], size=n, p=[0.90, 0.10])
    engagement_levels = np.random.choice(["high", "medium", "low"], size=n, p=[0.20, 0.60, 0.20])
    
    # Store personas in a DataFrame for fast merging later
    personas = pd.DataFrame({
        "customer_id": cids,
        "primary_channel": primary_channels,
        "customer_type": customer_types,
        "engagement_level": engagement_levels
    })
    
    acquisition_channels = vectorized_channel_choice(personas["primary_channel"])
    acquisition_dates = random_datetimes(DATE_START, DATE_END - timedelta(days=90), n)
    
    segments = np.random.choice(
        ["High Value", "Medium Value", "Low Value", "New", "At Risk"],
        size=n, p=[0.10, 0.30, 0.35, 0.15, 0.10]
    )
    
    total_orders = np.clip(np.random.poisson(5, size=n), 0, None)
    total_revenue = np.clip(np.random.normal(total_orders * 120, 100, size=n), 0, None).round(2)
    avg_order_value = np.where(total_orders > 0, (total_revenue / total_orders).round(2), 0)
    
    # Last purchase date logic (25% null)
    last_purchases = random_datetimes(DATE_START, DATE_END, n)
    last_purchases = np.where(last_purchases < acquisition_dates, acquisition_dates + timedelta(days=1), last_purchases)
    last_purchases = pd.Series(last_purchases).where(np.random.rand(n) > 0.25, pd.NaT)
    
    customers = pd.DataFrame({
        "customer_id": cids,
        "age": np.random.randint(18, 76, size=n),
        "gender": np.random.choice(["Male", "Female", "Other"], size=n),
        "location": np.random.choice(CITY_POOL, size=n),
        "customer_segment": segments,
        "acquisition_channel": acquisition_channels,
        "acquisition_date": acquisition_dates,
        "last_purchase_date": last_purchases,
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "customer_status": np.random.choice(["active", "inactive", "churned"], size=n, p=[0.72, 0.18, 0.10]),
        "lifetime_value": (total_revenue * np.random.uniform(0.8, 1.3, size=n)).round(2)
    })
    
    return customers, personas

# =============================================================================
# CAMPAIGNS
# =============================================================================

def generate_campaigns(n=500):
    channels = np.random.choice(CHANNELS, size=n)
    start_dates = random_datetimes(DATE_START, DATE_END - timedelta(days=30), n)
    end_dates = start_dates + pd.to_timedelta(np.random.randint(7, 46, size=n), unit='d')
    
    campaigns = pd.DataFrame({
        "campaign_id": [f"camp_{i:04d}" for i in range(n)],
        "campaign_name": [f"{ch}_camp_{i}" for i, ch in enumerate(channels)],
        "campaign_type": np.random.choice(["acquisition", "retention", "reactivation"], size=n),
        "channel": channels,
        "target_audience": np.random.choice(["All Users", "High Value", "New Users", "At Risk"], size=n),
        "budget": np.random.randint(1000, 50001, size=n),
        "start_date": start_dates,
        "end_date": end_dates,
        "performance_score": np.clip(np.random.normal(0.65, 0.15, size=n), 0, 1).round(2)
    })
    return campaigns

# =============================================================================
# SESSIONS
# =============================================================================

def generate_sessions(personas, n=200000):
    sample_personas = personas.sample(n=n, replace=True).reset_index(drop=True)
    
    channels = vectorized_channel_choice(sample_personas["primary_channel"])
    is_primary = channels == sample_personas["primary_channel"]
    is_noisy = sample_personas["customer_type"] == "noisy"
    
    # Base engagement
    engagement = np.where(is_primary, np.random.normal(68, 15, size=n), np.random.normal(52, 15, size=n))
    engagement = np.where(is_noisy, np.random.normal(50, 25, size=n), engagement)
    engagement = np.clip(engagement, 5, 100).round(2)
    
    duration = np.where(is_primary, np.random.normal(700, 250, size=n), np.random.normal(450, 250, size=n))
    duration = np.clip(duration, 20, None).astype(int)
    
    bounce = np.where(is_primary, np.random.normal(0.30, 0.15, size=n), np.random.normal(0.45, 0.15, size=n))
    bounce = np.clip(bounce, 0, 1).round(2)
    
    df = pd.DataFrame({
        "session_id": [f"sess_{i:08d}" for i in range(n)],
        "customer_id": sample_personas["customer_id"],
        "session_start_timestamp": random_datetimes(DATE_START, DATE_END, n),
        "channel": channels,
        "device_type": np.random.choice(["Mobile", "Desktop", "Tablet"], size=n),
        "page_views": np.clip(np.random.poisson(5, size=n), 1, None),
        "session_duration_seconds": duration,
        "engagement_score": engagement,
        "bounce_rate": bounce
    })
    
    # Intentional missingness
    df.loc[np.random.rand(n) < 0.05, "engagement_score"] = np.nan
    return df

# =============================================================================
# EMAIL ENGAGEMENT
# =============================================================================

def generate_email_events(personas, campaigns, n=80000):
    email_camps = campaigns.loc[campaigns["channel"] == "Email", "campaign_id"].values
    sample_personas = personas.sample(n=n, replace=True).reset_index(drop=True)
    
    is_email_persona = sample_personas["primary_channel"] == "Email"
    
    # Vectorized event type selection using probabilities
    event_types = np.where(
        is_email_persona,
        np.random.choice(["sent", "opened", "clicked"], size=n, p=[0.50, 0.35, 0.15]),
        np.random.choice(["sent", "opened", "clicked"], size=n, p=[0.72, 0.22, 0.06])
    )
    
    open_counts = np.where(np.isin(event_types, ["opened", "clicked"]), np.random.randint(1, 5, size=n), 0)
    click_counts = np.where(event_types == "clicked", np.random.randint(1, 3, size=n), 0)
    
    tt_open = np.where(is_email_persona, np.random.randint(2, 121, size=n), np.random.randint(30, 1441, size=n))
    
    df = pd.DataFrame({
        "email_event_id": [f"email_{i:07d}" for i in range(n)],
        "campaign_id": np.random.choice(email_camps, size=n),
        "customer_id": sample_personas["customer_id"],
        "event_timestamp": random_datetimes(DATE_START, DATE_END, n),
        "event_type": event_types,
        "open_count": open_counts,
        "click_count": click_counts,
        "time_to_open_minutes": tt_open.astype(float)
    })
    
    df.loc[np.random.rand(n) < 0.08, "time_to_open_minutes"] = np.nan
    return df

# =============================================================================
# SOCIAL ENGAGEMENT
# =============================================================================

def generate_social_events(personas, n=60000):
    sample_personas = personas.sample(n=n, replace=True).reset_index(drop=True)
    platforms = np.random.choice(SOCIAL_PLATFORMS, size=n)
    
    is_match = platforms == sample_personas["primary_channel"]
    sentiment = np.where(is_match, np.random.normal(0.70, 0.18, size=n), np.random.normal(0.55, 0.18, size=n))
    sentiment = np.clip(sentiment, 0, 1).round(2)
    
    df = pd.DataFrame({
        "social_engagement_id": [f"social_{i:07d}" for i in range(n)],
        "customer_id": sample_personas["customer_id"],
        "platform": platforms,
        "engagement_timestamp": random_datetimes(DATE_START, DATE_END, n),
        "engagement_type": np.random.choice(["like", "comment", "share", "follow"], size=n),
        "sentiment_score": sentiment
    })
    
    df.loc[np.random.rand(n) < 0.05, "sentiment_score"] = np.nan
    return df

# =============================================================================
# CONTENT ENGAGEMENT
# =============================================================================

def generate_content_events(personas, n=70000):
    sample_personas = personas.sample(n=n, replace=True).reset_index(drop=True)
    
    is_video_persona = sample_personas["primary_channel"].isin(["YouTube", "Instagram"])
    completion = np.where(is_video_persona, np.random.normal(0.75, 0.20, size=n), np.random.normal(0.55, 0.20, size=n))
    completion = np.clip(completion, 0, 1).round(2)
    
    time_spent = np.clip(np.random.normal(240, 120, size=n), 10, None).astype(int)
    
    df = pd.DataFrame({
        "engagement_id": [f"content_{i:07d}" for i in range(n)],
        "customer_id": sample_personas["customer_id"],
        "engagement_timestamp": random_datetimes(DATE_START, DATE_END, n),
        "content_type": np.random.choice(CONTENT_TYPES, size=n),
        "engagement_type": np.random.choice(["view", "download", "watch"], size=n),
        "time_spent_seconds": time_spent,
        "completion_percentage": completion
    })
    
    df.loc[np.random.rand(n) < 0.04, "completion_percentage"] = np.nan
    return df

# =============================================================================
# TOUCHPOINTS
# =============================================================================

def generate_touchpoints(personas, campaigns, n=150000):
    sample_personas = personas.sample(n=n, replace=True).reset_index(drop=True)
    channels = vectorized_channel_choice(sample_personas["primary_channel"])
    
    # Fast assignment of random campaign IDs based on matching channels
    camp_dict = campaigns.groupby('channel')['campaign_id'].apply(list).to_dict()
    campaign_ids = [np.random.choice(camp_dict.get(ch, ["Unknown"])) for ch in channels]
    
    tp_types = np.random.choice(["impression", "click", "form_submit"], size=n, p=[0.65, 0.28, 0.07])
    
    revenue = np.where(tp_types == "form_submit", np.random.normal(120, 30, size=n), np.random.normal(20, 30, size=n))
    revenue = np.clip(revenue, 0, None).round(2)
    
    df = pd.DataFrame({
        "touchpoint_id": [f"tp_{i:08d}" for i in range(n)],
        "customer_id": sample_personas["customer_id"],
        "timestamp": random_datetimes(DATE_START, DATE_END, n),
        "channel": channels,
        "touchpoint_type": tp_types,
        "campaign_id": campaign_ids,
        "revenue_attributed": revenue
    })
    
    df.loc[np.random.rand(n) < 0.03, "revenue_attributed"] = np.nan
    return df

# =============================================================================
# ORDERS & ORDER ITEMS
# =============================================================================

def generate_orders_and_items(customers, n=35000):
    cids = np.random.choice(customers["customer_id"].values, size=n)
    
    orders = pd.DataFrame({
        "order_id": [f"ord_{i:07d}" for i in range(n)],
        "customer_id": cids,
        "order_timestamp": random_datetimes(DATE_START, DATE_END, n),
        "payment_method": np.random.choice(["Credit Card", "Debit Card", "PayPal", "UPI"], size=n),
        "order_value": np.clip(np.random.normal(180, 120, size=n), 15, None).round(2),
        "discount_amount": np.clip(np.random.normal(15, 8, size=n), 0, None).round(2)
    })
    
    # Generate Order Items (1 to 3 items per order)
    n_items_per_order = np.random.randint(1, 4, size=n)
    total_items = n_items_per_order.sum()
    
    order_ids_repeated = np.repeat(orders["order_id"].values, n_items_per_order)
    qty = np.random.randint(1, 3, size=total_items)
    unit_prices = np.clip(np.random.normal(120, 80, size=total_items), 20, None).round(2)
    
    order_items = pd.DataFrame({
        "order_item_id": [f"item_{i:08d}" for i in range(total_items)],
        "order_id": order_ids_repeated,
        "product_name": np.random.choice(PRODUCTS, size=total_items),
        "quantity": qty,
        "unit_price": unit_prices,
        "total_price": (qty * unit_prices).round(2)
    })
    
    return orders, order_items

# =============================================================================
# SAVE PIPELINE
# =============================================================================

def save_all(output_dir="data/raw"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating vectorized customers & personas...")
    customers, personas = generate_customers()

    print("Generating vectorized campaigns...")
    campaigns = generate_campaigns()

    print("Generating vectorized sessions...")
    sessions = generate_sessions(personas)

    print("Generating vectorized email engagement...")
    email_events = generate_email_events(personas, campaigns)

    print("Generating vectorized social engagement...")
    social_events = generate_social_events(personas)

    print("Generating vectorized content engagement...")
    content_events = generate_content_events(personas)

    print("Generating vectorized touchpoints...")
    touchpoints = generate_touchpoints(personas, campaigns)

    print("Generating vectorized orders & items...")
    orders, order_items = generate_orders_and_items(customers)

    # SAVE CSVs
    customers.to_csv(output_dir / "customers.csv", index=False)
    campaigns.to_csv(output_dir / "campaigns_meta.csv", index=False)
    sessions.to_csv(output_dir / "sessions.csv", index=False)
    email_events.to_csv(output_dir / "email_engagement.csv", index=False)
    social_events.to_csv(output_dir / "social_media_engagement.csv", index=False)
    content_events.to_csv(output_dir / "content_engagement.csv", index=False)
    touchpoints.to_csv(output_dir / "marketing_touchpoints.csv", index=False)
    orders.to_csv(output_dir / "orders.csv", index=False)
    order_items.to_csv(output_dir / "order_items.csv", index=False)

    print("\nDONE.")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    save_all()