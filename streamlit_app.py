import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Channel Affinity Portal",
    page_icon="📡",
    layout="wide"
)

# Paths setup
DATA_DIR = Path("./data")
RAW_PATH = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed"


@st.cache_data
def load_dashboard_datasets():
    cust = pd.read_csv(RAW_PATH / "customers.csv")
    preds = pd.read_csv(PROCESSED_PATH / "ml_predictions.csv")
    ml_metrics = pd.read_csv(PROCESSED_PATH / "ml_metrics.csv")
    rule_metrics = pd.read_csv(PROCESSED_PATH / "rule_based_metrics_test.csv")
    return cust, preds, ml_metrics, rule_metrics

customers, predictions, ml_metrics, rule_metrics = load_dashboard_datasets()

# Match customer records with predicted outputs
merged_df = pd.merge(customers, predictions, on="customer_id", how="left")


# ---------------------------------------------------------------------------
# Navigation Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Navigation")
    active_page = st.radio(
        "Select Page",
        ["Overview", "Customer Explorer", "Budget Allocation"]
    )


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if active_page == "Overview":
    st.title("📡 Channel Affinity Prediction System")

    st.markdown("""
    Predicting the most effective customer communication channel using
    behavioral engagement, marketing touchpoints, session activity,
    purchase history and recency signals.
    """)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Analyzed", f"{len(customers):,}")
    k2.metric("Channels Predicted", "7")
    k3.metric("Holdout Accuracy", "33.5%")
    k4.metric("Lift Over Random", "2.3x")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Recommended Channel Distribution")
        dist = predictions["rank_1"].value_counts().reset_index()
        dist.columns = ["Channel", "Customers"]

        fig = px.bar(dist, x="Channel", y="Customers", color="Channel")
        fig.update_layout(
            height=450,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Model Comparison")
        compare = pd.DataFrame({
            "Model": ["Rule-Based", "LightGBM"],
            "Accuracy": [29.3, 33.5]
        })

        fig2 = px.bar(compare, x="Model", y="Accuracy", text="Accuracy")
        fig2.update_traces(texttemplate="%{text:.1f}%")
        fig2.update_layout(
            height=450,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig2, width="stretch")

    st.divider()
    st.subheader("Model Drivers")
    st.markdown("""
    **Top drivers of channel affinity**

    - Session engagement behaviour
    - Channel recency patterns
    - Email response behaviour
    - Historical touchpoint interactions
    - Customer lifecycle stage
    - Revenue and purchasing behaviour

    These signals collectively determine the probability of future engagement
    across Email, Social, Paid Media, Direct and Organic channels.
    """)


# ---------------------------------------------------------------------------
# Page: Customer Explorer
# ---------------------------------------------------------------------------

elif active_page == "Customer Explorer":
    st.title("👤 Individual Customer Profiles")
    
    selected_id = st.selectbox("Search Customer Identifier", merged_df["customer_id"])
    cust_row = merged_df[merged_df["customer_id"] == selected_id].iloc[0]
    
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("Demographic & Lifecycle Profile")
        st.write(f"**Age / Gender:** {cust_row['age']} | {cust_row['gender']}")
        st.write(f"**Geographic Location:** {cust_row['location']}")
        st.write(f"**Segment Assignment:** {cust_row['customer_segment']}")
        st.write(f"**Account Status:** {cust_row['customer_status']}")
        st.write(f"**Estimated Lifetime Value:** ${cust_row['lifetime_value']:,.2f}")
        
    with right_col:
        st.subheader("Algorithmic Channel Rankings")
        st.success(f"**Primary Preference (Rank 1):** {cust_row['rank_1']}")
        st.info(f"**Secondary Preference (Rank 2):** {cust_row['rank_2']}")
        st.info(f"**Tertiary Preference (Rank 3):** {cust_row['rank_3']}")
        st.metric("Model Confidence", f"{cust_row['confidence']:.1%}")
        
        st.success(
            f"""
            Recommended next action:

            Allocate primary communication through
            **{cust_row['rank_1']}**

            Confidence: **{cust_row['confidence']:.1%}**
            """
        )
        
    st.divider()
    
    # Isolate probability features dynamically
    prob_cols = {col.replace("prob_", "").replace("_", " "): cust_row[col] 
                 for col in merged_df.columns if col.startswith("prob_")}
    
    prob_df = pd.DataFrame({
        "Channel Option": prob_cols.keys(),
        "Probability Score": prob_cols.values()
    }).sort_values(by="Probability Score", ascending=False)
    
    display_probs = prob_df.copy()
    display_probs["Probability Score"] = (display_probs["Probability Score"] * 100).round(2)

    st.subheader("Probability Scores")
    st.dataframe(
        display_probs.rename(columns={"Probability Score": "Probability (%)"}),
        hide_index=True,
        width="stretch"
    )
    
    fig = px.bar(prob_df, x="Channel Option", y="Probability Score", labels={"Probability Score": "Likelihood"})
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Page: Budget Allocation
# ---------------------------------------------------------------------------

elif active_page == "Budget Allocation":
    st.title("💰 Strategic Budget Simulator")
    st.markdown("Distribute spend based on the actual affinity footprint of the active customer base.")
    
    base_share = predictions["rank_1"].value_counts(normalize=True).mul(100).reset_index()
    base_share.columns = ["Channel Target", "Audience Concentration (%)"]
    
    # Horizontal bar plot layout replacing old pie chart
    fig = px.bar(
        base_share.sort_values("Audience Concentration (%)"),
        x="Audience Concentration (%)",
        y="Channel Target",
        orientation="h",
        text="Audience Concentration (%)"
    )
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    
    working_budget = st.number_input("Enter Total Projected Monthly Budget ($)", min_value=0, value=100000, step=5000)
    base_share["Dynamic Allocation ($)"] = (base_share["Audience Concentration (%)"] / 100) * working_budget
    
    # Formatting layout columns for clean output
    styled_allocations = base_share.copy()
    styled_allocations["Audience Concentration (%)"] = styled_allocations["Audience Concentration (%)"].map("{:.2f}%".format)
    styled_allocations["Dynamic Allocation ($)"] = styled_allocations["Dynamic Allocation ($)"].map("${:,.2f}".format)
    
    st.subheader("Calculated Channel Resource Allocations")
    st.dataframe(styled_allocations, hide_index=True, width="stretch")
    st.caption("Projections optimized automatically using LightGBM Rank-1 propensity models.")