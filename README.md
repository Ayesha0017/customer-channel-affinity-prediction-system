# Customer Channel Affinity Prediction System

## Project Overview

Marketing teams often communicate with customers through multiple channels such as Email, Facebook, Instagram, Google Ads, Organic Search, Direct traffic, and YouTube.

Sending every campaign through every channel is expensive and inefficient. The objective of this project is to predict the communication channel a customer is most likely to engage with in the upcoming period, allowing marketing teams to allocate budget more effectively.

This project builds both:

* A rules-based channel affinity framework
* A machine learning channel affinity model using LightGBM

The final system recommends the best communication channel for each customer and provides channel-level budget allocation insights.

---

## Business Problem

Given a customer's historical behavior:

* Website sessions
* Marketing touchpoints
* Email engagement
* Social engagement
* Content consumption
* Purchase history

Predict:

> Which channel is this customer most likely to engage with next?

The output can be used for:

* Marketing budget allocation
* Campaign targeting
* Customer journey optimization
* Next-best-channel recommendation systems

---

## Dataset

The project uses synthetic customer interaction data across multiple business sources.

### Raw Tables

* customers.csv
* orders.csv
* order_items.csv
* marketing_touchpoints.csv
* sessions.csv
* email_engagement.csv
* social_media_engagement.csv
* content_engagement.csv
* campaigns_meta.csv

---

## Feature Engineering

Behavioral features were created from historical customer activity.

### Feature Categories

#### Customer Profile Features

* Age
* Gender
* Acquisition Channel
* Customer Segment
* Lifetime Value
* Revenue Metrics

#### Touchpoint Features

For each channel:

* Touchpoint Count
* Impressions
* Clicks
* Revenue Attribution
* Click Through Rate
* Momentum Metrics

#### Session Features

For each channel:

* Session Count
* Average Engagement
* Average Duration
* Weighted Engagement

#### Email Features

* Open Rate
* Click Rate
* Median Time To Open

#### Social Features

* Engagement Counts
* Platform Activity

#### Content Features

* Video Watch Time
* Content Consumption Metrics

#### Recency Features

* Channel Recency
* Session Recency
* Interaction Recency

Total Feature Count:

**140+ engineered features**

---

## Target Construction

A channel affinity score was calculated for each customer using a future label window.

Customers were assigned:

* Preferred Channel (Rank 1)
* Secondary Channel (Rank 2)
* Tertiary Channel (Rank 3)

Low-confidence labels were excluded from model training.

---

## Temporal Validation Strategy

To avoid data leakage, a strict temporal split was implemented.

| Window         | Period       | Purpose             |
| -------------- | ------------ | ------------------- |
| Feature Window | Jun–Oct 2023 | Feature Engineering |
| Label Window   | Nov 2023     | Training Labels     |
| Holdout Window | Dec 2023     | Final Evaluation    |

No information from the holdout period was used during training.

---

## Rule-Based Framework

A scoring engine was created using behavioral engagement signals.

Evaluation Results:

* Top-1 Accuracy: 29.3%
* Top-3 Accuracy: 63.3%
* Lift Over Random: 2.1x

---

## Machine Learning Framework

Model:

* LightGBM Multiclass Classifier

Training Strategy:

* High-confidence labels only
* Stratified 5-Fold Cross Validation
* Temporal Holdout Evaluation
* Leakage-safe pipeline

### Cross Validation

* Accuracy: 34.13% ± 0.51%

### Holdout Results

* Top-1 Accuracy: 33.5%
* Top-3 Accuracy: 70.4%
* Macro F1: 0.339
* Lift Over Random: 2.3x

Baseline Random Accuracy:

* 14.3%

---

## Key Findings

### Most Predictive Signals

* Email response behavior
* Session engagement
* Session duration
* Touchpoint recency
* Customer acquisition source
* Revenue history
* Customer lifetime value

### Business Insights

The model identified meaningful differences in customer communication preferences.

Example channel allocation recommendations:

| Channel        | Recommended Share |
| -------------- | ----------------- |
| Organic Search | 17.7%             |
| Email          | 15.2%             |
| Facebook       | 14.1%             |
| YouTube        | 14.0%             |
| Instagram      | 13.9%             |
| Google Ads     | 12.8%             |
| Direct         | 12.2%             |

---

## Streamlit Dashboard

An interactive dashboard was developed for:

* Customer-level channel recommendations
* Probability-based channel rankings
* Model performance monitoring
* Budget allocation simulation
* Marketing planning

---

## Project Structure

```text
customer-channel-affinity-ml/

data/
├── raw/
├── processed/

models/
├── lgbm_channel_affinity.joblib

notebooks/
├── 01_eda.ipynb
├── 02_feature_exploration.ipynb
├── 03_model_training.ipynb
├── 04_evaluation.ipynb

reports/

src/
├── dataset.py

├── features/
│   ├── build_features.py
│   └── label_builder.py

├── models/
│   ├── rule_based.py
│   └── ml_model.py

streamlit_app.py
requirements.txt
README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* Plotly
* Streamlit
* Matplotlib
* Seaborn

---

## Future Improvements

* XGBoost comparison
* Deep learning channel ranking models
* Multi-touch attribution integration
* Real-time prediction pipeline
* Marketing campaign optimization engine

---

## Author

Ayesha Firdaus Honnur

Data Analytics | Machine Learning | Marketing Analytics
