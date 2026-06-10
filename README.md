# ✈️ Flight Price Prediction

A machine learning project that analyzes **~300,000 Indian domestic flights** to identify key pricing drivers and predict ticket prices using regression analysis.

---

## 📌 Overview

Flight prices are dynamic and influenced by dozens of variables. This project builds a predictive pipeline to estimate ticket costs and — more importantly — **quantify which factors drive airline pricing the most**, generating business-facing insights relevant to revenue management and retail intelligence.

**Target Variable:** `price` — the actual INR cost of a flight ticket (continuous → Regression problem)

---

## 📊 Dataset

- **Source:** [Kaggle — Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- **Size:** ~300,000 domestic flight records
- **Features:**

| Category | Features |
|---|---|
| Carrier & Comfort | `airline`, `stops`, `class` |
| Route | `source_city`, `destination_city`, `duration` |
| Timing & Urgency | `departure_time`, `arrival_time`, `days_left` |

---

## 🔍 Exploratory Data Analysis

Key insights uncovered during EDA:

- **Seat class** is the single most dominant pricing factor — Business Class tickets are **7x more expensive** than Economy (median ₹54,208 vs ₹7,418)
- **Booking lead time** follows a clear pattern — prices are stable 15–45 days before departure, then spike violently in the last 1–2 days ("panic-booking surge")
- **Airlines cluster into two tiers** — Vistara and Air India form a premium tier driven by Business Class availability; SpiceJet, IndiGo, AirAsia, and GO FIRST compete in a tight budget bracket
- **Route pricing** is driven by distance and hub demand — Chennai routes command highest median prices; Delhi trunk routes are most competitive
- **Departure/arrival time** shows a "convenience premium" — morning and evening flights cost more; late-night and early-morning flights are heavily discounted

---

## ⚙️ Methodology

### 1. Data Cleaning
- Zero missing values — no imputation required
- Outlier removal at **0.5% threshold** (rather than standard 5%) to preserve natural pricing variance while removing genuine anomalies

### 2. Feature Engineering

Intentional encoding strategy based on variable type:

| Variable | Encoding | Reason |
|---|---|---|
| `airline`, `source_city`, `destination_city` | One-Hot Encoding | Nominal — no inherent ranking |
| `stops` | Ordinal Encoding (0, 1, 2) | Ordinal — more stops = higher cost |
| `class` | Binary Encoding (0/1) | Only two states — avoids dummy variable trap |

### 3. Target Transformation
- Applied **log transformation** (`np.log1p`) on `price` to reduce right skewness
- All metrics evaluated on reverse-transformed (actual INR) values for interpretability

### 4. Model Pipeline

Built using `sklearn.pipeline.Pipeline` with:
- `ColumnTransformer` for preprocessing
- `StandardScaler` for numeric features
- `LinearRegression` as the estimator
- Separate pipeline with **PCA** for dimensionality reduction comparison

---

## 📈 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (No PCA) | ₹4,536 | ₹7,983 | 0.8711 |
| Linear Regression + PCA |  ₹4,618 | ₹7,936 | 0.8726 |

> Run the notebook to see exact metric values — they depend on the train/test split seed.

### Top Feature Coefficients
Feature importance extracted directly from model coefficients — `class_encoded` (Business vs Economy) had the highest absolute impact by a significant margin.

---

## 💡 Business Research Questions Answered

**Q: Does price vary with airlines?**
Yes — a strict premium/budget divide exists. Vistara and Air India dominate the premium tier; budget carriers cluster tightly together.

**Q: How does last-minute booking affect price?**
Prices are stable from 45 days out but spike sharply in the final 1–2 days. The sweet spot for booking is 15–45 days in advance.

**Q: Does departure/arrival time affect price?**
Yes — morning and evening slots carry a "convenience premium." Late-night and early-morning flights are cheapest.

**Q: How does route affect price?**
Chennai-based routes are most expensive (distance + hub demand). Delhi trunk routes are cheapest due to intense airline competition.

**Q: Economy vs Business — how large is the gap?**
Business Class commands a **7.3x price multiplier** over Economy — the single largest pricing signal in the entire dataset.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment:** Kaggle Notebooks

---



---
