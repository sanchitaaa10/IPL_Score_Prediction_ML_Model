# IPL Final Score Prediction Using Machine Learning

An end-to-end Machine Learning and Sports Analytics system designed to predict the final innings score of an Indian Premier League (IPL) cricket match using situational match momentum (overs, wickets, current score, recent 5-over scoring rate, venue, and team matchups).

The project features a full exploratory data analysis (EDA) pipeline, comparative benchmarking of multiple regression algorithms, model serialization, and an interactive **Streamlit Web Dashboard**.

---

## Overview

In Twenty20 (T20) cricket, match momentum fluctuates with every over. Estimating an expected first-innings score or target score is critical for team strategy, resource planning, and sports analytics. This project leverages historical IPL ball-by-ball data (2008–2023) to train regression models capable of predicting the final total at crucial match milestones (5th, 10th, and 15th overs).

---

## Problem Statement

To formulate a supervised regression model that accurately estimates the total runs scored in an IPL innings based on the state of the match at a given prediction point, incorporating cumulative score, wicket loss, recent run rate, team strength, and venue characteristics.

---

## Objectives

1. **Data Preprocessing & Cleaning:** Clean raw IPL ball-by-ball records, eliminate duplicates, and handle delivery-level extras.
2. **Feature Engineering:** Derive cumulative scores, wickets fallen, completed overs, current run rate (CRR), and rolling 5-over surge metrics (`last_5_over_runs` and `last_5_over_wickets`).
3. **Milestone Extraction:** Extract match situations at key prediction points (5, 10, and 15 completed overs).
4. **Exploratory Data Analysis (EDA):** Analyze score distributions, team-wise scoring patterns, prediction stage effects, and feature correlations.
5. **Model Training & Comparison:** Implement and compare **Linear Regression**, **Decision Tree Regression**, and **Random Forest Regression** using a time-based train-test split.
6. **Deployment:** Build a web application using Streamlit for live match scenario prediction and interactive match analytics.

---

## Dataset

- **File:** `IPL_ball_by_ball_updated.csv`
- **Raw Records:** 243,815 rows × 22 columns
- **Time Span:** IPL 2008 to 2023
- **Granularity:** Delivery-by-delivery tracking of matches, seasons, venues, teams, batsmen, bowlers, runs, extras, and dismissals.

---

## Features

The model is trained on **9 exact features** (3 categorical and 6 numerical):

| # | Feature Name | Type | Description |
|---|--------------|------|-------------|
| 1 | `batting_team` | Categorical | Team currently batting |
| 2 | `bowling_team` | Categorical | Team currently bowling |
| 3 | `venue` | Categorical | Match stadium / location |
| 4 | `overs_completed` | Numerical | Milestone overs completed (5, 10, or 15) |
| 5 | `current_score` | Numerical | Total runs scored up to current over |
| 6 | `wickets_lost` | Numerical | Total wickets lost up to current over (0–10) |
| 7 | `current_run_rate` | Numerical | Derived: `current_score / overs_completed` |
| 8 | `last_5_over_runs` | Numerical | Runs scored during the last 5 overs |
| 9 | `last_5_over_wickets` | Numerical | Wickets fallen during the last 5 overs |

**Target Variable:** `final_score` (Total innings score)

---

## Machine Learning Algorithms

The project evaluates three supervised regression models using an `sklearn.pipeline.Pipeline` with `ColumnTransformer` (One-Hot Encoding for categorical features with `handle_unknown='ignore'` and passthrough for numerical features):

1. **Linear Regression (Baseline):** Fits an ordinary least squares linear equation.
2. **Decision Tree Regressor:** Non-linear decision trees (`max_depth=10`, `random_state=42`).
3. **Random Forest Regressor:** Ensemble of 200 bagging decision trees (`n_estimators=200`, `max_depth=15`, `random_state=42`, `n_jobs=-1`).

---

## Model Results

Models were evaluated on unseen future seasons using a time-based train-test split:
- **Training Set:** Seasons $\le$ 2021 (5,160 samples)
- **Testing Set:** Seasons $\ge$ 2022 (878 samples)

| Model | MAE (Runs) | MSE | RMSE (Runs) | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 19.38 | 619.45 | 24.89 | 0.3322 |
| **Decision Tree Regression** | 20.41 | 698.36 | 26.43 | 0.2472 |
| **Random Forest Regression (Best)** | **18.65** | **576.78** | **24.02** | **0.3782** |

---

## Best Model

**Random Forest Regression** achieved the best predictive performance:
- Lowest Mean Absolute Error (**18.65 runs**)
- Lowest Root Mean Squared Error (**24.02 runs**)
- Highest Coefficient of Determination (**R² = 0.3782**)

---

## Streamlit Dashboard

The web dashboard is organized into 5 dedicated pages:

1. 🏠 **Dashboard:** Tournament high-level KPIs (Matches, Deliveries, Venues, Model benchmarks) and score distribution overview.
2. 🎯 **Score Prediction:** Interactive match scenario inputs with live Run Rate calculation, input validation, projected score card ($\pm 24$ runs RMSE range), and projected death-over run rates.
3. 📊 **Match Analytics:** 6 EDA charts (Score distribution, Team averages, Current vs Final scatter, Milestone boxplots, Venue rankings, and Correlation heatmap).
4. 🤖 **Model Performance:** Comparative benchmark table, visual metric comparisons (MAE, RMSE, R²), and decoded Top 10 feature importances.
5. ℹ️ **About Project:** Architecture diagrams, methodology, limitations, and future scope.

---

## Project Structure

```
IPL_Score_Predictor/
│
├── app.py                                                # Streamlit Web Dashboard
├── train_model.py                                        # Standalone ML Training & Serialization
├── test_app.py                                           # Automated Test Suite (10 test scenarios)
├── random_forest_model.pkl                               # Serialized Trained Random Forest Pipeline
├── model_metadata.json                                   # Cached evaluation metrics & metadata
├── IPL_ball_by_ball_updated.csv                          # Historical IPL ball-by-ball dataset
├── IPL_Final_Score_Prediction_Using_Machine_Learning.ipynb # Source ML Jupyter Notebook
├── requirements.txt                                      # Project dependencies
└── README.md                                             # Project Documentation
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the repository:**
   ```bash
   cd IPL_Score_Predictor
   ```

2. **Create and activate a virtual environment:**
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: Train / Verify Model (Optional if `.pkl` is present)
```bash
python train_model.py
```

### Step 2: Run Automated Tests
```bash
python test_app.py
```

### Step 3: Launch Streamlit Dashboard
```bash
streamlit run app.py
```

Open your browser and navigate to the local URL (typically `http://localhost:8501`).

---

## Example Prediction

**Match Situation (Test Sample Index 0):**
- **Batting Team:** Chennai Super Kings
- **Bowling Team:** Kolkata Knight Riders
- **Venue:** Wankhede Stadium, Mumbai
- **Overs Completed:** 5.0
- **Current Score:** 29 runs
- **Wickets Lost:** 2
- **Current Run Rate:** 5.80 RPO
- **Last 5 Over Runs:** 29 runs
- **Last 5 Over Wickets:** 2

**Model Output:**
- **Actual Final Score:** 131 Runs
- **Predicted Final Score:** 149.08 Runs
- **Prediction Error:** 18.08 Runs (Well within $\pm 24.02$ RMSE)

---

## Limitations

1. **Historical Data Dependency:** Predictions reflect past tournament scoring patterns; recent rule changes (e.g., Impact Player rule) may elevate average scoring rates.
2. **Unobserved Conditions:** In-game factors like dew factor, humidity, wind, and pitch wear are not recorded in standard ball-by-ball sheets.
3. **Player Head-to-Head:** Individual batter vs bowler matchups and player injuries are not explicitly modeled.
4. **T20 Volatility:** Unpredictable events (such as 30-run death overs or sudden batting collapses) introduce inherent statistical variance.

---

## Future Scope

- Integrating live ball-by-ball webhooks for real-time win probability and dynamic score forecasting.
- Adding batsman and bowler player-level impact vectors.
- Incorporating match toss outcome, weather API telemetry, and pitch report features.
- Experimenting with Gradient Boosting frameworks (XGBoost, LightGBM, CatBoost) and Monte Carlo ball simulations.

---

## Authors

- [Add Team Members]
- College / Academic Project Presentation
