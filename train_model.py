"""
train_model.py
=============================================================================
IPL Final Score Prediction — Model Training & Serialization Script
Matches exact methodology from IPL_Final_Score_Prediction_Using_Machine_Learning.ipynb
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_and_preprocess_data(csv_path="IPL_ball_by_ball_updated.csv"):
    print(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Raw dataset shape: {df.shape}")

    data = df.copy()
    data['start_date'] = pd.to_datetime(data['start_date'], errors='coerce')

    numeric_columns = [
        'runs_off_bat',
        'extras',
        'wides',
        'noballs',
        'byes',
        'legbyes',
        'penalty'
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    # Remove exact duplicate rows
    data = data.drop_duplicates()

    # Total runs per ball & over number
    data['total_runs'] = data['runs_off_bat'] + data['extras']
    data['over_number'] = data['ball'].astype(str).str.split('.').str[0].astype(int)
    data['is_wicket'] = data['player_dismissed'].notna().astype(int)

    # Final score per match and innings
    final_scores = (
        data.groupby(['match_id', 'innings'])['total_runs']
        .sum()
        .reset_index()
        .rename(columns={'total_runs': 'final_score'})
    )
    data = data.merge(final_scores, on=['match_id', 'innings'], how='left')

    # Regular first and second innings only
    data_regular = data[data['innings'].isin([1, 2])].copy()
    data_regular = data_regular.sort_values(
        ['match_id', 'innings', 'over_number', 'ball']
    )

    # Cumulative calculations
    grouped = data_regular.groupby(['match_id', 'innings'], sort=False)
    data_regular['current_score'] = grouped['total_runs'].cumsum()
    data_regular['wickets_lost'] = grouped['is_wicket'].cumsum().clip(upper=10)

    # Over data aggregation
    over_data = (
        data_regular.groupby(['match_id', 'innings', 'over_number'])
        .agg(
            over_runs=('total_runs', 'sum'),
            over_wickets=('is_wicket', 'sum')
        )
        .reset_index()
    )

    over_data['last_5_over_runs'] = (
        over_data.groupby(['match_id', 'innings'])['over_runs']
        .transform(lambda x: x.rolling(window=5, min_periods=1).sum())
    )
    over_data['last_5_over_wickets'] = (
        over_data.groupby(['match_id', 'innings'])['over_wickets']
        .transform(lambda x: x.rolling(window=5, min_periods=1).sum())
    )

    # Select prediction points: 5th, 10th, and 15th overs (over_number 4, 9, 14)
    prediction_points = data_regular[
        data_regular['over_number'].isin([4, 9, 14])
    ].copy()

    # Keep final delivery of each selected over
    prediction_points = (
        prediction_points
        .sort_values(['match_id', 'innings', 'over_number', 'ball'])
        .groupby(['match_id', 'innings', 'over_number'], as_index=False)
        .tail(1)
    )

    prediction_points = prediction_points.merge(
        over_data[['match_id', 'innings', 'over_number', 'last_5_over_runs', 'last_5_over_wickets']],
        on=['match_id', 'innings', 'over_number'],
        how='left'
    )

    prediction_points['overs_completed'] = prediction_points['over_number'] + 1
    prediction_points['current_run_rate'] = (
        prediction_points['current_score'] / prediction_points['overs_completed']
    )

    # Select final ML columns
    ml_data = prediction_points[
        [
            'match_id',
            'season',
            'venue',
            'batting_team',
            'bowling_team',
            'overs_completed',
            'current_score',
            'wickets_lost',
            'current_run_rate',
            'last_5_over_runs',
            'last_5_over_wickets',
            'final_score'
        ]
    ].copy().dropna(subset=['final_score'])

    print(f"Prepared ML dataset shape: {ml_data.shape}")
    return ml_data, df


def train_and_evaluate(ml_data):
    features = [
        'batting_team',
        'bowling_team',
        'venue',
        'overs_completed',
        'current_score',
        'wickets_lost',
        'current_run_rate',
        'last_5_over_runs',
        'last_5_over_wickets'
    ]
    target = 'final_score'

    # Time-based train-test split
    train_data = ml_data[ml_data['season'] <= 2021].copy()
    test_data = ml_data[ml_data['season'] >= 2022].copy()

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    print(f"Training records: {len(X_train)} | Testing records: {len(X_test)}")

    categorical_features = ['batting_team', 'bowling_team', 'venue']
    numerical_features = [
        'overs_completed',
        'current_score',
        'wickets_lost',
        'current_run_rate',
        'last_5_over_runs',
        'last_5_over_wickets'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categorical',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                categorical_features
            ),
            (
                'numerical',
                'passthrough',
                numerical_features
            )
        ]
    )

    # 1. Linear Regression
    linear_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)

    # 2. Decision Tree Regressor
    dt_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeRegressor(max_depth=10, random_state=42))
    ])
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)

    # 3. Random Forest Regressor
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    def evaluate(name, y_true, preds):
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, preds)
        return {
            'Model': name,
            'MAE': round(float(mae), 2),
            'MSE': round(float(mse), 2),
            'RMSE': round(float(rmse), 2),
            'R² Score': round(float(r2), 4)
        }

    results = [
        evaluate('Linear Regression', y_test, linear_preds),
        evaluate('Decision Tree Regression', y_test, dt_preds),
        evaluate('Random Forest Regression', y_test, rf_preds)
    ]
    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Comparison Table:")
    print(results_df.to_string(index=False))

    # Verify sample prediction
    sample = X_test.iloc[[0]]
    actual_score = float(y_test.iloc[0])
    pred_score = float(rf_model.predict(sample)[0])
    print(f"\nSample Prediction Check:")
    print(f"Actual: {actual_score:.2f} | Predicted: {pred_score:.2f} | Error: {abs(actual_score - pred_score):.2f}")

    # Extract feature importances
    feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
    rf_importances = rf_model.named_steps['model'].feature_importances_
    fi_list = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, rf_importances)
    ]
    fi_list = sorted(fi_list, key=lambda x: x['importance'], reverse=True)

    # Save model and metadata
    model_file = "random_forest_model.pkl"
    joblib.dump(rf_model, model_file)
    print(f"\nSaved best model Pipeline to {model_file}")

    metadata = {
        "features": features,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target": target,
        "teams": sorted(ml_data['batting_team'].unique().tolist()),
        "venues": sorted(ml_data['venue'].unique().tolist()),
        "seasons": sorted(ml_data['season'].unique().tolist()),
        "results": results,
        "feature_importances": fi_list[:25],
        "sample_prediction": {
            "input": sample.to_dict(orient='records')[0],
            "actual": actual_score,
            "predicted": round(pred_score, 2),
            "error": round(abs(actual_score - pred_score), 2)
        }
    }

    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved model metadata to model_metadata.json")

    return rf_model, results_df, metadata


if __name__ == "__main__":
    ml_data, _ = load_and_preprocess_data()
    train_and_evaluate(ml_data)
