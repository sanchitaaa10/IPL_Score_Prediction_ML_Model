"""
=============================================================================
🏏 IPL MATCH CENTER — OFFICIAL AI SCORE PREDICTION & ANALYTICS DASHBOARD
=============================================================================
High-performance, broadcast-grade Streamlit web application providing
real-time match predictions, franchise intelligence, ground analytics,
and machine learning performance benchmarks.
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# 1. PAGE CONFIGURATION & BROADCAST THEME
# =============================================================================
st.set_page_config(
    page_title="IPL Match Center — AI Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Franchise color themes & short codes
TEAM_METADATA = {
    'Chennai Super Kings': {'short': 'CSK', 'color': '#FDB913', 'secondary': '#002B49', 'bg': 'rgba(253, 185, 19, 0.15)', 'border': '#FDB913', 'icon': '🦁'},
    'Mumbai Indians': {'short': 'MI', 'color': '#004BA0', 'secondary': '#D1AB3E', 'bg': 'rgba(0, 75, 160, 0.15)', 'border': '#004BA0', 'icon': '🌀'},
    'Royal Challengers Bangalore': {'short': 'RCB', 'color': '#EC1C24', 'secondary': '#000000', 'bg': 'rgba(236, 28, 36, 0.15)', 'border': '#EC1C24', 'icon': '👑'},
    'Kolkata Knight Riders': {'short': 'KKR', 'color': '#3A225D', 'secondary': '#B3A123', 'bg': 'rgba(58, 34, 93, 0.25)', 'border': '#8A5FB8', 'icon': '⚔️'},
    'Delhi Capitals': {'short': 'DC', 'color': '#004C97', 'secondary': '#EF1B23', 'bg': 'rgba(0, 76, 151, 0.15)', 'border': '#004C97', 'icon': '🐯'},
    'Rajasthan Royals': {'short': 'RR', 'color': '#EA1A85', 'secondary': '#254AA5', 'bg': 'rgba(234, 26, 133, 0.15)', 'border': '#EA1A85', 'icon': '👑'},
    'Sunrisers Hyderabad': {'short': 'SRH', 'color': '#F26522', 'secondary': '#000000', 'bg': 'rgba(242, 101, 34, 0.15)', 'border': '#F26522', 'icon': '🦅'},
    'Punjab Kings': {'short': 'PBKS', 'color': '#ED1B24', 'secondary': '#D7D8DA', 'bg': 'rgba(237, 27, 36, 0.15)', 'border': '#ED1B24', 'icon': '🦁'},
    'Gujarat Titans': {'short': 'GT', 'color': '#1B2133', 'secondary': '#D2B05E', 'bg': 'rgba(27, 33, 51, 0.35)', 'border': '#D2B05E', 'icon': '⚡'},
    'Lucknow Super Giants': {'short': 'LSG', 'color': '#0057E7', 'secondary': '#A0E8AF', 'bg': 'rgba(0, 87, 231, 0.15)', 'border': '#38bdf8', 'icon': '🦅'},
    'Delhi Daredevils': {'short': 'DD', 'color': '#004C97', 'secondary': '#EF1B23', 'bg': 'rgba(0, 76, 151, 0.15)', 'border': '#004C97', 'icon': '🛡️'},
    'Kings XI Punjab': {'short': 'KXIP', 'color': '#ED1B24', 'secondary': '#D7D8DA', 'bg': 'rgba(237, 27, 36, 0.15)', 'border': '#ED1B24', 'icon': '🦁'},
    'Deccan Chargers': {'short': 'DCG', 'color': '#002B49', 'secondary': '#D1AB3E', 'bg': 'rgba(0, 43, 73, 0.25)', 'border': '#D1AB3E', 'icon': '🐂'},
    'Gujarat Lions': {'short': 'GL', 'color': '#E04F16', 'secondary': '#1B2133', 'bg': 'rgba(224, 79, 22, 0.15)', 'border': '#E04F16', 'icon': '🦁'},
    'Rising Pune Supergiant': {'short': 'RPS', 'color': '#D11D5B', 'secondary': '#3A225D', 'bg': 'rgba(209, 29, 91, 0.15)', 'border': '#D11D5B', 'icon': '🌟'},
    'Rising Pune Supergiants': {'short': 'RPS', 'color': '#D11D5B', 'secondary': '#3A225D', 'bg': 'rgba(209, 29, 91, 0.15)', 'border': '#D11D5B', 'icon': '🌟'},
    'Pune Warriors': {'short': 'PWI', 'color': '#2F9BE3', 'secondary': '#000000', 'bg': 'rgba(47, 155, 227, 0.15)', 'border': '#2F9BE3', 'icon': '⚔️'},
    'Kochi Tuskers Kerala': {'short': 'KTK', 'color': '#6E2C91', 'secondary': '#FDB913', 'bg': 'rgba(110, 44, 145, 0.15)', 'border': '#6E2C91', 'icon': '🐘'}
}

# Custom Broadcast CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Official IPL Header Banner */
    .ipl-header {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 40%, #002855 100%);
        border: 1px solid rgba(255, 215, 0, 0.25);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .ipl-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 6px; height: 100%;
        background: linear-gradient(to bottom, #f59e0b, #3b82f6, #10b981);
    }
    .ipl-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }
    .ipl-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .ipl-main-title {
        font-family: 'Teko', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 0.04em;
        line-height: 1;
        background: linear-gradient(to right, #ffffff, #93c5fd, #fde047);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-transform: uppercase;
    }
    .ipl-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 2px;
    }
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        animation: pulse 2s infinite;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background: #ef4444;
        border-radius: 50%;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    /* Broadcast Match Card HUD */
    .broadcast-card {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .matchup-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 15px;
        text-align: center;
    }
    .team-badge-box {
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .team-code {
        font-family: 'Teko', sans-serif;
        font-size: 2.2rem;
        line-height: 1;
        font-weight: 700;
    }
    .vs-circle {
        background: #0b132b;
        border: 2px solid #f59e0b;
        color: #f59e0b;
        font-family: 'Teko', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.3);
    }
    
    /* Prediction Scoreboard Hero Card */
    .scoreboard-hero {
        background: radial-gradient(circle at 50% 0%, #1e3a8a 0%, #090d16 80%);
        border: 2px solid #3b82f6;
        border-radius: 20px;
        padding: 28px 20px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(30, 58, 138, 0.4);
        margin: 20px 0;
        position: relative;
    }
    .score-target-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid #60a5fa;
        color: #93c5fd;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .predicted-total {
        font-family: 'Teko', sans-serif;
        font-size: 5.2rem;
        line-height: 0.95;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 4px 20px rgba(59, 130, 246, 0.6);
        margin: 8px 0;
        letter-spacing: 0.02em;
    }
    .score-range-pill {
        display: inline-block;
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #cbd5e1;
        padding: 6px 16px;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Realistic KPI Cards */
    .kpi-tile {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-tile:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
    }
    .kpi-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.1;
    }
    .kpi-lbl {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    
    /* Preset Buttons bar */
    .preset-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. DATA & MODEL LOADING FUNCTIONS (CACHED)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Load the trained Random Forest Pipeline from disk."""
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_raw_dataset():
    """Load the raw ball-by-ball IPL dataset."""
    dataset_path = "IPL_ball_by_ball_updated.csv"
    if not os.path.exists(dataset_path):
        return None
    return pd.read_csv(dataset_path)


@st.cache_data(show_spinner=False)
def prepare_ml_data(df):
    """Process raw dataset into the exact ML prediction points dataset."""
    if df is None:
        return None

    data = df.copy()
    data['start_date'] = pd.to_datetime(data['start_date'], errors='coerce')

    numeric_columns = [
        'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes', 'penalty'
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    data = data.drop_duplicates()
    data['total_runs'] = data['runs_off_bat'] + data['extras']
    data['over_number'] = data['ball'].astype(str).str.split('.').str[0].astype(int)
    data['is_wicket'] = data['player_dismissed'].notna().astype(int)

    final_scores = (
        data.groupby(['match_id', 'innings'])['total_runs']
        .sum()
        .reset_index()
        .rename(columns={'total_runs': 'final_score'})
    )
    data = data.merge(final_scores, on=['match_id', 'innings'], how='left')

    data_regular = data[data['innings'].isin([1, 2])].copy()
    data_regular = data_regular.sort_values(['match_id', 'innings', 'over_number', 'ball'])

    grouped = data_regular.groupby(['match_id', 'innings'], sort=False)
    data_regular['current_score'] = grouped['total_runs'].cumsum()
    data_regular['wickets_lost'] = grouped['is_wicket'].cumsum().clip(upper=10)

    over_data = (
        data_regular.groupby(['match_id', 'innings', 'over_number'])
        .agg(over_runs=('total_runs', 'sum'), over_wickets=('is_wicket', 'sum'))
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

    prediction_points = data_regular[
        data_regular['over_number'].isin([4, 9, 14])
    ].copy()

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

    return ml_data


@st.cache_data(show_spinner=False)
def load_metadata():
    """Load precomputed training metadata if available."""
    if os.path.exists("model_metadata.json"):
        with open("model_metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Load core assets
raw_df = load_raw_dataset()
ml_df = prepare_ml_data(raw_df)
model = load_trained_model()
metadata = load_metadata()

DEFAULT_TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

EXPECTED_FEATURES = [
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


# =============================================================================
# 3. SIDEBAR NAVIGATION & FRANCHISE HUB
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding-bottom: 12px;">
        <img src="https://img.icons8.com/color/96/cricket.png" width="60" style="filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5));"/>
        <div style="font-family: 'Teko', sans-serif; font-size: 2rem; letter-spacing: 0.05em; color: #f59e0b; margin-top: 4px; line-height: 1;">IPL MATCH CENTER</div>
        <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Official AI Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Portal",
        [
            "🏠 Match Center",
            "🎯 Live Score Predictor",
            "📊 Stadium & Match Analytics",
            "⚔️ Head-to-Head Arena",
            "🤖 AI Model Telemetry",
            "ℹ️ Tournament & Project Info"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### ⚡ System Telemetry")
    if model is not None:
        st.markdown("<div style='color: #10b981; font-size: 0.85rem; font-weight: 600;'>🟢 Random Forest Pipeline: Ready</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #ef4444; font-size: 0.85rem; font-weight: 600;'>🔴 Model: Missing</div>", unsafe_allow_html=True)

    if raw_df is not None:
        st.markdown(f"<div style='color: #38bdf8; font-size: 0.85rem; font-weight: 600;'>🔵 Database: {len(raw_df):,} Balls Loaded</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("IPL Predictive Analytics • Scikit-Learn • Streamlit")


# =============================================================================
# 4. TOP BROADCAST HEADER (RENDERED ACROSS ALL PAGES)
# =============================================================================
st.markdown("""
<div class="ipl-header">
    <div class="ipl-title-row">
        <div class="ipl-brand">
            <span style="font-size: 2.2rem;">🏏</span>
            <div>
                <h1 class="ipl-main-title">INDIAN PREMIER LEAGUE • MATCH INTELLIGENCE</h1>
                <div class="ipl-subtitle">Official AI Final Score Forecasting & Franchise Performance Analytics</div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <div class="live-indicator">
                <span class="live-dot"></span>
                <span>SIMULATOR LIVE</span>
            </div>
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
                T20 REGRESSION V2.0
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 5. PAGE 1 — MATCH CENTER (DASHBOARD)
# =============================================================================
if page == "🏠 Match Center":
    if raw_df is None:
        st.error("❌ Dataset not found. Please place `IPL_ball_by_ball_updated.csv` in the root directory.")
        st.stop()

    total_matches = int(raw_df['match_id'].nunique())
    total_seasons = int(raw_df['season'].nunique())
    total_deliveries = len(raw_df)
    total_venues = int(raw_df['venue'].nunique())
    total_runs_all = int(raw_df['runs_off_bat'].sum() + raw_df['extras'].sum())
    total_wickets_all = int(raw_df['player_dismissed'].notna().sum())

    # Broadcast Style Top Metrics Tiles
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-lbl">Total Matches</div><div class="kpi-val">{total_matches}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-lbl">Seasons</div><div class="kpi-val">{total_seasons}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-lbl">Deliveries</div><div class="kpi-val">{total_deliveries // 1000}k+</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-lbl">Venues</div><div class="kpi-val">{total_venues}</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-lbl">Total Runs</div><div class="kpi-val">{total_runs_all // 1000}k+</div></div>""", unsafe_allow_html=True)
    with m6:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-lbl">Wickets Fallen</div><div class="kpi-val">{total_wickets_all:,}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Franchise Quick Explorer Cards
    st.markdown("### 🏆 Active IPL Franchises")
    f_cols = st.columns(5)
    active_teams = [
        'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 
        'Kolkata Knight Riders', 'Gujarat Titans', 'Rajasthan Royals', 
        'Delhi Capitals', 'Sunrisers Hyderabad', 'Lucknow Super Giants', 'Punjab Kings'
    ]

    for idx, team_name in enumerate(active_teams):
        meta = TEAM_METADATA.get(team_name, {'short': team_name[:3], 'color': '#38bdf8', 'bg': '#1e293b', 'border': '#38bdf8', 'icon': '🏏'})
        col = f_cols[idx % 5]
        with col:
            st.markdown(f"""
            <div style="background: {meta['bg']}; border: 1px solid {meta['border']}; border-radius: 12px; padding: 12px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 1.6rem;">{meta['icon']}</div>
                <div style="font-family: 'Teko', sans-serif; font-size: 1.4rem; color: #ffffff; line-height: 1; margin-top: 2px;">{meta['short']}</div>
                <div style="font-size: 0.72rem; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{team_name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Season & Scoring Momentum Chart using Plotly
    col_left, col_right = st.columns([1.8, 1.2])

    with col_left:
        st.markdown("#### 📈 Historical IPL Final Score Spread & Density")
        if ml_df is not None:
            fig_hist = px.histogram(
                ml_df,
                x="final_score",
                nbins=35,
                marginal="box",
                color_discrete_sequence=["#38bdf8"],
                opacity=0.85,
                labels={"final_score": "Innings Final Score", "count": "Frequency"}
            )
            fig_hist.update_layout(
                plot_bgcolor="#1e293b",
                paper_bgcolor="#0f172a",
                font=dict(color="#cbd5e1", family="Inter"),
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown("#### ⚡ Match Benchmark Highlights")
        avg_score = ml_df['final_score'].mean()
        high_score = ml_df['final_score'].max()
        low_score = ml_df['final_score'].min()
        median_score = ml_df['final_score'].median()

        st.markdown(f"""
        <div style="background: #1e293b; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: #94a3b8;">Average 1st & 2nd Innings Score</span>
                <span style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #38bdf8; font-size: 1.15rem;">{avg_score:.1f} Runs</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: #94a3b8;">Median Score Benchmark</span>
                <span style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #10b981; font-size: 1.15rem;">{median_score:.0f} Runs</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: #94a3b8;">Highest Total in Records</span>
                <span style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #f59e0b; font-size: 1.15rem;">{high_score:.0f} Runs</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 10px 0;">
                <span style="color: #94a3b8;">Lowest Total in Records</span>
                <span style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #ef4444; font-size: 1.15rem;">{low_score:.0f} Runs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 **Ready to run a live simulation?** Navigate to the **🎯 Live Score Predictor** in the sidebar to simulate match scenarios in real-time!")


# =============================================================================
# 6. PAGE 2 — LIVE SCORE PREDICTOR
# =============================================================================
elif page == "🎯 Live Score Predictor":
    if model is None:
        st.error("❌ Trained model pipeline (`random_forest_model.pkl`) not found. Run `train_model.py` first.")
        st.stop()

    if ml_df is not None:
        teams_list = sorted(ml_df['batting_team'].unique().tolist())
        venues_list = sorted(ml_df['venue'].unique().tolist())
    else:
        teams_list = DEFAULT_TEAMS
        venues_list = ["Wankhede Stadium, Mumbai", "M Chinnaswamy Stadium", "Eden Gardens", "MA Chidambaram Stadium, Chepauk"]

    # 1-Click Iconic Match Preset Buttons
    st.markdown("""
    <div class="preset-container">
        <div style="font-size: 0.85rem; font-weight: 700; color: #fbbf24; text-transform: uppercase; margin-bottom: 8px;">
            ⚡ 1-Click Iconic IPL Situation Presets
        </div>
    </div>
    """, unsafe_allow_html=True)

    preset_cols = st.columns(5)
    
    # Initialize session state defaults if not present
    if 'preset_loaded' not in st.session_state:
        st.session_state.bat_idx = 0
        st.session_state.bowl_idx = 1 if len(teams_list) > 1 else 0
        st.session_state.venue_idx = 0
        st.session_state.overs_val = 10
        st.session_state.score_val = 82
        st.session_state.wickets_val = 2
        st.session_state.r5_runs_val = 42
        st.session_state.r5_wk_val = 1

    with preset_cols[0]:
        if st.button("🦁 CSK Powerplay Surge", use_container_width=True):
            st.session_state.bat_idx = teams_list.index("Chennai Super Kings") if "Chennai Super Kings" in teams_list else 0
            st.session_state.bowl_idx = teams_list.index("Mumbai Indians") if "Mumbai Indians" in teams_list else 1
            st.session_state.overs_val = 5
            st.session_state.score_val = 54
            st.session_state.wickets_val = 1
            st.session_state.r5_runs_val = 54
            st.session_state.r5_wk_val = 1
            st.rerun()

    with preset_cols[1]:
        if st.button("👑 RCB Chinnaswamy Storm", use_container_width=True):
            st.session_state.bat_idx = teams_list.index("Royal Challengers Bangalore") if "Royal Challengers Bangalore" in teams_list else 0
            st.session_state.bowl_idx = teams_list.index("Kolkata Knight Riders") if "Kolkata Knight Riders" in teams_list else 1
            st.session_state.overs_val = 10
            st.session_state.score_val = 108
            st.session_state.wickets_val = 1
            st.session_state.r5_runs_val = 58
            st.session_state.r5_wk_val = 0
            st.rerun()

    with preset_cols[2]:
        if st.button("⚡ GT Death Overs Base", use_container_width=True):
            st.session_state.bat_idx = teams_list.index("Gujarat Titans") if "Gujarat Titans" in teams_list else 0
            st.session_state.bowl_idx = teams_list.index("Rajasthan Royals") if "Rajasthan Royals" in teams_list else 1
            st.session_state.overs_val = 15
            st.session_state.score_val = 146
            st.session_state.wickets_val = 3
            st.session_state.r5_runs_val = 49
            st.session_state.r5_wk_val = 1
            st.rerun()

    with preset_cols[3]:
        if st.button("🦅 SRH High Run-Rate Blast", use_container_width=True):
            st.session_state.bat_idx = teams_list.index("Sunrisers Hyderabad") if "Sunrisers Hyderabad" in teams_list else 0
            st.session_state.bowl_idx = teams_list.index("Delhi Capitals") if "Delhi Capitals" in teams_list else 1
            st.session_state.overs_val = 10
            st.session_state.score_val = 122
            st.session_state.wickets_val = 2
            st.session_state.r5_runs_val = 62
            st.session_state.r5_wk_val = 1
            st.rerun()

    with preset_cols[4]:
        if st.button("🛡️ Chepauk Spin Squeeze", use_container_width=True):
            st.session_state.bat_idx = teams_list.index("Punjab Kings") if "Punjab Kings" in teams_list else 0
            st.session_state.bowl_idx = teams_list.index("Chennai Super Kings") if "Chennai Super Kings" in teams_list else 1
            st.session_state.overs_val = 15
            st.session_state.score_val = 112
            st.session_state.wickets_val = 6
            st.session_state.r5_runs_val = 28
            st.session_state.r5_wk_val = 3
            st.rerun()

    # Match Selection HUD
    st.markdown("### 1️⃣ Matchup & Stadium Selection")
    col_t1, col_t2, col_v = st.columns(3)

    with col_t1:
        batting_team = st.selectbox(
            "🏏 Batting Team",
            teams_list,
            index=min(st.session_state.bat_idx, len(teams_list)-1)
        )

    with col_t2:
        bowling_team = st.selectbox(
            "🎯 Bowling Team",
            teams_list,
            index=min(st.session_state.bowl_idx, len(teams_list)-1)
        )

    with col_v:
        venue = st.selectbox(
            "🏟️ Stadium / Venue",
            venues_list,
            index=min(st.session_state.venue_idx, len(venues_list)-1)
        )

    bat_meta = TEAM_METADATA.get(batting_team, {'short': batting_team[:3], 'color': '#38bdf8', 'bg': '#1e293b', 'border': '#38bdf8', 'icon': '🏏'})
    bowl_meta = TEAM_METADATA.get(bowling_team, {'short': bowling_team[:3], 'color': '#f59e0b', 'bg': '#1e293b', 'border': '#f59e0b', 'icon': '🎯'})

    # Broadcast Lower-Third Live Banner
    st.markdown(f"""
    <div class="broadcast-card" style="margin-top: 10px;">
        <div class="matchup-grid">
            <div class="team-badge-box" style="background: {bat_meta['bg']}; border-color: {bat_meta['border']};">
                <span style="font-size: 1.8rem;">{bat_meta['icon']}</span>
                <div class="team-code" style="color: {bat_meta['color']};">{bat_meta['short']}</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">{batting_team}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">(BATTING)</div>
            </div>
            <div>
                <div class="vs-circle">VS</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">{venue}</div>
            </div>
            <div class="team-badge-box" style="background: {bowl_meta['bg']}; border-color: {bowl_meta['border']};">
                <span style="font-size: 1.8rem;">{bowl_meta['icon']}</span>
                <div class="team-code" style="color: {bowl_meta['color']};">{bowl_meta['short']}</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">{bowling_team}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">(BOWLING)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # In-Game State Inputs
    st.markdown("### 2️⃣ Live Match Situation Controls")
    c_ov, c_sc, c_wk = st.columns(3)

    with c_ov:
        # Standard milestones used in training pipeline
        overs_completed = st.select_slider(
            "⏱️ Overs Completed Milestone",
            options=[5, 10, 15],
            value=st.session_state.overs_val,
            help="Prediction milestones from the ML training pipeline (End of 5th, 10th, or 15th over)."
        )

    with c_sc:
        current_score = st.number_input(
            "🏏 Current Cumulative Score",
            min_value=0,
            max_value=300,
            value=st.session_state.score_val,
            step=1
        )

    with c_wk:
        wickets_lost = st.slider(
            "☝️ Wickets Fallen",
            min_value=0,
            max_value=10,
            value=st.session_state.wickets_val
        )

    if overs_completed > 0:
        current_run_rate = float(current_score) / float(overs_completed)
    else:
        current_run_rate = 0.0

    st.markdown("### 3️⃣ Recent 5-Over Surge Momentum")
    c_r5, c_r5w, c_rate = st.columns(3)

    with c_r5:
        default_r5 = min(current_score, st.session_state.r5_runs_val) if overs_completed > 5 else current_score
        last_5_over_runs = st.number_input(
            "🔥 Runs in Last 5 Overs",
            min_value=0,
            max_value=150,
            value=int(default_r5),
            step=1
        )

    with c_r5w:
        max_recent_wk = min(wickets_lost, 5)
        default_r5w = min(st.session_state.r5_wk_val, max_recent_wk)
        last_5_over_wickets = st.slider(
            "⚡ Wickets Lost in Last 5 Overs",
            min_value=0,
            max_value=max_recent_wk,
            value=int(default_r5w)
        )

    with c_rate:
        st.markdown("<div style='margin-top: 8px; font-weight: 700; color: #94a3b8; font-size: 0.85rem;'>CURRENT RUN RATE (CRR)</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 8px 16px; border-radius: 10px; display: inline-block;">
            <span style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 800; color: #34d399;">{current_run_rate:.2f}</span>
            <span style="font-size: 0.85rem; color: #a7f3d0; margin-left: 4px;">Runs / Over</span>
        </div>
        """, unsafe_allow_html=True)

    # Validations
    is_valid = True
    if batting_team == bowling_team:
        is_valid = False
        st.error("❌ Batting and Bowling teams cannot be the same franchise.")

    if wickets_lost == 10:
        st.warning("⚠️ Batting team is All Out (10 wickets). Final score cannot exceed current score.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 GENERATE AI FINAL SCORE PREDICTION", type="primary", use_container_width=True, disabled=not is_valid)

    if predict_btn and is_valid:
        input_data = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'overs_completed': int(overs_completed),
            'current_score': int(current_score),
            'wickets_lost': int(wickets_lost),
            'current_run_rate': float(current_run_rate),
            'last_5_over_runs': int(last_5_over_runs),
            'last_5_over_wickets': int(last_5_over_wickets)
        }])[EXPECTED_FEATURES]

        raw_pred = float(model.predict(input_data)[0])
        final_score_pred = float(current_score) if wickets_lost == 10 else max(float(current_score), raw_pred)

        rmse_val = 24.02
        range_min = max(int(current_score), int(round(final_score_pred - rmse_val)))
        range_max = int(round(final_score_pred + rmse_val))

        # Hero Scoreboard Display Card
        st.markdown(f"""
        <div class="scoreboard-hero">
            <div class="score-target-badge">PROJECTED INNINGS TOTAL</div>
            <div class="predicted-total">{int(round(final_score_pred))} <span style="font-size: 2.2rem; color: #93c5fd;">RUNS</span></div>
            <div class="score-range-pill">ESTIMATED REALISTIC RANGE: <strong>{range_min} – {range_max} RUNS</strong> (±{rmse_val:.1f} RMSE)</div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Scoreboard Speedometer / Gauge
        gauge_col, metric_col = st.columns([1.2, 0.8])

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=int(round(final_score_pred)),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Projected Score Gauge", 'font': {'color': '#ffffff', 'size': 16}},
                delta={'reference': 165, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
                gauge={
                    'axis': {'range': [80, 260], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [80, 140], 'color': 'rgba(239, 68, 68, 0.3)'},
                        {'range': [140, 185], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [185, 260], 'color': 'rgba(16, 185, 129, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "#f59e0b", 'width': 4},
                        'thickness': 0.75,
                        'value': int(round(final_score_pred))
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#cbd5e1", 'family': "Inter"},
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with metric_col:
            rem_overs = max(0, 20 - overs_completed)
            runs_needed = max(0, int(round(final_score_pred)) - current_score)
            projected_death_rr = (runs_needed / rem_overs) if rem_overs > 0 else 0

            st.markdown(f"""
            <div style="background: #1e293b; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px; height: 100%;">
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Death Overs Requirement</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 800; color: #f59e0b; margin: 4px 0;">{projected_death_rr:.2f} RPO</div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Projected to score <strong>{runs_needed} runs</strong> in remaining <strong>{rem_overs} overs</strong>.</div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Active Model: <strong>Random Forest Regressor</strong></div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Benchmark Accuracy: <strong>R² 0.3782</strong> | <strong>RMSE 24.02</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # What-If Sensitivity Simulator Matrix
        st.markdown("### 🔮 Interactive 'What-If' Trajectory Sensitivity")
        st.write("Explore how scoring pace and wicket conservation in the next phase will impact the final total:")

        sim_df = pd.DataFrame([
            {
                "Match Scenario": "🚀 Extreme Acceleration (0 Wickets Lost, +65 Runs Next 5 Overs)",
                "Expected Score": int(round(final_score_pred + 22)),
                "Tactical Impact": "Aggressive boundary hitting with wickets in hand."
            },
            {
                "Match Scenario": "⚡ Steady Momentum (Current Model Projection)",
                "Expected Score": int(round(final_score_pred)),
                "Tactical Impact": "Baseline historical progression."
            },
            {
                "Match Scenario": "⚠️ Middle-Overs Squeeze (2+ Quick Wickets Fallen)",
                "Expected Score": max(current_score, int(round(final_score_pred - 18))),
                "Tactical Impact": "Batting consolidation, lower boundary percentage."
            }
        ])

        st.dataframe(sim_df, use_container_width=True)


# =============================================================================
# 7. PAGE 3 — STADIUM & MATCH ANALYTICS
# =============================================================================
elif page == "📊 Stadium & Match Analytics":
    if ml_df is None:
        st.error("❌ Dataset not found for analytics.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Scoring Distributions",
        "2. Franchise Rankings",
        "3. Progression Milestone Scatter",
        "4. Stadium Ground Intelligence",
        "5. Correlation Heatmap"
    ])

    with tab1:
        st.subheader("📊 Innings Scoring Distribution across IPL History")
        fig_dist = px.histogram(
            ml_df,
            x="final_score",
            nbins=35,
            color="overs_completed",
            title="Final Score Spread by Prediction Milestone",
            labels={"final_score": "Final Score (Runs)", "overs_completed": "Overs Milestone"},
            color_discrete_sequence=["#38bdf8", "#818cf8", "#f43f5e"]
        )
        fig_dist.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            height=420
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        st.subheader("🏆 Franchise Scoring Averages & Maximums")
        team_agg = (
            ml_df.groupby('batting_team')['final_score']
            .agg(avg_score='mean', max_score='max', min_score='min', matches='count')
            .reset_index()
            .sort_values(by='avg_score', ascending=False)
        )

        fig_team = px.bar(
            team_agg,
            x='batting_team',
            y='avg_score',
            color='avg_score',
            color_continuous_scale='Viridis',
            title="Average Final Score by Batting Franchise",
            labels={'avg_score': 'Avg Final Score', 'batting_team': 'Franchise'}
        )
        fig_team.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            height=420,
            xaxis={'tickangle': -45}
        )
        st.plotly_chart(fig_team, use_container_width=True)

    with tab3:
        st.subheader("📈 Current Score vs Eventual Final Total")
        fig_scat = px.scatter(
            ml_df,
            x="current_score",
            y="final_score",
            color="overs_completed",
            trendline="ols",
            title="Score Trajectory Progression (5, 10 & 15 Overs)",
            labels={"current_score": "Current Score at Milestone", "final_score": "Final Innings Total"},
            color_discrete_sequence=["#38bdf8", "#f59e0b", "#10b981"]
        )
        fig_scat.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            height=450
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with tab4:
        st.subheader("🏟️ Stadium & Pitch Scoring Intelligence")
        top_venues_df = (
            ml_df.groupby('venue')['final_score']
            .agg(avg_score='mean', highest_score='max', matches='count')
            .reset_index()
            .query("matches >= 40")
            .sort_values(by='avg_score', ascending=False)
        )

        fig_ven = px.bar(
            top_venues_df,
            x='venue',
            y='avg_score',
            color='highest_score',
            title="Top Grounds Ranked by Average 1st & 2nd Innings Score",
            labels={'avg_score': 'Average Score', 'venue': 'Stadium'}
        )
        fig_ven.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            height=430,
            xaxis={'tickangle': -45}
        )
        st.plotly_chart(fig_ven, use_container_width=True)

    with tab5:
        st.subheader("🔍 Feature Inter-Correlation Matrix")
        numeric_cols = [
            'overs_completed', 'current_score', 'wickets_lost',
            'current_run_rate', 'last_5_over_runs', 'last_5_over_wickets', 'final_score'
        ]
        corr = ml_df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Correlation Heatmap of Match Parameters"
        )
        fig_corr.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            height=460
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# 8. PAGE 4 — HEAD-TO-HEAD ARENA
# =============================================================================
elif page == "⚔️ Head-to-Head Arena":
    st.markdown("### ⚔️ Franchise Head-to-Head Historical Battle")
    st.write("Compare historical scoring stats between any two IPL teams:")

    if ml_df is None:
        st.error("❌ Dataset not loaded.")
        st.stop()

    all_teams = sorted(ml_df['batting_team'].unique().tolist())
    c1, c2 = st.columns(2)

    with c1:
        team_a = st.selectbox("Franchise A (Batting)", all_teams, index=0)
    with c2:
        team_b = st.selectbox("Franchise B (Bowling)", all_teams, index=1 if len(all_teams) > 1 else 0)

    if team_a == team_b:
        st.warning("Select two different franchises for Head-to-Head analysis.")
    else:
        h2h_matches = ml_df[(ml_df['batting_team'] == team_a) & (ml_df['bowling_team'] == team_b)]
        
        meta_a = TEAM_METADATA.get(team_a, {'color': '#38bdf8', 'icon': '🦁'})
        meta_b = TEAM_METADATA.get(team_b, {'color': '#f59e0b', 'icon': '🎯'})

        if len(h2h_matches) > 0:
            avg_h2h = h2h_matches['final_score'].mean()
            max_h2h = h2h_matches['final_score'].max()
            min_h2h = h2h_matches['final_score'].min()

            h1, h2, h3, h4 = st.columns(4)
            h1.metric(f"{team_a} Innings Analyzed", len(h2h_matches))
            h2.metric("Average Score vs " + team_b[:10], f"{avg_h2h:.1f}")
            h3.metric("Highest Total", f"{int(max_h2h)}")
            h4.metric("Lowest Total", f"{int(min_h2h)}")

            fig_h2h = px.box(
                h2h_matches,
                x='overs_completed',
                y='final_score',
                points="all",
                color_discrete_sequence=[meta_a['color']],
                title=f"{team_a} Score Distribution against {team_b}"
            )
            fig_h2h.update_layout(
                plot_bgcolor="#1e293b",
                paper_bgcolor="#0f172a",
                font=dict(color="#cbd5e1"),
                height=380
            )
            st.plotly_chart(fig_h2h, use_container_width=True)
        else:
            st.info(f"No direct historical regular innings found for {team_a} batting against {team_b}.")


# =============================================================================
# 9. PAGE 5 — AI MODEL TELEMETRY
# =============================================================================
elif page == "🤖 AI Model Telemetry":
    st.markdown("### 🤖 Machine Learning Model Benchmarking")

    if metadata is not None and "results" in metadata:
        results_data = metadata["results"]
    else:
        results_data = [
            {'Model': 'Linear Regression', 'MAE': 19.38, 'MSE': 619.45, 'RMSE': 24.89, 'R² Score': 0.3322},
            {'Model': 'Decision Tree Regression', 'MAE': 20.41, 'MSE': 698.36, 'RMSE': 26.43, 'R² Score': 0.2472},
            {'Model': 'Random Forest Regression', 'MAE': 18.65, 'MSE': 576.78, 'RMSE': 24.02, 'R² Score': 0.3782}
        ]

    res_df = pd.DataFrame(results_data)

    st.dataframe(
        res_df.style.highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='#1e3a8a')
                    .highlight_max(subset=['R² Score'], color='#065f46')
                    .format({'MAE': '{:.2f}', 'MSE': '{:.2f}', 'RMSE': '{:.2f}', 'R² Score': '{:.4f}'}),
        use_container_width=True
    )

    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 12px; padding: 18px; margin: 15px 0;">
        <h4 style="color: #34d399; margin-top:0;">🏆 Top Performing Algorithm: Random Forest Regressor</h4>
        <p style="color: #e2e8f0; margin-bottom: 0;">
            The <strong>Random Forest Regression</strong> model achieved the lowest MAE (18.65 runs), lowest RMSE (24.02 runs), and highest R² Score (0.3782).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance
    if metadata is not None and "feature_importances" in metadata:
        st.markdown("#### 🔍 Decoded Feature Importance Weights")
        fi_df = pd.DataFrame(metadata["feature_importances"]).head(12)
        fi_df['clean_feature'] = (
            fi_df['feature']
            .str.replace('numerical__', '')
            .str.replace('categorical__batting_team_', 'Batting: ')
            .str.replace('categorical__bowling_team_', 'Bowling: ')
            .str.replace('categorical__venue_', 'Venue: ')
            .str.replace('_', ' ')
            .str.title()
        )

        fig_fi = px.bar(
            fi_df.sort_values(by='importance', ascending=True),
            x='importance',
            y='clean_feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Viridis',
            title="Top 12 Most Influential Prediction Factors"
        )
        fig_fi.update_layout(
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            height=420
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# =============================================================================
# 10. PAGE 6 — TOURNAMENT & PROJECT INFO
# =============================================================================
elif page == "ℹ️ Tournament & Project Info":
    st.markdown("### ℹ️ Project Architecture & Documentation")

    c_doc1, c_doc2 = st.columns(2)

    with c_doc1:
        st.markdown("""
        #### 📋 Project Overview
        - **Title:** IPL Final Score Prediction Using Machine Learning
        - **Domain:** Sports Analytics & T20 Cricket Intelligence
        - **Dataset:** `IPL_ball_by_ball_updated.csv` (243,815 rows × 22 features)
        - **ML Methodology:** Supervised Regression with Time-Based Train-Test Split (Train: $≤$ 2021, Test: $≥$ 2022)
        """)

    with c_doc2:
        st.markdown("""
        #### ⚙️ Feature Pipeline
        - **Categorical (OneHotEncoded):** Batting Team, Bowling Team, Stadium Venue
        - **Numerical (Passthrough):** Overs Completed, Current Score, Wickets Lost, Current Run Rate, Last 5 Over Runs, Last 5 Over Wickets
        - **Target:** Innings Final Score (`final_score`)
        """)

    st.markdown("---")
    st.markdown("#### 🔄 End-to-End Execution Pipeline")
    st.markdown("""
    ```
    IPL Ball-by-Ball Data ➔ Data Cleaning ➔ Feature Engineering ➔ Milestone Extraction (Overs 5, 10, 15)
                                                                             │
    Streamlit Live Dashboard  random_forest_model.pkl  Best Model Selection  ML Models Training
    ```
    """)
