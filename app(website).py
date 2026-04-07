"""
Website Traffic Forecasting — Streamlit Web App
First Quadrant Labs | Afeef Anver sha
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, os, joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Forecaster — First Quadrant Labs",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.02em; }

.main { background-color: #F8FAFC; }
.stSidebar { background-color: #0F172A !important; }
.stSidebar .stMarkdown, .stSidebar label, .stSidebar .stSelectbox label {
    color: #CBD5E1 !important;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    color: #F1F5F9 !important;
}

div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 0.75rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.7rem !important;
    color: #0F172A !important;
    font-weight: 600;
}

.banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    color: white;
}
.banner h1 { color: white !important; font-size: 1.9rem; margin: 0 0 6px 0; }
.banner p  { color: #94A3B8; margin: 0; font-size: 0.95rem; }

.section-tag {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
    letter-spacing: 0.06em;
}

.stButton button {
    background: #1D4ED8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: background 0.2s;
}
.stButton button:hover { background: #1E40AF !important; }

.insight-card {
    background: #F0FDF4;
    border-left: 4px solid #16A34A;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #14532D;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = {'SARIMA':'#1D4ED8','Prophet':'#15803D','Random Forest':'#B91C1C',
           'XGBoost':'#B45309','LSTM':'#6D28D9'}
NUM_COLS = ['Page_Loads','Unique_Visits','First_Time_Visits','Returning_Visits']

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace('.','_',regex=False).str.replace(' ','_')
    for col in NUM_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',','',regex=False).astype(float)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df.sort_values('Date').reset_index(drop=True)
    Q1,Q3 = df['Page_Loads'].quantile([0.25,0.75])
    df['Page_Loads'] = df['Page_Loads'].clip(lower=Q1-3*(Q3-Q1), upper=Q3+3*(Q3-Q1))
    df['DayName'] = df['Date'].dt.day_name()
    return df

def build_features(df):
    f = df.copy()
    f['day_of_week']   = f['Date'].dt.dayofweek
    f['day_of_month']  = f['Date'].dt.day
    f['month']         = f['Date'].dt.month
    f['week_of_year']  = f['Date'].dt.isocalendar().week.astype(int)
    f['quarter']       = f['Date'].dt.quarter
    f['year']          = f['Date'].dt.year
    f['is_weekend']    = (f['day_of_week'] >= 5).astype(int)
    for k in [1,2]:
        f[f'sin_week_{k}'] = np.sin(2*np.pi*k*f['day_of_week']/7)
        f[f'cos_week_{k}'] = np.cos(2*np.pi*k*f['day_of_week']/7)
    for lag in [1,2,3,7,14,21,28]:
        f[f'lag_{lag}'] = f['Page_Loads'].shift(lag)
    for w in [7,14,30]:
        s = f['Page_Loads'].shift(1)
        f[f'roll_mean_{w}'] = s.rolling(w).mean()
        f[f'roll_std_{w}']  = s.rolling(w).std()
        f[f'roll_max_{w}']  = s.rolling(w).max()
        f[f'roll_min_{w}']  = s.rolling(w).min()
    f['exp_mean']       = f['Page_Loads'].shift(1).expanding().mean()
    f['return_ratio']   = f['Returning_Visits']  / (f['Unique_Visits']+1)
    f['new_ratio']      = f['First_Time_Visits'] / (f['Unique_Visits']+1)
    f['loads_per_visit']= f['Page_Loads']        / (f['Unique_Visits']+1)
    for col in ['Unique_Visits','Returning_Visits']:
        f[f'{col}_lag1'] = f[col].shift(1)
        f[f'{col}_lag7'] = f[col].shift(7)

    # Drop ALL object/string columns except Date.
    # Raw CSV columns like 'Day', 'DayName', 'Day_Of_Week' cause
    # sklearn's 'could not convert string to float' error if they
    # slip through into FEAT_COLS.
    str_cols = [c for c in f.columns if f[c].dtype == object and c != 'Date']
    f = f.drop(columns=str_cols, errors='ignore')

    return f.dropna().reset_index(drop=True)

def metric_row(y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = mean_absolute_percentage_error(y_true, y_pred)*100
    mae   = np.mean(np.abs(y_true-y_pred))
    r2    = r2_score(y_true, y_pred)
    return rmse, mae, mape, r2

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Traffic Forecaster")
    st.markdown("*First Quadrant Labs*")
    st.markdown("---")

    uploaded = st.file_uploader("Upload traffic CSV", type=['csv'],
                                help="Columns: Date, Page.Loads, Unique.Visits, etc.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    split_pct    = st.slider("Train/Test Split %", 60, 90, 80, step=5)
    forecast_days= st.slider("Forecast Horizon (days)", 7, 90, 30)

    st.markdown("---")
    st.markdown("### 🤖 Models to Train")
    use_rf      = st.checkbox("Random Forest",   value=True)
    use_xgb     = st.checkbox("XGBoost",         value=True)
    use_prophet = st.checkbox("Prophet",         value=True)
    use_sarima  = st.checkbox("SARIMA",          value=False,
                              help="Slower — enable for full comparison")

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown('<p style="color:#475569;font-size:0.78rem;">Execution order:<br>'
                'NB1 → NB3 → NB2 → App</p>', unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>Website Traffic Forecasting</h1>
  <p>First Quadrant Labs · Afeef Anver sha · Multi-model daily traffic prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Sample data fallback ───────────────────────────────────────────────────────
if uploaded is None:
    st.info("👆  Upload your CSV in the sidebar to begin, or see the demo below with sample data.")

    sample_rows = [
        ("9/14/2014","Sunday",1,2146,1582,1430,152),
        ("9/15/2014","Monday",2,3621,2528,2297,231),
        ("9/16/2014","Tuesday",3,3698,2630,2352,278),
        ("9/17/2014","Wednesday",4,3667,2614,2327,287),
        ("9/18/2014","Thursday",5,3316,2366,2130,236),
        ("9/19/2014","Friday",6,2815,1863,1622,241),
    ]
    demo_df = pd.DataFrame(sample_rows,
        columns=["Date","Day","Day_Of_Week","Page_Loads","Unique_Visits","First_Time_Visits","Returning_Visits"])
    st.markdown("**Sample data structure:**")
    st.dataframe(demo_df, use_container_width=True)
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
df = load_data(uploaded)

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 EDA", "🤖 Train & Evaluate", "🔮 Forecast", "📋 Report"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<span class="section-tag">EXPLORATORY ANALYSIS</span>', unsafe_allow_html=True)

    # KPI cards
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Avg Page Loads",  f"{df['Page_Loads'].mean():,.0f}")
    c3.metric("Peak Day",        f"{df['Page_Loads'].max():,.0f}")
    c4.metric("Avg Unique Visits",f"{df['Unique_Visits'].mean():,.0f}")
    c5.metric("Date Range",      f"{(df['Date'].max()-df['Date'].min()).days} days")

    st.markdown("---")

    # Time series
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=NUM_COLS, vertical_spacing=0.05)
    colors_ts = ['#1D4ED8','#15803D','#B91C1C','#B45309']

    def hex_to_rgba(hex_color, alpha=0.12):
        """Convert #RRGGBB hex to 'rgba(r,g,b,alpha)' for Plotly."""
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    for i, (col, color) in enumerate(zip(NUM_COLS, colors_ts), 1):
        fig.add_trace(go.Scatter(x=df['Date'], y=df[col], name=col,
                                 line=dict(color=color, width=1.5),
                                 fill='tozeroy', fillcolor=hex_to_rgba(color, 0.12)),
                      row=i, col=1)
    fig.update_layout(height=700, title="Traffic Metrics Over Time",
                      showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = df.groupby('DayName')['Page_Loads'].mean().reindex(day_order)
        bar_colors = ['#1D4ED8' if d not in ['Saturday','Sunday'] else '#94A3B8' for d in day_order]
        fig2 = go.Figure(go.Bar(x=dow.index, y=dow.values,
                                marker_color=bar_colors,
                                text=dow.values.round(0),
                                textposition='outside'))
        fig2.update_layout(title="Avg Page Loads by Day of Week",
                           template='plotly_white', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        df_monthly = df.copy()
        df_monthly['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly = df_monthly.groupby('Month')['Page_Loads'].mean().reset_index()
        fig3 = px.line(monthly, x='Month', y='Page_Loads', markers=True,
                       title="Monthly Average Page Loads",
                       color_discrete_sequence=['#15803D'])
        fig3.update_layout(template='plotly_white', height=350)
        fig3.update_xaxes(tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    # Seasonal decomposition
    st.markdown("---")
    st.markdown('<span class="section-tag">SEASONAL DECOMPOSITION (period=7)</span>',
                unsafe_allow_html=True)
    try:
        series_d = df.set_index('Date')['Page_Loads']
        decomp   = seasonal_decompose(series_d, model='additive', period=7)
        fig4 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             subplot_titles=['Observed','Trend','Seasonal','Residual'],
                             vertical_spacing=0.06)
        for i, (data, color) in enumerate(zip(
                [series_d, decomp.trend, decomp.seasonal, decomp.resid],
                ['#475569','#1D4ED8','#15803D','#B91C1C']), 1):
            fig4.add_trace(go.Scatter(x=data.index, y=data.values, line=dict(color=color, width=1.3)),
                           row=i, col=1)
        fig4.update_layout(height=600, showlegend=False, template='plotly_white')
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.warning(f"Decomposition: {e}")

    # Correlation
    st.markdown("---")
    corr = df[NUM_COLS].corr()
    fig5 = px.imshow(corr, text_auto='.3f', color_continuous_scale='Blues',
                     title="Correlation Matrix", aspect='auto',
                     zmin=0.8, zmax=1.0)
    fig5.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig5, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<span class="section-tag">MODEL TRAINING & EVALUATION</span>',
                unsafe_allow_html=True)

    if not run_btn:
        st.info("Click **▶ Run Analysis** in the sidebar to train models.")
    else:
        df_feat  = build_features(df)
        split_i  = int(len(df_feat) * split_pct / 100)
        train_df = df_feat.iloc[:split_i]
        test_df  = df_feat.iloc[split_i:]

        FEAT_COLS = [c for c in df_feat.columns
                     if c not in ['Date','DayName','Page_Loads'] + NUM_COLS]

        X_tr, y_tr = train_df[FEAT_COLS], train_df['Page_Loads']
        X_te, y_te = test_df[FEAT_COLS],  test_df['Page_Loads']

        results   = {}
        all_preds = {}

        progress = st.progress(0, text="Starting training...")

        # Random Forest
        if use_rf:
            progress.progress(10, "Training Random Forest...")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
            tscv = TimeSeriesSplit(n_splits=3)
            rf = RandomizedSearchCV(
                RandomForestRegressor(random_state=42),
                {'n_estimators':[100,200,300],'max_depth':[None,10,15],
                 'min_samples_leaf':[1,2,4],'max_features':['sqrt','log2']},
                n_iter=12, cv=tscv, scoring='neg_root_mean_squared_error',
                random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            rf_pred = rf.best_estimator_.predict(X_te)
            results['Random Forest'] = metric_row(y_te.values, rf_pred)
            all_preds['Random Forest'] = (test_df['Date'].values, y_te.values, rf_pred)

        # XGBoost
        if use_xgb:
            progress.progress(30, "Training XGBoost...")
            import xgboost as xgb_lib
            xgb_m = xgb_lib.XGBRegressor(n_estimators=300, max_depth=5,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, random_state=42,
                                          tree_method='hist')
            xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            xgb_pred = xgb_m.predict(X_te)
            results['XGBoost'] = metric_row(y_te.values, xgb_pred)
            all_preds['XGBoost'] = (test_df['Date'].values, y_te.values, xgb_pred)

        # Prophet
        if use_prophet:
            progress.progress(55, "Training Prophet...")
            from prophet import Prophet as Proph
            prop_tr = train_df[['Date','Page_Loads']].rename(columns={'Date':'ds','Page_Loads':'y'})
            prop_te = test_df[['Date','Page_Loads']].rename(columns={'Date':'ds','Page_Loads':'y'})
            pm = Proph(yearly_seasonality=True, weekly_seasonality=True,
                       daily_seasonality=False, seasonality_mode='multiplicative',
                       changepoint_prior_scale=0.05)
            pm.fit(prop_tr)
            fc = pm.predict(pm.make_future_dataframe(periods=len(prop_te)))
            prop_pred = np.maximum(
                fc[fc['ds'].isin(prop_te['ds'])]['yhat'].values, 0)
            prop_pred = prop_pred[:len(prop_te)]
            results['Prophet'] = metric_row(prop_te['y'].values[:len(prop_pred)], prop_pred)
            all_preds['Prophet'] = (prop_te['ds'].values[:len(prop_pred)],
                                    prop_te['y'].values[:len(prop_pred)], prop_pred)
            st.session_state['prophet_model'] = pm

        # SARIMA
        if use_sarima:
            progress.progress(75, "Training SARIMA (this may take a while)...")
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            ts_tr = train_df.set_index('Date')['Page_Loads']
            ts_te = test_df.set_index('Date')['Page_Loads']
            sm = SARIMAX(ts_tr, order=(1,1,1), seasonal_order=(1,1,1,7),
                         enforce_stationarity=False, enforce_invertibility=False)
            sm_fit = sm.fit(disp=False)
            sm_pred = np.maximum(sm_fit.forecast(steps=len(ts_te)).values, 0)
            results['SARIMA'] = metric_row(ts_te.values, sm_pred)
            all_preds['SARIMA'] = (ts_te.index.values, ts_te.values, sm_pred)

        progress.progress(100, "Done!")
        st.session_state['results']   = results
        st.session_state['all_preds'] = all_preds

        # ── Metric Table ──────────────────────────────────────────────────
        st.markdown("### 📊 Performance Metrics")
        rec = []
        for name, (rmse,mae,mape,r2) in results.items():
            rec.append({'Model':name,'RMSE':f'{rmse:,.0f}','MAE':f'{mae:,.0f}',
                        'MAPE':f'{mape:.2f}%','R²':f'{r2:.4f}'})
        st.dataframe(pd.DataFrame(rec).set_index('Model'), use_container_width=True)

        best = min(results, key=lambda k: results[k][0])
        st.success(f"🏆 Best Model: **{best}** with RMSE = {results[best][0]:,.0f}")

        # ── Bar charts ────────────────────────────────────────────────────
        st.markdown("### 📈 Metric Comparison")
        fig_m = make_subplots(rows=1, cols=4,
                              subplot_titles=['RMSE','MAE','MAPE (%)','R²'])
        for i, metric_idx in enumerate([0,1,2,3]):
            names = list(results.keys())
            vals  = [results[n][metric_idx] for n in names]
            colors= [PALETTE.get(n,'#64748B') for n in names]
            fig_m.add_trace(go.Bar(x=names, y=vals, marker_color=colors, showlegend=False),
                            row=1, col=i+1)
        fig_m.update_layout(height=350, template='plotly_white')
        st.plotly_chart(fig_m, use_container_width=True)

        # ── Forecast overlays ─────────────────────────────────────────────
        st.markdown("### 🔁 Forecast vs Actual")
        for name, (dates, actuals, preds_v) in all_preds.items():
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=dates, y=actuals, name='Actual',
                                       line=dict(color='#1E293B', width=2)))
            fig_f.add_trace(go.Scatter(x=dates, y=preds_v, name=name,
                                       line=dict(color=PALETTE.get(name,'#64748B'),
                                                 width=1.8, dash='dash')))
            fig_f.update_layout(title=f"{name} — Test Set Forecast",
                                 template='plotly_white', height=320,
                                 legend=dict(orientation='h'))
            st.plotly_chart(fig_f, use_container_width=True)

        # ── Residuals ─────────────────────────────────────────────────────
        st.markdown("### 📉 Residual Distributions")
        fig_r = go.Figure()
        for name, (dates, actuals, preds_v) in all_preds.items():
            res = preds_v - actuals
            fig_r.add_trace(go.Violin(y=res, name=name,
                                      fillcolor=PALETTE.get(name,'#64748B'),
                                      line_color='white', opacity=0.75,
                                      box_visible=True, meanline_visible=True))
        fig_r.add_hline(y=0, line_dash='dot', line_color='black')
        fig_r.update_layout(title='Residual Distributions (Violin)',
                             template='plotly_white', height=400)
        st.plotly_chart(fig_r, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<span class="section-tag">FUTURE FORECAST</span>', unsafe_allow_html=True)

    if 'prophet_model' not in st.session_state:
        st.info("Train models first (Tab 2 → Run Analysis with Prophet enabled).")
    else:
        pm   = st.session_state['prophet_model']
        fut  = pm.make_future_dataframe(periods=forecast_days)
        fc   = pm.predict(fut)
        fc30 = fc.tail(forecast_days)[['ds','yhat','yhat_lower','yhat_upper']].copy()
        fc30['yhat'] = np.maximum(fc30['yhat'], 0)

        hist60 = df.tail(90)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist60['Date'], y=hist60['Page_Loads'],
                                    name='Historical', line=dict(color='#1E293B', width=1.5)))
        fig_fc.add_trace(go.Scatter(x=fc30['ds'], y=fc30['yhat'],
                                    name='Forecast', line=dict(color='#15803D', width=2, dash='dash')))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc30['ds'], fc30['ds'].iloc[::-1]]),
            y=pd.concat([fc30['yhat_upper'], fc30['yhat_lower'].iloc[::-1]]),
            fill='toself', fillcolor='rgba(21,128,61,0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='90% CI', showlegend=True))
        fig_fc.add_vline(x=df['Date'].max(), line_dash='dot', line_color='gray')
        fig_fc.update_layout(title=f'{forecast_days}-Day Ahead Forecast (Prophet)',
                              template='plotly_white', height=450,
                              legend=dict(orientation='h'))
        st.plotly_chart(fig_fc, use_container_width=True)

        # KPIs
        c1,c2,c3 = st.columns(3)
        c1.metric("Forecast Avg", f"{fc30['yhat'].mean():,.0f}")
        c2.metric("Forecast Peak", f"{fc30['yhat'].max():,.0f}")
        c3.metric("Forecast Min",  f"{fc30['yhat'].min():,.0f}")

        # Table
        st.markdown("#### Forecast Table")
        fc_display = fc30.rename(columns={'ds':'Date','yhat':'Forecast',
                                          'yhat_lower':'Lower_CI','yhat_upper':'Upper_CI'})
        fc_display[['Forecast','Lower_CI','Upper_CI']] = \
            fc_display[['Forecast','Lower_CI','Upper_CI']].round(0)
        st.dataframe(fc_display.reset_index(drop=True), use_container_width=True, height=350)

        csv_fc = fc_display.to_csv(index=False).encode()
        st.download_button("⬇ Download Forecast CSV", csv_fc,
                           file_name="traffic_forecast.csv", mime='text/csv')

        # Prophet components
        st.markdown("---")
        st.markdown("#### Prophet Trend & Seasonality Components")
        fig_comp = pm.plot_components(fc)
        st.pyplot(fig_comp)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Report
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<span class="section-tag">PROJECT REPORT</span>', unsafe_allow_html=True)

    st.markdown("""
    ## Website Traffic Forecasting
    **First Quadrant Labs | Afeef Anver sha**

    ---

    ### 📌 Objective
    Develop robust multi-model forecasting pipeline to accurately predict daily website
    page loads, enabling data-driven resource allocation and content scheduling.

    ---

    ### 🗂 Dataset
    | Field | Description |
    |---|---|
    | `Page_Loads` | Target variable — total page loads per day |
    | `Unique_Visits` | Distinct visitors per day |
    | `First_Time_Visits` | New visitors |
    | `Returning_Visits` | Returning visitors |

    ---

    ### 🔬 Approach

    **NB1 — EDA & Preprocessing**
    - Parsed dates, stripped commas from numeric fields
    - ADF + KPSS stationarity tests; seasonal decomposition (period=7)
    - ACF/PACF plots for ARIMA order selection
    - Soft outlier clipping (3×IQR)

    **NB3 — Feature Engineering & Training**
    - Calendar features (DOW, month, quarter, Fourier terms)
    - 7 lag features (1,2,3,7,14,21,28 days)
    - Rolling statistics (mean/std/max/min over 7,14,30 days)
    - Visitor ratio features
    - Hyperparameter tuning via RandomizedSearchCV + TimeSeriesSplit

    **NB2 — Evaluation**
    - RMSE, MAE, MAPE, sMAPE, R², Bias
    - Residual analysis, scatter plots, violin distributions
    - 30-day ahead Prophet forecast with confidence intervals

    ---

    ### 🤖 Models
    | Model | Type | Notes |
    |---|---|---|
    | SARIMA(1,1,1)(1,1,1,7) | Statistical | Weekly seasonal ARIMA |
    | Prophet | Bayesian | Trend + weekly + yearly seasonality |
    | Random Forest | ML Ensemble | RandomizedSearchCV tuned |
    | XGBoost | Gradient Boost | 30-iter hyperparameter search |
    | LSTM | Deep Learning | 2-layer, dropout=0.2, lookback=14 |

    ---

    ### 💡 Business Recommendations
    """)

    insights = [
        "📅 Schedule high-traffic content releases on Tuesday–Thursday (weekday peak)",
        "🖥️ Scale server capacity 20–30% higher on Mon–Thu to prevent latency spikes",
        "📣 Run marketing campaigns mid-week to maximize reach and conversion",
        "🔄 Retrain models quarterly as traffic patterns shift seasonally",
        "🚨 Alert if MAPE drifts beyond 15% — indicates structural traffic change",
    ]
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    if 'results' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Model Results (current session)")
        rec = []
        for name, (rmse,mae,mape,r2) in st.session_state['results'].items():
            rec.append({'Model':name,'RMSE':f'{rmse:,.0f}','MAE':f'{mae:,.0f}',
                        'MAPE':f'{mape:.2f}%','R²':f'{r2:.4f}'})
        st.dataframe(pd.DataFrame(rec).set_index('Model'), use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### 📁 Execution Order
    ```
    NB1_EDA_Preprocessing.ipynb      → Cleans data, saves data/processed_traffic.csv
    NB3_Training_FeatureEngineering.ipynb → Trains & saves models/, predictions/
    NB2_Model_Evaluation.ipynb       → Loads predictions, compares models
    streamlit run app.py             → Interactive dashboard
    ```
    *Submission deadline: 10 April 2026 | projects@firstquadrantlabs.com*
    """)