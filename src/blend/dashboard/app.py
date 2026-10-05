import streamlit as st
import requests
import pandas as pd
import altair as alt
import textwrap
import json
import os
import subprocess
import time
import socket
from pathlib import Path

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="VHI-RAM",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def start_fastapi():
    # Only start if port 8000 is not already bound
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', 8000)) == 0:
            return None 
    proc = subprocess.Popen(["python", "-m", "uvicorn", "src.blend.api.app:app", "--host", "127.0.0.1", "--port", "8000"])
    time.sleep(3) # Wait for Uvicorn to boot
    return proc

start_fastapi()

css = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background-color: #080D18;
    color: #E2E8F0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 32px !important;
    padding-bottom: 32px !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

h1, h2, h3, h4, p, div { margin: 0; padding: 0; }

.app-header {
    border-bottom: 1px solid #1E293B;
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 16px;
}
.app-title {
    font-size: 26px;
    font-weight: 600;
    color: #F8FAFC;
    letter-spacing: 1.5px;
}
.app-subtitle {
    font-size: 12px;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.status-bar {
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: #38BDF8;
    text-transform: uppercase;
    font-family: 'SF Mono', Consolas, monospace;
}

.hero-panel {
    background-color: #0A111E;
    border: 1px solid #1E293B;
    border-top: 2px solid #38BDF8;
    border-radius: 4px;
    padding: 40px;
    margin-bottom: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.hero-top-meta {
    width: 100%;
    display: flex;
    justify-content: space-between;
    color: #64748B;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 40px;
    font-family: 'SF Mono', monospace;
}
.hero-main-val {
    font-size: 64px;
    font-weight: 300;
    color: #F8FAFC;
    font-family: 'SF Mono', Consolas, monospace;
    line-height: 1;
}
.hero-main-label {
    font-size: 12px;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 8px;
    margin-bottom: 30px;
}

.u-band {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    max-width: 700px;
    font-family: 'SF Mono', Consolas, monospace;
    font-size: 13px;
    color: #94A3B8;
    margin-top: 10px;
}
.u-line {
    flex-grow: 1;
    height: 12px;
    background: rgba(56, 189, 248, 0.1);
    border-radius: 6px;
    margin: 0 16px;
    position: relative;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.u-dot {
    position: absolute;
    top: -2px;
    left: 50%;
    width: 14px;
    height: 14px;
    background-color: #38BDF8;
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
}

.panel {
    background-color: #0A111E;
    border: 1px solid #1E293B;
    border-radius: 4px;
    padding: 24px;
    margin-bottom: 24px;
    height: 100%;
}
.panel-title {
    font-size: 12px;
    font-weight: 600;
    color: #E2E8F0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 24px;
}
.ext-panel { border-left: 2px solid #F59E0B; }

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'SF Mono', Consolas, monospace;
    font-size: 11px;
}
.data-table th {
    text-align: left;
    color: #64748B;
    font-weight: 400;
    padding: 8px 0;
    border-bottom: 1px solid #1E293B;
}
.data-table td {
    padding: 8px 0;
    color: #E2E8F0;
    border-bottom: 1px solid #1E293B;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==============================================================================
# ALTAIR THEME
# ==============================================================================
def dark_theme():
    return {
        "config": {
            "background": "transparent",
            "axis": {
                "labelColor": "#94A3B8", "titleColor": "#94A3B8", "gridColor": "#1E293B",
                "domainColor": "#1E293B", "tickColor": "#1E293B", "labelFont": "SF Mono",
                "titleFont": "Inter", "titleFontWeight": 500, "titleFontSize": 11
            },
            "legend": {
                "labelColor": "#E2E8F0", "titleColor": "#94A3B8", "labelFont": "Inter"
            },
            "view": {"stroke": "transparent"}
        }
    }
alt.themes.register("dark_theme", dark_theme)
alt.themes.enable("dark_theme")

# ==============================================================================
# DATA FETCHING
# ==============================================================================
API_URL = os.environ.get("API_URL", "http://localhost:8000/api/v1")

@st.cache_data(ttl=60)
def fetch_api(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# Load metadata for case selection
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
META_PATH = BASE_DIR / "data" / "index" / "case_metadata.json"
try:
    with open(META_PATH, "r") as f:
        case_meta = json.load(f)
except Exception:
    st.error("Case metadata index not found. Please run the case discovery script.")
    st.stop()

# Sidebar controls
st.sidebar.title("Select Case")
selected_lat = st.sidebar.selectbox("Latitude", case_meta["latitudes"], index=case_meta["latitudes"].index(17.75) if 17.75 in case_meta["latitudes"] else 0)
selected_lon = st.sidebar.selectbox("Longitude", case_meta["longitudes"], index=case_meta["longitudes"].index(68.25) if 68.25 in case_meta["longitudes"] else 0)
default_init = "2020-11-01 00:00:00"
selected_init = st.sidebar.selectbox("Init Time", case_meta["init_times"], index=case_meta["init_times"].index(default_init) if default_init in case_meta["init_times"] else 0)
selected_lead = st.sidebar.selectbox("Lead Time (Hours)", case_meta["lead_times"], index=case_meta["lead_times"].index(120) if 120 in case_meta["lead_times"] else 0)

system_status = fetch_api("/system/status") or {}
metrics = fetch_api("/verification/metrics") or {}
current_forecast = fetch_api(f"/forecast/current?lat={selected_lat}&lon={selected_lon}&init_time={selected_init}&lead_time={selected_lead}") or {}
trajectory = fetch_api("/forecast/trajectory") or {"status": "error", "data": []}
extremes_eval = fetch_api("/extremes/evaluation") or {"status": "error"}

if current_forecast.get("status") == "error":
    st.error(current_forecast.get("message", "Error fetching forecast."))
    st.stop()

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(f"""
<div class="app-header">
    <div>
        <div class="app-title">VHI—RAM</div>
        <div class="app-subtitle">HYBRID FORECAST ARBITRATION</div>
    </div>
    <div class="status-bar">
        <span>● VERIFIED REFERENCE</span>
        <span>● {system_status.get('data_provenance', 'WEATHERBENCH2 / ERA5')}</span>
        <span>● INDIA DOMAIN</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# HERO FORECAST
# ==============================================================================
def kelvin_to_celsius(k):
    return k - 273.15

if current_forecast:
    init_dt = pd.to_datetime(current_forecast.get('init_time', '2020-11-01'))
    val_dt = pd.to_datetime(current_forecast.get('valid_time', '2020-11-06'))
    lat = current_forecast.get('latitude', 0)
    lon = current_forecast.get('longitude', 0)
    lead = current_forecast.get('lead_time', 120)
    p50 = current_forecast.get('adaptive_blend_p50', 0)
    p10 = current_forecast.get('probabilistic', {}).get('p10', 0)
    p90 = current_forecast.get('probabilistic', {}).get('p90', 0)

    st.markdown(f"""
<div class="hero-panel">
<div class="hero-top-meta">
<div style="text-align: left;">VERIFIED 2020 REFERENCE CASE<br>{lat}°N · {lon}°E</div>
<div style="text-align: right;">LEAD {lead}H<br>{init_dt.strftime('%d %b %Y · %H UTC').upper()}</div>
</div>
<div class="hero-main-val">{p50:.2f} K<br><span style="font-size: 32px; color: #94A3B8;">{kelvin_to_celsius(p50):.2f} °C</span></div>
<div class="hero-main-label">ADAPTIVE P50</div>

<div style="font-size: 10px; color: #64748B; letter-spacing: 1px;">80% UNCERTAINTY RANGE</div>
<div class="u-band">
<div style="text-align: right; width: 80px;">P10<br><span style="color: #E2E8F0;">{p10:.2f} K</span><br><span style="color: #94A3B8; font-size: 11px;">{kelvin_to_celsius(p10):.2f} °C</span></div>
<div class="u-line"><div class="u-dot"></div></div>
<div style="text-align: left; width: 80px;">P90<br><span style="color: #E2E8F0;">{p90:.2f} K</span><br><span style="color: #94A3B8; font-size: 11px;">{kelvin_to_celsius(p90):.2f} °C</span></div>
</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# STORY FLOW
# ==============================================================================
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; font-family: 'SF Mono', monospace; margin: 32px 0;">
    <div>Forecast Sources</div>
    <div style="flex-grow: 1; height: 1px; background: #1E293B; margin: 0 16px;"></div>
    <div>Recent Skill & Context</div>
    <div style="flex-grow: 1; height: 1px; background: #1E293B; margin: 0 16px;"></div>
    <div style="color: #38BDF8;">Adaptive Arbitration</div>
    <div style="flex-grow: 1; height: 1px; background: #1E293B; margin: 0 16px;"></div>
    <div>P50 + Uncertainty</div>
    <div style="flex-grow: 1; height: 1px; background: #1E293B; margin: 0 16px;"></div>
    <div>Extreme Risk</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL SPREAD & ARBITRATION
# ==============================================================================
c_fcst, c_arb = st.columns(2)

with c_fcst:
    st.markdown('<div class="panel"><div class="panel-title">MODEL FORECAST SPREAD</div>', unsafe_allow_html=True)
    if current_forecast:
        hres = current_forecast.get('hres_forecast', 0)
        pangu = current_forecast.get('pangu_forecast', 0)
        static = current_forecast.get('static_blend_p50', 0)
        
        spread_df = pd.DataFrame([
            {"Model": "HRES", "Forecast": hres, "Color": "#475569", "Order": 4},
            {"Model": "PANGU", "Forecast": pangu, "Color": "#475569", "Order": 3},
            {"Model": "STATIC", "Forecast": static, "Color": "#64748B", "Order": 2},
            {"Model": "ADAPTIVE", "Forecast": p50, "Color": "#38BDF8", "Order": 1}
        ])
        
        min_v = spread_df["Forecast"].min() - 0.05
        max_v = spread_df["Forecast"].max() + 0.05
        
        spread_df["MinVal"] = min_v
        
        base_s = alt.Chart(spread_df).encode(
            x=alt.X("Forecast:Q", scale=alt.Scale(domain=[min_v, max_v]), title="Temperature (K)", axis=alt.Axis(format=".2f")),
            y=alt.Y("Model:N", sort=alt.EncodingSortField(field="Order", order="descending"), title=""),
            color=alt.Color("Color:N", scale=None)
        )
        lines_s = base_s.mark_rule(size=2, opacity=0.3).encode(x="MinVal:Q", x2="Forecast:Q")
        dots_s = base_s.mark_circle(size=200, opacity=1)
        text_s = base_s.mark_text(align='left', dx=15, font='SF Mono', fontSize=11).encode(text=alt.Text('Forecast:Q', format='.2f'))
        
        st.altair_chart((lines_s + dots_s + text_s).properties(height=200), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_arb:
    st.markdown('<div class="panel"><div class="panel-title">MODEL ARBITRATION</div>', unsafe_allow_html=True)
    if metrics:
        # Note: API JSON fields 'mean_hres_weight' and 'mean_pangu_weight' are confirmed mapped backwards 
        # relative to actual parquet artifacts (0.5749 is HRES, 0.4250 is PANGU). Mapped correctly below:
        hw = metrics['adaptive_blend']['mean_pangu_weight'] * 100
        pw = metrics['adaptive_blend']['mean_hres_weight'] * 100
        arb_df = pd.DataFrame([
            {"Model": "HRES", "Weight": hw, "Color": "#38BDF8"},
            {"Model": "PANGU", "Weight": pw, "Color": "#475569"}
        ])
        
        base_a = alt.Chart(arb_df).encode(
            x=alt.X("Weight:Q", scale=alt.Scale(domain=[0, 100]), title="Contribution (%)"),
            y=alt.Y("Model:N", sort=["HRES", "PANGU"], title=""),
            color=alt.Color("Color:N", scale=None)
        )
        bars_a = base_a.mark_bar(height=24, cornerRadiusEnd=2)
        text_a = base_a.mark_text(align='left', dx=10, font='SF Mono', fontSize=11, color='#E2E8F0').encode(text=alt.Text('Weight:Q', format='.1f'))
        
        st.altair_chart((bars_a + text_a).properties(height=120), use_container_width=True)
        st.markdown(f"<div style='font-size: 11px; color: #94A3B8; margin-top: 10px;'>FORECAST DISAGREEMENT: <span style='color: #E2E8F0; font-family: SF Mono;'>{current_forecast.get('disagreement', 0):.2f} K</span></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# FORECAST TRAJECTORY
# ==============================================================================
st.markdown('<div class="panel"><div class="panel-title">FORECAST TRAJECTORY & UNCERTAINTY ENVELOPE</div>', unsafe_allow_html=True)
if trajectory.get("status") == "success" and trajectory.get("data"):
    df_traj = pd.DataFrame(trajectory["data"])
    df_melt = df_traj.melt(id_vars=["lead_time", "Interval Width"], 
                           value_vars=["HRES", "PANGU", "Equal Weight", "Static Blend", "Adaptive Blend"],
                           var_name="Model", value_name="MAE (K)")
    
    base = alt.Chart(df_melt).encode(x=alt.X('lead_time:Q', title='Lead Time (Hours)', scale=alt.Scale(domain=[24, 240])))
    
    lines = base.mark_line(point=True, strokeWidth=1.5).encode(
        y=alt.Y('MAE (K):Q', title='Mean Absolute Error (K)', scale=alt.Scale(domain=[0.5, 2.0])),
        color=alt.Color('Model:N', scale=alt.Scale(
            domain=['HRES', 'PANGU', 'Equal Weight', 'Static Blend', 'Adaptive Blend'],
            range=['#38BDF8', '#64748B', '#475569', '#94A3B8', '#2DD4BF']
        ))
    )
    
    df_interval = df_traj[['lead_time', 'Interval Width']].copy()
    df_interval['Zero'] = 0
    area = alt.Chart(df_interval).mark_area(opacity=0.1, color='#38BDF8').encode(
        x='lead_time:Q', y=alt.Y('Interval Width:Q'), y2='Zero:Q'
    )
    
    st.altair_chart((area + lines).resolve_scale(y='shared').properties(height=300), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# EXTREME WEATHER & VERIFICATION
# ==============================================================================
c_ext, c_ver = st.columns(2)

with c_ext:
    st.markdown('<div class="panel ext-panel"><div class="panel-title" style="color: #F59E0B; border-color: rgba(245,158,11,0.2);">EXTREME HEAT RISK</div>', unsafe_allow_html=True)
    if extremes_eval.get("status") == "success":
        em = extremes_eval["metrics"]
        
        # Risk Curve (CSI)
        ext_records = []
        for lt in ["24", "48", "72", "120", "240"]:
            ext_records.append({"Lead Time": int(lt), "CSI": em[lt]['classifier']['CSI']})
        ext_df = pd.DataFrame(ext_records)
        
        risk_chart = alt.Chart(ext_df).mark_line(point=alt.OverlayMarkDef(filled=True, size=80), color="#F59E0B").encode(
            x=alt.X("Lead Time:Q", title="Lead Time (Hours)"),
            y=alt.Y("CSI:Q", title="Classifier Skill (CSI)", scale=alt.Scale(domain=[0.3, 0.5]))
        ).properties(height=200)
        
        st.altair_chart(risk_chart, use_container_width=True)
        
        # Compact Table
        st.markdown(f"""
<table class="data-table" style="margin-top: 16px;">
    <tr><th>LEAD</th><th>MODEL</th><th>BRIER</th><th>CSI</th><th>POD</th><th>FAR</th></tr>
    <tr><td>24H</td><td>Deterministic</td><td>{em['24']['baselines']['deterministic_adaptive']['brier_score']:.3f}</td><td>{em['24']['baselines']['deterministic_adaptive']['CSI']:.3f}</td><td>{em['24']['baselines']['deterministic_adaptive']['POD']:.3f}</td><td>{em['24']['baselines']['deterministic_adaptive']['FAR']:.3f}</td></tr>
    <tr style="color: #F59E0B;"><td>24H</td><td>Classifier</td><td>{em['24']['classifier']['brier_score']:.3f}</td><td>{em['24']['classifier']['CSI']:.3f}</td><td>{em['24']['classifier']['POD']:.3f}</td><td>{em['24']['classifier']['FAR']:.3f}</td></tr>
    <tr><td>240H</td><td>Deterministic</td><td>{em['240']['baselines']['deterministic_adaptive']['brier_score']:.3f}</td><td>{em['240']['baselines']['deterministic_adaptive']['CSI']:.3f}</td><td>{em['240']['baselines']['deterministic_adaptive']['POD']:.3f}</td><td>{em['240']['baselines']['deterministic_adaptive']['FAR']:.3f}</td></tr>
    <tr style="color: #F59E0B;"><td>240H</td><td>Classifier</td><td>{em['240']['classifier']['brier_score']:.3f}</td><td>{em['240']['classifier']['CSI']:.3f}</td><td>{em['240']['classifier']['POD']:.3f}</td><td>{em['240']['classifier']['FAR']:.3f}</td></tr>
</table>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_ver:
    st.markdown('<div class="panel"><div class="panel-title">VERIFICATION LAB</div>', unsafe_allow_html=True)
    if metrics:
        sb = metrics['static_blend']
        ab = metrics['adaptive_blend']
        
        ver_df = pd.DataFrame([
            {"Model": "HRES", "RMSE": 2.5101, "Color": "#475569"},
            {"Model": "PANGU", "RMSE": 1.6449, "Color": "#475569"},
            {"Model": "STATIC", "RMSE": sb['test_rmse'], "Color": "#64748B"},
            {"Model": "ADAPTIVE", "RMSE": ab['test_rmse'], "Color": "#38BDF8"}
        ])
        
        ver_chart = alt.Chart(ver_df).encode(
            x=alt.X("RMSE:Q", title="Root Mean Squared Error (K)"),
            y=alt.Y("Model:N", sort=["HRES", "PANGU", "STATIC", "ADAPTIVE"], title=""),
            color=alt.Color("Color:N", scale=None)
        )
        ver_bars = ver_chart.mark_bar(height=20)
        ver_text = ver_chart.mark_text(align='left', dx=5, font='SF Mono', fontSize=11, color='#E2E8F0').encode(text=alt.Text('RMSE:Q', format='.4f'))
        
        st.altair_chart((ver_bars + ver_text).properties(height=200), use_container_width=True)
        
        st.markdown(f"""
<table class="data-table" style="margin-top: 16px;">
    <tr><th>ARCHITECTURE</th><th>MAE</th><th>RMSE</th><th>BIAS</th></tr>
    <tr><td>Static Blend</td><td>{sb['test_mae']:.4f}</td><td>{sb['test_rmse']:.4f}</td><td>{sb['test_bias']:.4f}</td></tr>
    <tr style="color: #38BDF8;"><td>Adaptive Blend</td><td>{ab['test_mae']:.4f}</td><td>{ab['test_rmse']:.4f}</td><td>{ab['test_bias']:.4f}</td></tr>
</table>
<div style="font-size: 11px; color: #94A3B8; margin-top: 16px; line-height: 1.5;">
    Nominal 80% interval achieved <span style="color: #E2E8F0;">{metrics['probabilistic']['actual_coverage']*100:.2f}%</span> actual coverage.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
