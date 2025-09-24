import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from iot2sdg.config import CONFIG
from iot2sdg.data.ingest_openaq import fetch_pm25_for_cities
from iot2sdg.data.traffic import load_or_simulate_monthly_traffic
from iot2sdg.processing.clean import resample_and_fill
from iot2sdg.indicators.sdg11 import aggregate_pm25_monthly, aggregate_traffic_monthly, combine_indicators
from iot2sdg.causal.did import prepare_panel, run_did
from iot2sdg.forecasting.models import arima_forecast_monthly
from iot2sdg.anomaly.detect import detect_anomalies

st.set_page_config(page_title="IoT2SDG11 Dashboard", layout="wide")
st.title("IoT2SDG11 — SDG-11 Indicators and Analytics")
st.caption("Explore air quality (PM2.5), transport activity, causal effects, forecasts, and anomalies.")

with st.sidebar:
	st.header("Settings")
	all_cities = CONFIG.all_cities()
	default_controls = all_cities[1:]
	col_a, col_b = st.columns(2)
	with col_a:
		treated = st.selectbox("Treated city", options=all_cities, index=0)
	with col_b:
		intervention = st.date_input("Intervention date", value=CONFIG.intervention_date)
	controls = st.multiselect("Control cities", options=[c for c in all_cities if c != treated], default=[c for c in default_controls if c != treated])
	date_range = st.date_input("Date range", value=(CONFIG.start_date, CONFIG.end_date))
	refresh = st.button("Fetch OpenAQ & Recompute")

@st.cache_data(show_spinner=False)
def get_pm25(treated_city: str, control_cities: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	return fetch_pm25_for_cities([treated_city] + control_cities, pd.Timestamp(start_date), pd.Timestamp(end_date))

if refresh:
	st.cache_data.clear()

start_sel = pd.Timestamp(date_range[0]) if isinstance(date_range, (list, tuple)) else pd.Timestamp(CONFIG.start_date)
end_sel = pd.Timestamp(date_range[1]) if isinstance(date_range, (list, tuple)) else pd.Timestamp(CONFIG.end_date)
pm25_raw = get_pm25(treated, controls, start_sel, end_sel)
traffic = load_or_simulate_monthly_traffic()

pm25_monthly = resample_and_fill(pm25_raw, value_col="pm25")
pm25_monthly = pm25_monthly.rename(columns={"pm25": "pm25_mean"})
traffic_monthly = aggregate_traffic_monthly(traffic)
indicators = combine_indicators(pm25_monthly.rename(columns={"pm25_mean": "pm25"}), traffic_monthly)

# KPI row
if not pm25_monthly.empty:
	latest = pm25_monthly.sort_values("timestamp").groupby("city").tail(1)
	avg_pm25 = float(pm25_monthly["pm25_mean"].mean()) if not pm25_monthly.empty else float("nan")
	first_last = pm25_monthly.sort_values("timestamp").groupby("city")["pm25_mean"].agg(lambda s: (s.iloc[0], s.iloc[-1]))
	trend = np.nan
	if treated in first_last.index:
		start_val, end_val = first_last.loc[treated]
		trend = (end_val - start_val)
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Avg PM2.5 (µg/m³)", f"{avg_pm25:.1f}")
	with col2:
		st.metric("Cities", f"{pm25_monthly['city'].nunique()}")
	with col3:
		st.metric(f"Trend ({treated})", f"{trend:+.1f}")

tab_overview, tab_ind, tab_did, tab_fc, tab_anom, tab_data = st.tabs([
	"Overview", "Indicators", "Causal (DiD)", "Forecasts", "Anomalies", "Data"
])

with tab_overview:
	st.subheader("PM2.5 — Monthly Averages")
	if not pm25_monthly.empty:
		fig = px.line(pm25_monthly, x="timestamp", y="pm25_mean", color="city", markers=True)
		fig.update_layout(height=420, legend_title_text="City")
		st.plotly_chart(fig, use_container_width=True)
	st.subheader("Traffic Index — Monthly")
	if not traffic_monthly.empty:
		fig2 = px.line(traffic_monthly, x="timestamp", y="traffic_index", color="city", markers=True)
		fig2.update_layout(height=420, legend_title_text="City")
		st.plotly_chart(fig2, use_container_width=True)

with tab_ind:
	st.subheader("SDG-11 Indicators (Merged)")
	st.dataframe(indicators.tail(200))
	if not indicators.empty:
		fig3 = px.scatter(indicators, x="traffic_index", y="pm25", color="city")
		fig3.update_layout(height=420)
		st.plotly_chart(fig3, use_container_width=True)

with tab_did:
	st.subheader("Difference-in-Differences (PM2.5)")
	panel = prepare_panel(pm25_monthly.rename(columns={"pm25_mean": "pm25"}), "pm25", treated, intervention)
	did_res = run_did(panel, outcome_col="pm25")
	col_a, col_b, col_c = st.columns(3)
	col_a.metric("DiD Effect", f"{did_res.effect:.2f}")
	col_b.metric("Std. Error", f"{did_res.se:.2f}")
	col_c.metric("p-value", f"{did_res.p_value:.3f}")
	with st.expander("Regression summary"):
		st.text(did_res.summary)

with tab_fc:
	st.subheader("Forecasts (PM2.5)")
	fc = arima_forecast_monthly(pm25_monthly.rename(columns={"pm25_mean": "pm25"}), value_col="pm25", horizon=6)
	st.dataframe(fc.head(100))
	if not fc.empty:
		figf = px.line(fc, x="timestamp", y="yhat", color="city")
		figf.update_layout(height=420)
		st.plotly_chart(figf, use_container_width=True)

with tab_anom:
	st.subheader("Anomalies (PM2.5)")
	pm25_anom = detect_anomalies(pm25_monthly.rename(columns={"pm25_mean": "pm25"}), value_col="pm25")
	st.dataframe(pm25_anom[pm25_anom["is_anomaly"]].head(200))
	if not pm25_anom.empty:
		plot_df = pm25_anom.rename(columns={"pm25_mean": "pm25"}) if "pm25_mean" in pm25_anom.columns else pm25_anom
		figa = px.scatter(plot_df, x="timestamp", y="pm25", color="city", symbol="is_anomaly", symbol_map={True: "x", False: "circle"})
		figa.update_layout(height=420)
		st.plotly_chart(figa, use_container_width=True)

with tab_data:
	st.subheader("Raw and Processed Data")
	colx, coly = st.columns(2)
	with colx:
		st.write("PM2.5 (raw)")
		st.dataframe(pm25_raw.head(100))
	with coly:
		st.write("PM2.5 (monthly)")
		st.dataframe(pm25_monthly.head(100))
