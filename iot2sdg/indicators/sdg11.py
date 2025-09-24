from __future__ import annotations

import pandas as pd


def aggregate_pm25_monthly(pm25: pd.DataFrame) -> pd.DataFrame:
	pm25_monthly = pm25.copy()
	pm25_monthly["timestamp"] = pd.to_datetime(pm25_monthly["timestamp"], utc=True)
	pm25_monthly = pm25_monthly.set_index("timestamp").groupby("city")["pm25"].resample("MS").mean().reset_index()
	return pm25_monthly


def aggregate_traffic_monthly(traffic: pd.DataFrame) -> pd.DataFrame:
	traffic_monthly = traffic.copy()
	traffic_monthly["timestamp"] = pd.to_datetime(traffic_monthly["timestamp"], utc=True)
	traffic_monthly = traffic_monthly.set_index("timestamp").groupby("city")["traffic_index"].resample("MS").mean().reset_index()
	return traffic_monthly


def combine_indicators(pm25_monthly: pd.DataFrame, traffic_monthly: pd.DataFrame) -> pd.DataFrame:
	merged = pd.merge(pm25_monthly, traffic_monthly, on=["timestamp", "city"], how="outer")
	return merged
