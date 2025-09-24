from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast_monthly(df: pd.DataFrame, value_col: str, horizon: int = 6) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["timestamp", "city", value_col, "yhat", "yhat_lower", "yhat_upper"])
	pred_frames = []
	for city, g in df.groupby("city"):
		series = g.set_index("timestamp")[value_col].asfreq("MS")
		series = series.fillna(method="ffill").fillna(method="bfill")
		model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
		res = model.fit(disp=False)
		forecast_res = res.get_forecast(steps=horizon)
		pred_mean = forecast_res.predicted_mean
		conf_int = forecast_res.conf_int(alpha=0.05)
		pred_df = pd.DataFrame({
			"timestamp": pred_mean.index,
			"city": city,
			"yhat": pred_mean.values,
			"yhat_lower": conf_int.iloc[:, 0].values,
			"yhat_upper": conf_int.iloc[:, 1].values,
		})
		pred_frames.append(pred_df)
	return pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
