from __future__ import annotations

import time
from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import requests

from iot2sdg.config import CONFIG

BASE_URL = "https://api.openaq.org/v2/measurements"


def _fetch_city_pm25(city: str, date_from: str, date_to: str, limit: int = 10000) -> pd.DataFrame:
	params = {
		"city": city,
		"parameter": CONFIG.parameter,
		"date_from": date_from,
		"date_to": date_to,
		"limit": limit,
		"page": 1,
		"sort": "asc",
		"order_by": "date",
	}
	frames: List[pd.DataFrame] = []
	while True:
		resp = requests.get(BASE_URL, params=params, timeout=30)
		# If the API is unavailable or returns non-2xx, surface to caller to decide fallback
		resp.raise_for_status()
		js = resp.json()
		results = js.get("results", [])
		if not results:
			break
		df = pd.json_normalize(results)
		frames.append(df)
		meta = js.get("meta", {})
		found = meta.get("found", 0)
		page = params["page"]
		pages = max(1, (found + limit - 1) // limit)
		if page >= pages:
			break
		params["page"] = page + 1
		time.sleep(0.2)
	if not frames:
		return pd.DataFrame(columns=["city", "location", "value", "unit", "date.utc"])  # minimal
	out = pd.concat(frames, ignore_index=True)
	return out


def _simulate_pm25_city(city: str, start: datetime, end: datetime) -> pd.DataFrame:
	# Generate a simple monthly PM2.5 series per city
	idx = pd.date_range(start, end, freq="MS", tz="UTC")
	if len(idx) == 0:
		return pd.DataFrame(columns=["timestamp", "city", "location", "pm25", "unit"])
	baseline = np.random.uniform(20, 70)
	seasonal = 10 * np.sin(2 * np.pi * (idx.month / 12.0))
	noise = np.random.normal(0, 5, len(idx))
	values = np.maximum(0.0, baseline + seasonal + noise)
	df = pd.DataFrame({
		"timestamp": idx,
		"city": city,
		"location": "Simulated Sensor",
		"pm25": values,
		"unit": "µg/m³",
	})
	return df


def fetch_pm25_for_cities(cities: List[str], start: datetime, end: datetime, cache: bool = True) -> pd.DataFrame:
	date_from = start.strftime("%Y-%m-%dT00:00:00Z")
	date_to = end.strftime("%Y-%m-%dT23:59:59Z")
	all_frames: List[pd.DataFrame] = []
	for city in cities:
		try:
			df = _fetch_city_pm25(city, date_from, date_to)
			if df.empty:
				# Fall back to simulation if no data returned
				sim = _simulate_pm25_city(city, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
				all_frames.append(sim)
				continue
			df["city"] = city
			df = df.rename(columns={"date.utc": "timestamp", "value": "pm25"})
			df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
			df = df[["timestamp", "city", "location", "pm25", "unit"]]
			all_frames.append(df)
		except requests.RequestException:
			# Network/API error → simulate for this city
			sim = _simulate_pm25_city(city, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
			all_frames.append(sim)
	if not all_frames:
		return pd.DataFrame()
	data = pd.concat(all_frames, ignore_index=True)
	if cache:
		path = CONFIG.raw_dir / "openaq_pm25.parquet"
		data.to_parquet(path, index=False)
	return data
