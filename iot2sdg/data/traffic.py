from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from iot2sdg.config import CONFIG


def load_or_simulate_monthly_traffic(csv_path: Optional[Path] = None) -> pd.DataFrame:
	if csv_path and Path(csv_path).exists():
		df = pd.read_csv(csv_path, parse_dates=["timestamp"])
		return df
	# Simulate monthly traffic ridership index per city
	idx = pd.date_range(CONFIG.start_date, CONFIG.end_date, freq="MS", tz="UTC")
	records = []
	for city in CONFIG.all_cities():
		baseline = np.random.uniform(60, 120)
		trend = np.linspace(0, 10, len(idx))
		noise = np.random.normal(0, 5, len(idx))
		series = baseline + trend + noise
		for t, v in zip(idx, series):
			records.append({"timestamp": t, "city": city, "traffic_index": float(max(0.0, v))})
	df = pd.DataFrame(records)
	path = CONFIG.raw_dir / "traffic_monthly.parquet"
	df.to_parquet(path, index=False)
	return df
