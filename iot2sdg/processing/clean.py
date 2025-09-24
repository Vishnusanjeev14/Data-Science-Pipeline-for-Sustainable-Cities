from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from iot2sdg.config import CONFIG


def resample_and_fill(df: pd.DataFrame, value_col: str, freq: str = "MS", group_col: str = "city", method: Literal["linear", "time", "nearest"] = "time", limit: Optional[int] = None) -> pd.DataFrame:
	if df.empty:
		return df
	df = df.copy()
	df = df.dropna(subset=[value_col])
	df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
	df = df.set_index("timestamp")
	out = []
	for key, g in df.groupby(group_col):
		monthly = g[value_col].resample(freq).mean()
		filled = monthly.interpolate(method=method, limit=limit)
		out.append(pd.DataFrame({group_col: key, value_col: filled}))
	result = pd.concat(out).reset_index(names=["timestamp"])  # timestamp is the resample index
	return result


def save_monthly(df: pd.DataFrame, name: str) -> None:
	path = CONFIG.processed_dir / f"{name}.parquet"
	df.to_parquet(path, index=False)
