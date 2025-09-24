from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, value_col: str, contamination: float = 0.02, random_state: int = 42) -> pd.DataFrame:
	if df.empty:
		return df.assign(is_anomaly=False)
	out_frames = []
	for city, g in df.groupby("city"):
		values = g[[value_col]].fillna(method="ffill").fillna(method="bfill").values
		clf = IsolationForest(contamination=contamination, random_state=random_state)
		labels = clf.fit_predict(values)
		flag = labels == -1
		gg = g.copy()
		gg["is_anomaly"] = flag
		out_frames.append(gg)
	return pd.concat(out_frames, ignore_index=True)
