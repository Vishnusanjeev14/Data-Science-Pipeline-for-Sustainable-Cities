from __future__ import annotations

import pandas as pd

from iot2sdg.config import CONFIG
from iot2sdg.data.ingest_openaq import fetch_pm25_for_cities
from iot2sdg.data.traffic import load_or_simulate_monthly_traffic
from iot2sdg.processing.clean import resample_and_fill, save_monthly
from iot2sdg.indicators.sdg11 import aggregate_traffic_monthly, combine_indicators
from iot2sdg.causal.did import prepare_panel, run_did
from iot2sdg.forecasting.models import arima_forecast_monthly
from iot2sdg.anomaly.detect import detect_anomalies


def main() -> None:
	cities = CONFIG.all_cities()
	pm25_raw = fetch_pm25_for_cities(cities, pd.Timestamp(CONFIG.start_date), pd.Timestamp(CONFIG.end_date))
	traffic = load_or_simulate_monthly_traffic()

	pm25_monthly = resample_and_fill(pm25_raw, value_col="pm25")
	save_monthly(pm25_monthly, "pm25_monthly")

	traffic_monthly = aggregate_traffic_monthly(traffic)
	save_monthly(traffic_monthly, "traffic_monthly")

	indicators = combine_indicators(pm25_monthly.rename(columns={"pm25": "pm25"}), traffic_monthly)
	indicators.to_parquet(CONFIG.processed_dir / "sdg11_indicators.parquet", index=False)

	panel = prepare_panel(pm25_monthly.rename(columns={"pm25": "pm25"}), "pm25", CONFIG.treated_city, CONFIG.intervention_date)
	did = run_did(panel, outcome_col="pm25")
	with open(CONFIG.tables_dir / "did_summary.txt", "w", encoding="utf-8") as f:
		f.write(did.summary)

	fc = arima_forecast_monthly(pm25_monthly.rename(columns={"pm25": "pm25"}), value_col="pm25", horizon=6)
	fc.to_parquet(CONFIG.outputs_dir / "forecast_pm25.parquet", index=False)

	anom = detect_anomalies(pm25_monthly.rename(columns={"pm25": "pm25"}), value_col="pm25")
	anom.to_parquet(CONFIG.outputs_dir / "anomalies_pm25.parquet", index=False)

	print("Pipeline completed.")


if __name__ == "__main__":
	main()
