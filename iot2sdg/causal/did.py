from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class DIDResult:
	effect: float
	se: float
	p_value: float
	summary: str


def prepare_panel(df: pd.DataFrame, outcome_col: str, treated_city: str, intervention_date: datetime) -> pd.DataFrame:
	panel = df.copy()
	panel["post"] = (pd.to_datetime(panel["timestamp"], utc=True) >= pd.Timestamp(intervention_date, tz="UTC")).astype(int)
	panel["treated"] = (panel["city"] == treated_city).astype(int)
	panel["did"] = panel["post"] * panel["treated"]
	return panel


def run_did(panel: pd.DataFrame, outcome_col: str) -> DIDResult:
	# Two-way FE via OLS with city and month fixed effects
	panel = panel.copy()
	panel["month"] = pd.to_datetime(panel["timestamp"], utc=True).dt.to_period("M").astype(str)
	model = smf.ols(formula=f"{outcome_col} ~ did + C(city) + C(month)", data=panel, missing="drop").fit(cov_type="HC1")
	effect = model.params.get("did", float("nan"))
	se = model.bse.get("did", float("nan"))
	pval = model.pvalues.get("did", float("nan"))
	return DIDResult(effect=effect, se=se, p_value=pval, summary=str(model.summary()))
