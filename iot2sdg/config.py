from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Dict


@dataclass(frozen=True)
class ProjectConfig:
    # Cities
    treated_city: str = "Delhi"
    control_cities: List[str] = field(default_factory=lambda: ["Mumbai", "Bengaluru", "Kolkata", "Chennai"])

    # Parameters
    parameter: str = "pm25"

    # Dates
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2024, 12, 31)
    intervention_date: date = date(2023, 7, 1)

    # Optional: allow mapping of city → intervention date
    city_dates: Dict[str, date] = field(default_factory=lambda: {
        "Delhi": date(2023, 7, 1),
        "Mumbai": date(2023, 8, 15),
        "Bengaluru": date(2023, 9, 1),
        "Kolkata": date(2023, 10, 5),
        "Chennai": date(2023, 11, 20),
    })

    # Directories
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    figures_dir: Path = Path("outputs/figures")
    tables_dir: Path = Path("outputs/tables")
    src_dir: Path = Path("src")

    def all_cities(self) -> List[str]:
        return [self.treated_city] + self.control_cities


CONFIG = ProjectConfig()

# Ensure directories exist
for p in [
    CONFIG.data_dir,
    CONFIG.raw_dir,
    CONFIG.interim_dir,
    CONFIG.processed_dir,
    CONFIG.outputs_dir,
    CONFIG.figures_dir,
    CONFIG.tables_dir,
    CONFIG.src_dir,
]:
    Path(p).mkdir(parents=True, exist_ok=True)
