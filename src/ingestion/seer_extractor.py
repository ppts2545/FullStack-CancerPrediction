"""Parse SEER Research Data fixed-width ASCII extracts.

SEER has no public REST API - access requires a signed Data Use Agreement
(https://seer.cancer.gov/data/access.html), after which SEER*Stat exports a
fixed-width .txt file. Column positions come from the SEER data dictionary
for the requested database version, kept in configs/seer_layout.yaml so a
DUA-version bump doesn't require touching code.
"""
import logging
from pathlib import Path

import pandas as pd

from common.config import load_yaml_config

logger = logging.getLogger(__name__)


def fetch_seer_extract(local_file_path: str, layout_config: str = "seer_layout.yaml") -> pd.DataFrame:
    path = Path(local_file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SEER extract not found at {local_file_path} - SEER has no API, "
            "the file must be exported via SEER*Stat under an approved DUA and "
            "mounted into the ingestion container first."
        )

    layout = load_yaml_config(layout_config)["fields"]
    colspecs = [(f["start"] - 1, f["end"]) for f in layout]  # SEER positions are 1-indexed inclusive
    names = [f["name"] for f in layout]

    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str)
    if df.empty:
        raise ValueError(f"SEER extract at {local_file_path} parsed to zero rows - check layout config")
    return df
