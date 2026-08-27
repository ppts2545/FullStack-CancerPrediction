"""Pull PM2.5 exposure data from the EPA Air Quality System (AQS) API.

Docs: https://aqs.epa.gov/aqsweb/documents/data_api.html
Requires free registration (email + key) at https://aqs.epa.gov/data/api.
Param code 88101 = PM2.5 FRM/FEM Mass (ug/m3).
"""
import logging

import pandas as pd
import requests

from common.config import env

logger = logging.getLogger(__name__)

API_BASE = "https://aqs.epa.gov/data/api"
PM25_PARAM_CODE = "88101"
TIMEOUT = 60


def fetch_pm25_by_county(state_code: str, county_code: str, bdate: str, edate: str) -> pd.DataFrame:
    """bdate/edate format: YYYYMMDD. Returns one row per monitor-site-day."""
    resp = requests.get(
        f"{API_BASE}/dailyData/byCounty",
        params={
            "email": env("EPA_AQS_EMAIL", required=True),
            "key": env("EPA_AQS_API_KEY", required=True),
            "param": PM25_PARAM_CODE,
            "bdate": bdate,
            "edate": edate,
            "state": state_code,
            "county": county_code,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()

    status = payload.get("Header", [{}])[0].get("status")
    if status and status != "Success":
        raise RuntimeError(f"EPA AQS request failed: {payload['Header']}")

    rows = payload.get("Data", [])
    if not rows:
        raise ValueError(
            f"EPA AQS returned no PM2.5 data for state={state_code} county={county_code} "
            f"between {bdate} and {edate}"
        )
    return pd.DataFrame(rows)
