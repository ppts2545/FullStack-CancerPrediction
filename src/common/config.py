"""Central config: lake paths and env-driven credentials. No hardcoded secrets."""
import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "/opt/airflow/configs"))

LAKE_BUCKET = os.environ.get("LAKE_BUCKET", "cancer-lake")

# lake layout: s3a://<bucket>/<layer>/<domain>/...
BRONZE_ROOT = f"s3a://{LAKE_BUCKET}/bronze"
SILVER_ROOT = f"s3a://{LAKE_BUCKET}/silver"
GOLD_ROOT = f"s3a://{LAKE_BUCKET}/gold"


def bronze_path(domain: str) -> str:
    return f"{BRONZE_ROOT}/{domain}"


def silver_path(domain: str) -> str:
    return f"{SILVER_ROOT}/{domain}"


def gold_path(domain: str) -> str:
    return f"{GOLD_ROOT}/{domain}"


def load_yaml_config(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value
