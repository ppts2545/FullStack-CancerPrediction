"""Pull mutation (MAF) and RNA-Seq expression metadata/files from GDC.

Docs: https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/
Controlled-access files (raw germline VCFs, some MAFs) require a dbGaP-issued
GDC_TOKEN with `X-Auth-Token` header. Open-access files (harmonized MAF,
FPKM/expression TSV) need no auth and are the default workflow in this project.
"""
import io
import json
import logging
import time

import pandas as pd
import requests

from common.config import env

logger = logging.getLogger(__name__)

API_BASE = env("GDC_API_BASE", "https://api.gdc.cancer.gov")
TIMEOUT = 120


def _auth_headers() -> dict:
    token = env("GDC_TOKEN")
    return {"X-Auth-Token": token} if token else {}


def search_files(
    project_id: str,
    data_category: str,
    data_format: str | None = None,
    size: int = 1000,
    access: str | None = "open",
) -> pd.DataFrame:
    """List file metadata (file_id, case_id, ...) matching a project + data type filter.

    Defaults to access="open" so results never include controlled-access files that
    would 403 on download without a dbGaP-issued GDC_TOKEN (see module docstring).
    """
    items = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_category", "value": [data_category]}},
    ]

    if data_format:
        formats = [data_format.upper(), data_format.lower()]
        items.append(
            {"op": "in", "content": {"field": "data_format", "value": formats}}
        )

    if access:
        items.append({"op": "in", "content": {"field": "access", "value": [access]}})

    filters = {
        "op": "and",
        "content": items
    }

    resp = requests.get(
        f"{API_BASE}/files",
        params={
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,cases.case_id,cases.submitter_id,data_type,data_format",
            "format": "JSON",
            "size": size,
        },
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )

    resp.raise_for_status()
    hits = resp.json().get("data", {}).get("hits", [])

    if not hits:
        raise ValueError(
            f"GDC returned no files for project_id={project_id}, "
            f"data_category={data_category}, data_format={data_format}, access={access}"
        )

    # GDC always nests "cases" as a list of dicts, even for a single case -
    # record_path unpacks it so downstream code can select "cases.case_id" directly.
    return pd.json_normalize(
        hits,
        record_path="cases",
        meta=["file_id", "file_name", "data_type", "data_format"],
        record_prefix="cases.",
    )


def download_file(file_id: str, max_retries: int = 4) -> bytes:
    """The GDC data endpoint drops connections mid-stream fairly often
    (ChunkedEncodingError / IncompleteRead), so retry transient failures with
    backoff. 401/403 is permanent (controlled-access) - fail fast on those.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{API_BASE}/data/{file_id}", headers=_auth_headers(), timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError as exc:
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    "GDC denied file download. This file may be controlled-access; "
                    "switch to open-access files or set GDC_TOKEN."
                ) from exc
            last_err = exc
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            last_err = exc
        logger.warning("GDC download %s attempt %d failed: %s", file_id, attempt + 1, last_err)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GDC download failed for {file_id} after {max_retries} attempts: {last_err}")


def fetch_rna_seq_counts(file_id: str) -> pd.DataFrame:
    """GDC STAR-Counts TSV: a '# gene-model' line, then the header, then four
    N_* summary rows, then one row per gene (gene_id, gene_name, gene_type,
    unstranded, stranded_*, tpm_unstranded, fpkm_*)."""
    raw = download_file(file_id)
    return pd.read_csv(io.BytesIO(raw), sep="\t", comment="#", skiprows=1)


def fetch_methylation_beta_values(file_id: str) -> pd.DataFrame:
    """GDC methylation array level-3 beta files are headerless 2-column TSVs:
    <probe_id>\t<beta_value>, ~485k probe rows each."""
    raw = download_file(file_id)
    return pd.read_csv(
        io.BytesIO(raw), sep="\t", comment="#", header=None, names=["probe_id", "beta_value"]
    )
