"""Pull clinical + somatic mutation data from the cBioPortal public REST API.

Docs: https://www.cbioportal.org/api/swagger-ui/index.html
No auth needed for the public instance; self-hosted/private instances need
an API token passed via the `Authorization` header instead.
"""
import logging

import pandas as pd
import requests

from common.config import env

logger = logging.getLogger(__name__)

API_BASE = env("CBIOPORTAL_API_BASE", "https://www.cbioportal.org/api")
PAGE_SIZE = 5000
TIMEOUT = 60


def _auth_headers() -> dict:
    token = env("CBIOPORTAL_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _paged_get(path: str, params: dict) -> list[dict]:
    results, page = [], 0
    while True:
        resp = requests.get(
            f"{API_BASE}{path}",
            params={**params, "pageSize": PAGE_SIZE, "pageNumber": page},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        batch = resp.json()
        results.extend(batch)
        if len(batch) < PAGE_SIZE:
            return results
        page += 1


def fetch_clinical_data(study_id: str) -> pd.DataFrame:
    """Demographics, TNM stage, survival outcome - normalized to one row per
    (sample, attribute).

    cBioPortal splits clinical attributes across two endpoints: SAMPLE-level
    (CANCER_TYPE, MUTATION_COUNT, ...) and PATIENT-level (AGE, SEX, OS_MONTHS,
    OS_STATUS, AJCC_PATHOLOGIC_TUMOR_STAGE, ...). Downstream cleaning pivots to
    one row per sample, so each patient attribute is broadcast onto every
    sample belonging to that patient here.
    """
    sample_rows = _paged_get(
        f"/studies/{study_id}/clinical-data",
        params={"clinicalDataType": "SAMPLE", "projection": "DETAILED"},
    )
    patient_rows = _paged_get(
        f"/studies/{study_id}/clinical-data",
        params={"clinicalDataType": "PATIENT", "projection": "DETAILED"},
    )
    if not sample_rows:
        raise ValueError(f"cBioPortal returned no clinical data for study_id={study_id}")

    keep = ["patientId", "sampleId", "clinicalAttributeId", "value"]
    sample_df = pd.DataFrame(sample_rows)[keep]

    frames = [sample_df]
    if patient_rows:
        patient_long = pd.DataFrame(patient_rows)[["patientId", "clinicalAttributeId", "value"]]
        sample_map = sample_df[["patientId", "sampleId"]].drop_duplicates()
        frames.append(sample_map.merge(patient_long, on="patientId", how="inner")[keep])

    return pd.concat(frames, ignore_index=True)


def fetch_mutations(study_id: str, molecular_profile_id: str, sample_list_id: str) -> pd.DataFrame:
    """Somatic mutation calls (MAF-equivalent) for a study's mutation profile."""
    resp = requests.post(
        f"{API_BASE}/molecular-profiles/{molecular_profile_id}/mutations/fetch",
        json={"sampleListId": sample_list_id},
        params={"projection": "DETAILED"},
        headers=_auth_headers(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError(f"cBioPortal returned no mutations for study_id={study_id}")

    df = pd.DataFrame(rows)
    # cBioPortal nests the gene as an object {hugoGeneSymbol, entrezGeneId, ...};
    # downstream pathway mapping joins on a plain HUGO symbol string.
    if "gene" in df.columns:
        df["gene"] = df["gene"].map(lambda g: g.get("hugoGeneSymbol") if isinstance(g, dict) else g)
    return df
