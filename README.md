# Multi-Modal Cancer Prediction Data Pipeline

Airflow + PySpark + Delta Lake scaffold implementing the architecture in
`docs/architecture.md` (see the shared design doc for the full rationale).
Bronze -> silver -> gold layering across 4 domains: cBioPortal (clinical +
mutation), GDC (mutation metadata + RNA-Seq + DNA methylation), SEER
(clinical extract), and EPA AQS (PM2.5 environmental exposure).

## Layout

```
dags/                     Airflow DAGs (one per pipeline stage)
src/common/               Spark session + config + Delta I/O helpers shared everywhere
src/ingestion/            One extractor module per external source + bronze writer
src/preprocessing/        Missing values, outliers, HIPAA anonymization, patient linkage
src/feature_engineering/  Categorical/ordinal encoding, genomic dim reduction, fusion
src/training/             Patient-level split, XGBoost classifier, CoxPH/RSF survival models
data_quality/schemas.py   Pandera schemas gating each source before it lands in bronze
configs/                  Per-source config (study IDs, SEER layout, HIPAA zip3 list, ...)
```

## Pipeline stages (DAGs, in dependency order)

1. `1_ingest_cbioportal`, `1_ingest_gdc` (mutation, RNA-Seq, DNA methylation), `1_ingest_environmental` (scheduled), `1_ingest_seer` (manual trigger - no public API, requires a DUA-gated SEER*Stat export) -> bronze
2. `2_preprocessing` -> silver (cleaned, de-identified, patient-linked)
3. `3_feature_engineering` -> gold (`patient_features_tcga` and `patient_features_seer_environmental` - kept separate; see the DAG docstring for why the two cohorts can't be fused per-patient)
4. `4_model_training` -> XGBoost classifier + CoxPH/Random Survival Forest, logged to MLflow

DAG IDs are prefixed `1_`/`2_`/`3_`/`4_` so the Airflow UI's alphabetical DAG list sorts in pipeline order.

## Running locally

```
cp .env.example .env      # fill in EPA_AQS_API_KEY/EMAIL and PHI_SALT (GDC_TOKEN optional)
docker compose up -d
```

## API keys and where to get them

Fill credentials in `.env` (copied from `.env.example`). The containers read
these values automatically via `docker-compose.yml`.

- `GDC_TOKEN` (optional): generate from your GDC account token page:
	https://portal.gdc.cancer.gov/ (Profile -> Token). Required for
	controlled-access downloads only. The default `configs/sources.yaml`
	settings use open-access formats and run without this token.
- `EPA_AQS_API_KEY` and `EPA_AQS_EMAIL`: register at:
	https://aqs.epa.gov/data/api
- `CBIOPORTAL_API_TOKEN` (optional): only needed for private/self-hosted
	cBioPortal deployments. The public cBioPortal API works without a token.
- `PHI_SALT`: set a strong random secret used for pseudonymization hashing.

- Airflow UI: http://localhost:8080 (admin/admin)
- Spark master UI: http://localhost:8082
- MinIO console: http://localhost:9001 (minioadmin/minioadmin) - create the `cancer-lake` bucket once before the first DAG run
- MLflow runs are logged to the local `./mlruns` file store (no separate server in this scaffold)

## Known scaffold limitations

- SEER extractor expects a file at `data/seer/latest_extract.txt`; it has no API, drop your approved SEER*Stat export there before triggering `1_ingest_seer`.
- `gene_pathway_map.yaml` and `seer_layout.yaml` are illustrative subsets - replace with the real SEER data dictionary and a maintained pathway source (MSigDB/Reactome) before real use.
- Patient linkage (`preprocessing/patient_linkage.py`) is deterministic exact-match blocking, meant for one institution's own EHR-vs-genomic-vs-lab linkage; a fuzzy/probabilistic tier (e.g. `splink`) is a deliberate v2 scope cut.
- The Laboratory Biomarkers modality (CEA/PSA, blood panels) from the original architecture doc has no dedicated ingestion extractor here - it's assumed to arrive via the same hospital EHR/LIS feed as clinical data, and isn't wired up as a separate public source since none was specified.
