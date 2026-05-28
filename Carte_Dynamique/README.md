## Project Overview

A Python data-science pipeline that fetches, preprocesses, clusters, and models COVID-19 epidemiological data for all 101 French départements (metropolitan + DOM). The pipeline is divided into four sequential phases; each phase consumes the outputs of the previous one.

## Running the Pipeline

All scripts use relative paths (e.g., `data/raw/...`) and must be executed from the directory that contains the `data/` folder. The canonical working directory is `src/`.

```bash
cd src
python "phase1_data_acquisition 1.py"   # downloads all raw data (~10–20 min, network-heavy)
python phase2_preprocessing.py          # feature engineering + master time-series
python phase3_clustering.py             # PCA + K-Means + DBSCAN
python phase4_Ml-models.py             # regression models + hyperparameter tuning
```

There is no test suite, linter config, or build system. The `bin/` directory is a mirror of `src/` (VS Code Java scaffolding artefact); the canonical sources are in `src/`.

## Dependencies

Please find `requirements.txt` file

## Data Layout

```
src/data/
  raw/          geographic GeoJSON, hospitalisations, population (INSEE)
  labs/         SI-DEP positivity-rate CSVs (old weekly + new daily)
  vaccines/     VAC-SI vaccination coverage CSVs + 14-day lag file
  locations/    vaccination centres GPS
  processed/    master_dept_daily.csv, master_dept_latest.csv,
                master_dept_geo.geojson, wave_periods.csv,
                dept_clusters_features.csv
  clustering/   PCA coords, K-Means/DBSCAN labels, cluster profiles
  models/       best_model.pkl, feature_importance.csv, predictions
```

## Architecture

### Phase 1 — Data Acquisition (`phase1_data_acquisition 1.py`)
Fetches 17 data sources (data.gouv.fr API, opendatasoft, GitHub mirrors, Ameli, DREES). Key helpers: `normalize_dep()` (zero-pads département codes to 2 chars, preserves Corsica 2A/2B and DOM 971–976), `fetch_latest_datagouv_url()` (resolves latest resource from a dataset slug), `apply_age_filter()` (keeps all-ages rows via `cl_age90`/`cage10`). Post-processing adds per-100K rates and a 14-day vaccination lag. Emits a merge-readiness check at the end.

### Phase 2 — Preprocessing (`phase2_preprocessing.py`)
Builds a single daily time-series per département by merging hospitalisations (SPF), SI-DEP lab tests, vaccination, and epidemiological indicators (R0/Ti/TO). Key derived features: `new_hosp`/`new_dc` (diff of stock), 7-day rolling averages, `cfr`, `icu_pressure`, `tx_pos_adj_vacc` (positivity adjusted for vaccine coverage). Wave labels (7 waves, 2020–2023) are assigned via a vectorised date-range merge. The OpenCovid19-FR dataset fills the pre-SPF 2020 gap. Rogue département codes (e.g. "00", "20") are dropped using the INSEE population table as the authoritative code list.

### Phase 3 — Clustering (`phase3_clustering.py`)
Inputs: `dept_clusters_features.csv` (101 depts × ~19 aggregated features). Pipeline: RobustScaler → full PCA (variance analysis) → 2D/3D PCA projections → K-Means elbow+silhouette+CH consensus → DBSCAN parameter search in PCA-reduced space. DOM départements (971–976) are set aside before metropolitan clustering and re-analysed separately. Wave-level clustering tracks whether a département's epidemic profile shifted across waves.

### Phase 4 — ML Models (`phase4_Ml-models.py`)
Target: `hosp_rate` (hospitalisations per 100K). Features: lagged predictors (`tx_pos_7j` lag-7, `couv_complet` lag-14, `R0_7j` lag-3, `Ti` lag-7, `new_hosp_rate` lag-1) plus contextual features (wave, cluster, temporal). Chronological 80/20 train-test split with `TimeSeriesSplit` CV. Models: Baseline, Ridge, Lasso, Random Forest, GradientBoosting, SVR — all wrapped in `RobustScaler` pipelines. The best model undergoes `RandomizedSearchCV` tuning and is serialised as `data/models/best_model.pkl`.

## Key Conventions

- **Département codes**: always zero-padded 2-char strings (`"01"`…`"95"`) except Corsica (`"2A"`, `"2B"`) and DOM (`"971"`–`"976"`). `normalize_dep()` is the single normalisation point; call it on any external column before joining.
- **Date column**: always renamed to `"jour"` after loading; all sources parse it with `pd.to_datetime(..., errors="coerce")`.
- **All-ages filter**: SI-DEP datasets contain age-stratified rows; `apply_age_filter()` keeps only `cl_age90 == "0"` (or `cage10 == "00"`) rows.
- **External sources can fail**: every network fetch is wrapped in `try/except` with a warning print and graceful degradation. Non-critical sources (FINESS, CovidTracker variants, SI-VIC) are skipped without aborting the phase.
- Phase 5 (interactive Folium map) is referenced in the Phase 4 report but not yet implemented.
