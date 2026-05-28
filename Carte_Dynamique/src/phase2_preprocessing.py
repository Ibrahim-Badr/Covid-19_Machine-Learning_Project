# ============================================================
# CARTE DYNAMIQUE — Phase 2: Preprocessing & Feature Engineering
# ============================================================
#
# INPUT  (data/raw/, data/vaccines/, data/labs/):
#   - departements.geojson              → geographic boundaries
#   - hospitalisations.csv              → hosp/rea/dc/rad stock (SPF)
#   - population_departements.csv       → INSEE 2020 census
#   - vaccination_dept.csv              → couv_complet by dept
#   - vaccination_lag14.csv             → couv_complet_lag14
#   - sidep_old.csv / sidep_new.csv     → tx_pos (positivity rate)
#   - indicateurs_suivi.csv             → R0 / Ti/100k / TO
#   - opencovid19_departements.csv      → 2020 gap filler
#   - hospitalisations_age_region.csv   → age stratification (region)
#   - sivic_regional.csv                → vacc status × hosp (region)
#
# OUTPUT (data/processed/):
#   - master_dept_daily.csv             → full daily time-series per dept
#   - master_dept_latest.csv            → latest snapshot per dept (map layer)
#   - master_dept_geo.geojson           → merged GeoJSON for Folium
#   - wave_periods.csv                  → labelled epidemic wave periods
#   - dept_clusters_features.csv        → feature matrix for clustering (Phase 3)
# ============================================================

import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs("data/processed", exist_ok=True)

# ── Constants ────────────────────────────────────────────────
ROLLING_WINDOW = 7        # 7-day rolling average
LAG_VACC_DAYS  = 14       # immune response delay
RATE_BASE      = 100_000  # per-100K normalization base
DOM_DEPS       = {"971", "972", "973", "974", "976"}


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV if it exists, return empty DataFrame otherwise."""
    if not os.path.exists(path):
        print(f"  ⚠️  Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def normalize_dep(series: pd.Series) -> pd.Series:
    """Normalize département codes to zero-padded 2-char strings."""
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace("DEP-", "", regex=False)
    s = s.str.zfill(2)
    return s


def to_date(series: pd.Series) -> pd.Series:
    """Safely parse a date column to datetime."""
    return pd.to_datetime(series, errors="coerce")


def rolling_mean(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """7-day rolling mean, min_periods=1."""
    return series.rolling(window=window, min_periods=1).mean()


def rate_100k(values: pd.Series, population: pd.Series) -> pd.Series:
    """Compute rate per 100K population."""
    return (pd.to_numeric(values, errors="coerce") / population * RATE_BASE).round(4)


# ============================================================
# STEP 1 — Load All Phase 1 Artefacts
# ============================================================

print("=" * 60)
print("📂 STEP 1 — Loading Phase 1 artefacts")
print("=" * 60)

# Geographic boundaries
gdf_dept = gpd.read_file("data/raw/departements.geojson")
gdf_dept["code"] = normalize_dep(gdf_dept["code"])
print(f"  ✓ GeoJSON: {len(gdf_dept)} départements")

# Population
df_pop = safe_read_csv("data/raw/population_departements.csv", dtype={"dep": str})
df_pop["dep"]        = normalize_dep(df_pop["dep"])
df_pop["population"] = pd.to_numeric(df_pop["population"], errors="coerce")
print(f"  ✓ Population: {len(df_pop)} depts | Total: {df_pop['population'].sum():,.0f}")

# Valid département codes (used for rogue-code filter in Step 7) [BUG-2]
VALID_DEPS = set(df_pop["dep"].unique())

# Hospitalisations
df_hosp = safe_read_csv("data/raw/hospitalisations.csv", dtype={"dep": str})
df_hosp["dep"]  = normalize_dep(df_hosp["dep"])
df_hosp["jour"] = to_date(df_hosp["jour"])
for _c in ["hosp", "rea", "dc", "rad"]:
    df_hosp[_c] = pd.to_numeric(df_hosp.get(_c, pd.Series(dtype=float)), errors="coerce")
print(f"  ✓ Hospitalisations: {len(df_hosp):,} rows | "
      f"{df_hosp['jour'].min().date()} → {df_hosp['jour'].max().date()}")

# Vaccination (all ages)
df_vaccin = safe_read_csv("data/vaccines/vaccination_dept.csv", dtype={"dep": str})
if not df_vaccin.empty:
    df_vaccin["dep"]  = normalize_dep(df_vaccin["dep"])
    df_vaccin["jour"] = to_date(df_vaccin["jour"])
    for _c in ["n_cum_dose1", "n_cum_complet", "couv_dose1", "couv_complet"]:
        if _c in df_vaccin.columns:
            df_vaccin[_c] = pd.to_numeric(df_vaccin[_c], errors="coerce")
    print(f"  ✓ Vaccination: {len(df_vaccin):,} rows")

# Vaccination lag14
df_vaccin_lag = safe_read_csv("data/vaccines/vaccination_lag14.csv", dtype={"dep": str})
if not df_vaccin_lag.empty:
    df_vaccin_lag["dep"]  = normalize_dep(df_vaccin_lag["dep"])
    df_vaccin_lag["jour"] = to_date(df_vaccin_lag["jour"])
    print(f"  ✓ Vaccination lag14: {len(df_vaccin_lag):,} rows")

# SI-DEP old + new
df_sidep_old = safe_read_csv("data/labs/sidep_old.csv", dtype={"dep": str})
df_sidep_new = safe_read_csv("data/labs/sidep_new.csv", dtype={"dep": str})

# Indicateurs (R0, Ti, TO) — [WARN-1] low_memory=False added
df_indic = safe_read_csv(
    "data/labs/indicateurs_suivi.csv", dtype={"dep": str}, low_memory=False
)
if not df_indic.empty:
    df_indic["dep"]  = normalize_dep(df_indic["dep"])
    df_indic["jour"] = to_date(df_indic["jour"])
    print(f"  ✓ Indicateurs: {len(df_indic):,} rows")

# OpenCovid19 gap filler — [WARN-1] low_memory=False added
df_oc = safe_read_csv(
    "data/raw/opencovid19_departements.csv", dtype={"dep": str}, low_memory=False
)
if not df_oc.empty:
    df_oc["dep"]  = normalize_dep(df_oc["dep"])
    df_oc["date"] = to_date(df_oc["date"])
    df_oc         = df_oc.rename(columns={"date": "jour"})
    print(f"  ✓ OpenCovid19: {len(df_oc):,} rows")


# ============================================================
# STEP 2 — Build Hospitalisations Base Time-Series
# Adds: daily new admissions (diff), 7d rolling averages,
#       rates per 100K, cumulative deaths per 100K
# ============================================================

print("\n" + "=" * 60)
print("🏥 STEP 2 — Hospitalisations: daily diff + rates")
print("=" * 60)

df_hosp = (
    df_hosp
    .sort_values(["dep", "jour"])
    .merge(df_pop[["dep", "population"]], on="dep", how="left")
)

# Daily new admissions / deaths / recoveries (diff of stock)
for _col, _new in [("hosp", "new_hosp"), ("rea", "new_rea"),
                   ("dc",   "new_dc"),   ("rad", "new_rad")]:
    if _col in df_hosp.columns:
        df_hosp[_new] = (
            df_hosp.groupby("dep")[_col]
            .diff()
            .clip(lower=0)  # negative diffs = data corrections → floor at 0
        )

# 7-day rolling averages
for _col in ["new_hosp", "new_rea", "new_dc", "hosp", "rea"]:
    if _col in df_hosp.columns:
        df_hosp[f"{_col}_7j"] = (
            df_hosp.groupby("dep")[_col]
            .transform(lambda s: rolling_mean(s))
        )

# Rates per 100K
for _col in ["hosp", "rea", "dc", "new_hosp", "new_dc"]:
    if _col in df_hosp.columns:
        df_hosp[f"{_col}_rate"] = rate_100k(df_hosp[_col], df_hosp["population"])

# [MIN-1] Drop 'sexe' — always 0 after Phase 1 filter, carries no information
df_hosp = df_hosp.drop(columns=["sexe"], errors="ignore")

print(f"  ✓ Columns added: "
      f"{[c for c in df_hosp.columns if c not in ['dep','jour','population']]}")
print(f"  ✓ Date range: {df_hosp['jour'].min().date()} → {df_hosp['jour'].max().date()}")


# ============================================================
# STEP 3 — Unify SI-DEP: tx_pos + tx_incid
# old (weekly): semaine_glissante | new (daily): jour
# Output: one daily-ish series per dept with tx_pos
# ============================================================

print("\n" + "=" * 60)
print("🔬 STEP 3 — SI-DEP: unify old + new positivity rate")
print("=" * 60)

sidep_frames = []

# --- Old format (weekly, semaine_glissante column) ---
if not df_sidep_old.empty:
    df_sidep_old["dep"] = normalize_dep(df_sidep_old["dep"])
    date_col_old = next(
        (c for c in df_sidep_old.columns
         if c.lower() in ["semaine_glissante", "jour", "date"]), None
    )
    if date_col_old:
        _raw = df_sidep_old[date_col_old].astype(str)
        df_sidep_old["jour"] = to_date(_raw.str[:10] if _raw.str.len().max() > 10 else _raw)
    for _c in ["P", "T", "tx_pos"]:
        if _c in df_sidep_old.columns:
            df_sidep_old[_c] = pd.to_numeric(df_sidep_old[_c], errors="coerce")
    if "tx_pos" not in df_sidep_old.columns and "P" in df_sidep_old.columns:
        df_sidep_old["tx_pos"] = (df_sidep_old["P"] / df_sidep_old["T"] * 100).round(2)
    keep_old = ["dep", "jour", "tx_pos"] + [c for c in ["P", "T"]
                                             if c in df_sidep_old.columns]
    sidep_frames.append(df_sidep_old[[c for c in keep_old if c in df_sidep_old.columns]])
    print(f"  ✓ SI-DEP old: {len(df_sidep_old):,} rows (weekly) | "
          f"{df_sidep_old['jour'].min().date()} → {df_sidep_old['jour'].max().date()}")

# --- New format (daily) ---
if not df_sidep_new.empty:
    df_sidep_new["dep"] = normalize_dep(df_sidep_new["dep"])
    # Normalize source-specific column names (Tp = positivity %, Ti = incidence/100K)
    _sidep_new_renames = {}
    if "Tp" in df_sidep_new.columns and "tx_pos" not in df_sidep_new.columns:
        _sidep_new_renames["Tp"] = "tx_pos"
    if "Ti" in df_sidep_new.columns and "ti" not in df_sidep_new.columns:
        _sidep_new_renames["Ti"] = "ti"
    if _sidep_new_renames:
        df_sidep_new = df_sidep_new.rename(columns=_sidep_new_renames)
        print(f"  ℹ️  SI-DEP new renames: {_sidep_new_renames}")
    date_col_new = next(
        (c for c in df_sidep_new.columns if c.lower() in ["jour", "date"]), None
    )
    if date_col_new:
        df_sidep_new["jour"] = to_date(df_sidep_new[date_col_new])
    for _c in ["P", "T", "tx_pos", "ti"]:
        if _c in df_sidep_new.columns:
            df_sidep_new[_c] = pd.to_numeric(df_sidep_new[_c], errors="coerce")
    if "tx_pos" not in df_sidep_new.columns and "P" in df_sidep_new.columns:
        df_sidep_new["tx_pos"] = (df_sidep_new["P"] / df_sidep_new["T"] * 100).round(2)
    keep_new = ["dep", "jour", "tx_pos"] + [c for c in ["P", "T", "ti"]
                                             if c in df_sidep_new.columns]
    sidep_frames.append(df_sidep_new[[c for c in keep_new if c in df_sidep_new.columns]])
    print(f"  ✓ SI-DEP new: {len(df_sidep_new):,} rows (daily) | "
          f"{df_sidep_new['jour'].min().date()} → {df_sidep_new['jour'].max().date()}")

if sidep_frames:
    df_sidep = (
        pd.concat(sidep_frames, ignore_index=True)
        .drop_duplicates(subset=["dep", "jour"])
        .sort_values(["dep", "jour"])
    )
    df_sidep["tx_pos_7j"] = (
        df_sidep.groupby("dep")["tx_pos"]
        .transform(lambda s: rolling_mean(s))
    )
    print(f"  ✓ Unified SI-DEP: {len(df_sidep):,} rows | "
          f"{df_sidep['jour'].min().date()} → {df_sidep['jour'].max().date()}")
else:
    df_sidep = pd.DataFrame(columns=["dep", "jour", "tx_pos", "tx_pos_7j"])
    print("  ⚠️  No SI-DEP data loaded")


# ============================================================
# STEP 4 — Vaccination: merge dept coverage + lag14
# ============================================================

print("\n" + "=" * 60)
print("💉 STEP 4 — Vaccination: coverage + lag14 merge")
print("=" * 60)

if not df_vaccin.empty:
    vacc_cols = ["dep", "jour"] + [
        c for c in ["couv_dose1", "couv_complet", "n_cum_dose1", "n_cum_complet"]
        if c in df_vaccin.columns
    ]
    df_vacc_clean = df_vaccin[vacc_cols].copy()

    if not df_vaccin_lag.empty and "couv_complet_lag14" in df_vaccin_lag.columns:
        df_vacc_clean = df_vacc_clean.merge(
            df_vaccin_lag[["dep", "jour", "couv_complet_lag14"]],
            on=["dep", "jour"], how="left"
        )
        print("  ✓ couv_complet_lag14 merged")
    else:
        df_vacc_clean = df_vacc_clean.sort_values(["dep", "jour"])
        df_vacc_clean["couv_complet_lag14"] = (
            df_vacc_clean.groupby("dep")["couv_complet"].shift(LAG_VACC_DAYS)
        )
        print("  ✓ couv_complet_lag14 computed on-the-fly")

    print(f"  ✓ Vaccination clean: {len(df_vacc_clean):,} rows | "
          f"Columns: {list(df_vacc_clean.columns)}")
else:
    df_vacc_clean = pd.DataFrame(columns=["dep", "jour"])
    print("  ⚠️  No vaccination data available")


# ============================================================
# STEP 5 — Indicateurs de Suivi: R0, Ti/100k, TO
# Dynamic column detection (column names vary by mirror source)
# ============================================================

print("\n" + "=" * 60)
print("📉 STEP 5 — Indicateurs épidémiques: R0 / Ti / TO")
print("=" * 60)

df_indic_clean = pd.DataFrame(columns=["dep", "jour"])

if not df_indic.empty:
    _col_map = {}
    for _target, _aliases in {
        "R0":     ["r", "r0", "r_effectif", "taux_reproduction"],
        "Ti":     ["ti", "ti100k", "taux_incidence", "incidence_rate", "tx_incid"],
        "TO":     ["to", "tx_occupation", "sae_rate", "to_rea"],
        "tx_pos": ["tx_pos", "taux_positivite", "positive_test_rate"],
    }.items():
        found = next((c for c in df_indic.columns if c.lower() in _aliases), None)
        if found:
            _col_map[found] = _target

    df_indic_clean = df_indic[["dep", "jour"] + list(_col_map.keys())].copy()
    df_indic_clean = df_indic_clean.rename(columns=_col_map)

    for _c in ["R0", "Ti", "TO", "tx_pos"]:
        if _c in df_indic_clean.columns:
            df_indic_clean[_c] = pd.to_numeric(df_indic_clean[_c], errors="coerce")

    if "R0" in df_indic_clean.columns:
        df_indic_clean["R0_7j"] = (
            df_indic_clean.sort_values(["dep", "jour"])
            .groupby("dep")["R0"]
            .transform(lambda s: rolling_mean(s))
        )

    print(f"  ✓ Indicateurs mapped: {list(_col_map.values())}")
    print(f"  ✓ {len(df_indic_clean):,} rows | "
          f"{df_indic_clean['jour'].min().date()} → {df_indic_clean['jour'].max().date()}")


# ============================================================
# STEP 6 — OpenCovid19-FR Gap Fill (pre-SPF period: 2020)
# Fills hosp/dc/rea for dates before 2020-03-18 (SPF start)
# ============================================================

print("\n" + "=" * 60)
print("🔓 STEP 6 — OpenCovid19-FR gap fill (pre-SPF 2020)")
print("=" * 60)

SPF_START = pd.Timestamp("2020-03-18")

if not df_oc.empty:
    oc_cols = {
        "hospitalises":  "hosp",
        "reanimation":   "rea",
        "deces":         "dc",
        "gueris":        "rad",
        "cas_confirmes": "cas_confirmes",
    }
    oc_available = {k: v for k, v in oc_cols.items() if k in df_oc.columns}
    df_oc_gap = df_oc[df_oc["jour"] < SPF_START][
        ["dep", "jour"] + list(oc_available.keys())
    ].copy()
    df_oc_gap = df_oc_gap.rename(columns=oc_available)

    for _c in oc_available.values():
        df_oc_gap[_c] = pd.to_numeric(df_oc_gap[_c], errors="coerce")

    df_oc_gap = df_oc_gap.merge(df_pop[["dep", "population"]], on="dep", how="left")

    for _c in ["hosp", "rea", "dc"]:
        if _c in df_oc_gap.columns:
            df_oc_gap[f"{_c}_rate"] = rate_100k(df_oc_gap[_c], df_oc_gap["population"])

    print(f"  ✓ Gap fill: {len(df_oc_gap):,} rows | "
          f"{df_oc_gap['jour'].min().date()} → {df_oc_gap['jour'].max().date()}")
else:
    df_oc_gap = pd.DataFrame()
    print("  ⚠️  OpenCovid19 data missing — 2020 gap unfilled")


# ============================================================
# STEP 7 — Master Merge: Build Daily Time-Series per Dept
# Join order: hosp → SI-DEP → vaccination → indicateurs
# ============================================================

print("\n" + "=" * 60)
print("🔗 STEP 7 — Master merge: daily time-series per dept")
print("=" * 60)

df_master = df_hosp.copy()

# Merge SI-DEP (tx_pos, tx_pos_7j)
if not df_sidep.empty:
    df_master = df_master.merge(
        df_sidep[["dep", "jour", "tx_pos", "tx_pos_7j"]],
        on=["dep", "jour"], how="left"
    )
    print(f"  ✓ SI-DEP merged: tx_pos coverage = "
          f"{df_master['tx_pos'].notna().mean()*100:.1f}%")

# Merge vaccination
if not df_vacc_clean.empty and "jour" in df_vacc_clean.columns:
    df_master = df_master.merge(df_vacc_clean, on=["dep", "jour"], how="left")
    if "couv_complet" in df_master.columns:
        print(f"  ✓ Vaccination merged: couv_complet coverage = "
              f"{df_master['couv_complet'].notna().mean()*100:.1f}%")

# Merge indicateurs (R0, Ti, TO)
if not df_indic_clean.empty and len(df_indic_clean.columns) > 2:
    indic_merge_cols = ["dep", "jour"] + [
        c for c in ["R0", "R0_7j", "Ti", "TO", "tx_pos"]
        if c in df_indic_clean.columns and c not in df_master.columns
    ]
    df_master = df_master.merge(
        df_indic_clean[indic_merge_cols], on=["dep", "jour"], how="left"
    )
    print(f"  ✓ Indicateurs merged: R0 coverage = "
          f"{df_master.get('R0', pd.Series()).notna().mean()*100:.1f}%")

# Prepend 2020 gap from OpenCovid19
if not df_oc_gap.empty:
    df_master = pd.concat([df_oc_gap, df_master], ignore_index=True)
    df_master = (
        df_master
        .drop_duplicates(subset=["dep", "jour"])
        .sort_values(["dep", "jour"])
    )
    print("  ✓ OpenCovid19 gap prepended")

# [BUG-2] Drop rogue département codes (e.g. "00", "20") not in INSEE population table
rogue_codes = set(df_master["dep"].unique()) - VALID_DEPS
if rogue_codes:
    print(f"  ⚠️  Dropping rogue dep codes: {sorted(rogue_codes)}")
    df_master = df_master[df_master["dep"].isin(VALID_DEPS)].copy()

print(f"\n  ✓ Master shape     : {df_master.shape}")
print(f"  ✓ Columns          : {list(df_master.columns)}")
print(f"  ✓ Date range       : {df_master['jour'].min().date()} → {df_master['jour'].max().date()}")
print(f"  ✓ Départements     : {df_master['dep'].nunique()} "
      f"({'✅ 101' if df_master['dep'].nunique() == 101 else '⚠️  unexpected count'})")


# ============================================================
# STEP 8 — Derived Epidemiological Features
# ============================================================

print("\n" + "=" * 60)
print("⚗️  STEP 8 — Derived epidemiological features")
print("=" * 60)

df_master = df_master.sort_values(["dep", "jour"])

# CFR — Case Fatality Rate (dc / hosp, %)
if "dc" in df_master.columns and "hosp" in df_master.columns:
    df_master["total_hosp_cases"] = (
        pd.to_numeric(df_master["dc"],  errors="coerce").fillna(0)
        + pd.to_numeric(df_master["rad"], errors="coerce").fillna(0)
    )
    df_master["cfr"] = (
        pd.to_numeric(df_master["dc"], errors="coerce")
        / df_master["total_hosp_cases"].replace(0, np.nan)
        * 100
    ).round(4)

# ICU pressure ratio (rea / hosp, %)
if "rea" in df_master.columns and "hosp" in df_master.columns:
    df_master["icu_pressure"] = (
        pd.to_numeric(df_master["rea"],  errors="coerce")
        / pd.to_numeric(df_master["hosp"], errors="coerce").replace(0, np.nan)
        * 100
    ).round(4)

# Vaccination × epidemic interaction: tx_pos adjusted by vacc coverage lag14
if "tx_pos" in df_master.columns and "couv_complet_lag14" in df_master.columns:
    df_master["tx_pos_adj_vacc"] = (
        df_master["tx_pos"]
        * (1 - df_master["couv_complet_lag14"].fillna(0) / 100)
    ).round(4)
    print("  ✓ tx_pos_adj_vacc computed (positivity adjusted for vaccine coverage)")

# Temporal features (seasonality / wave analysis)
df_master["week"]        = df_master["jour"].dt.isocalendar().week.astype(int)
df_master["year"]        = df_master["jour"].dt.year
df_master["month"]       = df_master["jour"].dt.month
df_master["day_of_week"] = df_master["jour"].dt.dayofweek   # 0=Monday

# Weekend flag (reporting dip correction marker)
df_master["is_weekend"] = df_master["day_of_week"].isin([5, 6]).astype(int)

# DOM flag
df_master["is_dom"] = df_master["dep"].isin(DOM_DEPS).astype(int)

print("  ✓ Features added: cfr, icu_pressure, tx_pos_adj_vacc, "
      "week, year, month, day_of_week, is_weekend, is_dom")


# ============================================================
# STEP 9 — Epidemic Wave Detection
# Labels each date with a wave number (0 = inter-wave)
# [WARN-2] Vectorized merge replaces O(n×7) row-by-row Python loop
# ============================================================

print("\n" + "=" * 60)
print("🌊 STEP 9 — Epidemic wave detection")
print("=" * 60)

_WAVE_DEFINITIONS = [
    (1, "2020-03-01", "2020-06-30", "Vague 1 — printemps 2020"),
    (2, "2020-09-01", "2021-02-28", "Vague 2 — automne/hiver 2020"),
    (3, "2021-03-01", "2021-06-30", "Vague 3 — printemps 2021 (Alpha)"),
    (4, "2021-07-01", "2021-11-30", "Vague 4 — été 2021 (Delta)"),
    (5, "2021-12-01", "2022-03-31", "Vague 5 — hiver 2021/22 (Omicron BA.1)"),
    (6, "2022-04-01", "2022-07-31", "Vague 6 — printemps 2022 (Omicron BA.2/BA.5)"),
    (7, "2022-10-01", "2023-02-28", "Vague 7 — hiver 2022/23"),
]

df_waves = pd.DataFrame(_WAVE_DEFINITIONS, columns=["wave", "start", "end", "label"])
df_waves["start"] = pd.to_datetime(df_waves["start"])
df_waves["end"]   = pd.to_datetime(df_waves["end"])
df_waves.to_csv("data/processed/wave_periods.csv", index=False)

# [WARN-2] Vectorized wave assignment — build a date→wave lookup then merge
df_date_wave = pd.DataFrame({"jour": df_master["jour"].unique()})
df_date_wave["wave"] = 0
for _, _row in df_waves.iterrows():
    _mask = (df_date_wave["jour"] >= _row["start"]) & (df_date_wave["jour"] <= _row["end"])
    df_date_wave.loc[_mask, "wave"] = int(_row["wave"])

df_master = df_master.merge(df_date_wave, on="jour", how="left")
df_master["wave"] = df_master["wave"].fillna(0).astype(int)

wave_counts = df_master["wave"].value_counts().sort_index()
print("  ✓ Wave labels assigned:")
for _w, _n in wave_counts.items():
    _label = df_waves.loc[df_waves["wave"] == _w, "label"].values
    _desc  = _label[0] if len(_label) else "inter-wave"
    print(f"       Wave {_w}: {_n:,} rows — {_desc}")


# ============================================================
# STEP 10 — Clustering Feature Matrix
# One row per département — aggregated statistics for ML Phase 3
# ============================================================

print("\n" + "=" * 60)
print("🧮 STEP 10 — Clustering feature matrix")
print("=" * 60)

# Keep only main epidemic period (waves 1–7) for stable aggregation
df_epi = df_master[df_master["wave"] > 0].copy()

_AGG_COLS = {}

for _c in ["hosp_rate", "new_hosp_rate", "rea_rate", "icu_pressure", "cfr"]:
    if _c in df_epi.columns:
        _AGG_COLS[_c] = ["mean", "max"]

if "tx_pos" in df_epi.columns:
    _AGG_COLS["tx_pos"] = ["mean", "max"]
if "tx_pos_7j" in df_epi.columns:
    _AGG_COLS["tx_pos_7j"] = ["mean"]
if "couv_complet" in df_epi.columns:
    _AGG_COLS["couv_complet"] = ["max"]
if "couv_complet_lag14" in df_epi.columns:
    _AGG_COLS["couv_complet_lag14"] = ["mean"]
# R0 excluded: 9.5% fill rate, quasi-constant after median-fill → adds noise, no discriminating power

if _AGG_COLS:
    df_cluster = df_epi.groupby("dep").agg(_AGG_COLS)
    df_cluster.columns = ["_".join(c).strip() for c in df_cluster.columns]
    df_cluster = df_cluster.reset_index()

    df_cluster = df_cluster.merge(df_pop[["dep", "population"]], on="dep", how="left")
    df_cluster["is_dom"] = df_cluster["dep"].isin(DOM_DEPS).astype(int)

    # Fill sparse depts with column medians
    numeric_cols = df_cluster.select_dtypes(include=[np.number]).columns
    df_cluster[numeric_cols] = df_cluster[numeric_cols].fillna(
        df_cluster[numeric_cols].median()
    )

    df_cluster.to_csv("data/processed/dept_clusters_features.csv", index=False)
    print(f"  ✓ Cluster matrix: {df_cluster.shape[0]} depts × {df_cluster.shape[1]} features")
    print(f"  ✓ Features: {list(df_cluster.columns)}")
else:
    df_cluster = pd.DataFrame()
    print("  ⚠️  No numeric columns available for clustering feature matrix")


# ============================================================
# STEP 11 — Latest Snapshot + GeoJSON Export
# One row per département — latest available date per metric
# [BUG-1] Drop 'nom' from df_latest before GeoJSON merge to avoid nom_x/nom_y collision
# ============================================================

print("\n" + "=" * 60)
print("🗺️  STEP 11 — Latest snapshot + GeoJSON export")
print("=" * 60)


def latest_per_dep(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return latest non-null value per dept for each column."""
    result = df[["dep"]].drop_duplicates().set_index("dep")
    for _col in cols:
        if _col in df.columns:
            _latest = (
                df[df[_col].notna()]
                .sort_values("jour")
                .groupby("dep")[_col].last()
            )
            result[_col] = _latest
    return result.reset_index()


_SNAPSHOT_COLS = [
    "hosp", "rea", "dc", "rad",
    "hosp_rate", "new_hosp_rate", "rea_rate",
    "new_hosp_7j", "cfr", "icu_pressure",
    "tx_pos", "tx_pos_7j", "tx_pos_adj_vacc",
    "couv_complet", "couv_complet_lag14", "couv_dose1",
    "R0", "R0_7j", "Ti", "TO",
    "wave",
]

df_latest = latest_per_dep(df_master, _SNAPSHOT_COLS)
df_latest = df_latest.merge(df_pop[["dep", "population"]], on="dep", how="left")
df_latest["is_dom"] = df_latest["dep"].isin(DOM_DEPS).astype(int)

# Add dept name from GeoJSON for the CSV snapshot
df_latest = df_latest.merge(
    gdf_dept[["code", "nom"]].rename(columns={"code": "dep"}),
    on="dep", how="left"
)
df_latest.to_csv("data/processed/master_dept_latest.csv", index=False)
print(f"  ✓ Latest snapshot: {df_latest.shape[0]} depts × {df_latest.shape[1]} cols")

# [BUG-1] Drop 'nom' before merging into GeoJSON — gdf_dept already carries 'nom',
#          keeping it in df_latest would create nom_x / nom_y suffix collision
gdf_map = gdf_dept.rename(columns={"code": "dep"}).merge(
    df_latest.drop(columns=["nom"], errors="ignore"),
    on="dep", how="left"
)
gdf_map.to_file("data/processed/master_dept_geo.geojson", driver="GeoJSON")
print(f"  ✓ GeoJSON exported: {len(gdf_map)} features | "
      f"Columns: {[c for c in gdf_map.columns if c != 'geometry'][:10]}...")


# ============================================================
# STEP 12 — Save Master Daily + Quality Report
# ============================================================

print("\n" + "=" * 60)
print("💾 STEP 12 — Save master daily + quality report")
print("=" * 60)

df_master.to_csv("data/processed/master_dept_daily.csv", index=False)
print(f"  ✓ master_dept_daily.csv saved: {df_master.shape}")

# ── Quality Report ───────────────────────────────────────────
print("\n  📋 DATA QUALITY REPORT")
print("  " + "-" * 50)

total_cells = df_master.shape[0] * df_master.shape[1]
null_cells  = df_master.isnull().sum().sum()
print(f"  Overall completeness  : {(1 - null_cells/total_cells)*100:.1f}%")
print(f"  Total rows            : {len(df_master):,}")
print(f"  Départements covered  : {df_master['dep'].nunique()} "
      f"({'✅ 101' if df_master['dep'].nunique() == 101 else '⚠️  check dep codes'})")
print(f"  Date range            : "
      f"{df_master['jour'].min().date()} → {df_master['jour'].max().date()}")
print(f"  Epidemic waves tagged : {df_master[df_master['wave']>0]['jour'].nunique()} days")

print("\n  Column-level completeness (key metrics):")
for _col in ["hosp", "rea", "dc", "tx_pos", "couv_complet", "R0", "Ti", "TO", "cfr"]:
    if _col in df_master.columns:
        _pct = df_master[_col].notna().mean() * 100
        # [MIN-2] round() instead of int() for accurate block count
        _filled = round(_pct / 5)
        _bar = "█" * _filled + "░" * (20 - _filled)
        print(f"    {_col:<22} [{_bar}] {_pct:5.1f}%")

# ── File Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📁 PHASE 2 OUTPUT FILES")
print("=" * 60)

_P2_FILES = [
    "data/processed/master_dept_daily.csv",
    "data/processed/master_dept_latest.csv",
    "data/processed/master_dept_geo.geojson",
    "data/processed/wave_periods.csv",
    "data/processed/dept_clusters_features.csv",
]
total_kb = 0
for _f in _P2_FILES:
    if os.path.exists(_f):
        _kb = os.path.getsize(_f) / 1024
        total_kb += _kb
        print(f"  ✓ {_f:<50} {_kb:>9.1f} KB")
    else:
        print(f"  ✗ {_f:<50} NOT CREATED")

print(f"\n  💾 Phase 2 output size: {total_kb/1024:.1f} MB")
print("\n🚀 Phase 2 complete — ready for Phase 3: Clustering (scikit-learn)")
