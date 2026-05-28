# ============================================================
# CARTE DYNAMIQUE — Phase 1: Full Data Acquisition (v10 - DEFINITIVE)
# ============================================================
#
# SOURCES:
#   1.  Geographic Boundaries       → gregoiredavid/france-geojson
#   2.  Kalisio COVID GeoJSON        → opendatasoft (day_hosp, day_intcare)
#   3.  Hospitalization Stock        → SPF / data.gouv.fr
#   3b. Daily incidence              → covered by Sources 2, 3, 10
#   3c. Hosp by Age (region-level)   → SPF / data.gouv.fr
#   4.  OpenCovid19-FR               → github.com/opencovid19-fr (2020 gap)
#   5.  Vaccination by Dept          → VAC-SI / data.gouv.fr
#   6.  Vaccination by Age + Dept    → VAC-SI / data.gouv.fr
#   7.  INSEE Population             → hardcoded 2020 census
#   8.  SI-DEP old (2020–2022)       → data.gouv.fr (weekly)
#   9.  SI-DEP new (2022–2023)       → data.gouv.fr (daily)
#   10. Indicateurs épidémiques      → AMP Métropole mirror (R0, Ti, TO)
#   11. Vaccination Centers GPS      → opendatasoft
#   12. Vaccination by Location      → Ameli (national)
#   13. Vaccination by Injector      → Ameli (national)
#   14. Vaccination by Brand         → VAC-SI vacsi-v-dep
#   15. COVID Variants               → DREES (national)
#   15b.CovidTracker Variants        → Rozier/github (dept-level attempt)
#   16. FINESS Establishments GPS    → Atlasanté / data.gouv.fr
#   17. SI-VIC hosp × vacc status    → DREES (regional)
#
# POST-PROCESSING:
#   - Rates per 100K                 → data/processed/
#   - Vaccination 14-day lag         → data/vaccines/
#
# ============================================================

import os

import geopandas as gpd
import pandas as pd
import requests

# ── Folder scaffold ──────────────────────────────────────────
for _folder in ["data/raw", "data/processed", "data/vaccines", "data/labs", "data/locations"]:
    os.makedirs(_folder, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_dep(series: pd.Series) -> pd.Series:
    """Normalize département codes to zero-padded 2-char strings (or 3 for DOM)."""
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace("DEP-", "", regex=False)
    s = s.str.zfill(2)
    return s


def fetch_latest_datagouv_url(dataset_id: str, resource_name_contains: str) -> str:
    """Return the URL of the most recently modified resource matching a name fragment."""
    resp = requests.get(
        f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/", timeout=15
    )
    resp.raise_for_status()
    matches = [
        r for r in resp.json().get("resources", [])
        if resource_name_contains.lower() in r["title"].lower()
    ]
    if not matches:
        raise ValueError(f"No resource found matching '{resource_name_contains}'")
    matches.sort(key=lambda r: r.get("last_modified", ""), reverse=True)
    return matches[0]["url"]


def debug_dataset_resources(dataset_id: str) -> None:
    """Print all resources of a data.gouv.fr dataset for debugging."""
    resp = requests.get(
        f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/", timeout=15
    )
    resp.raise_for_status()
    resources = resp.json().get("resources", [])
    print(f"\n  📋 {dataset_id} — {len(resources)} resources:")
    for r in sorted(resources, key=lambda x: x.get("last_modified", ""), reverse=True):
        print(f"     [{r.get('last_modified','?')[:10]}] [{r.get('format','?'):6}] {r['title']}")
        print(f"          → {r['url']}")


def _get_sidep_candidates(dataset_id: str) -> list:
    """Return sorted list of CSV resources for a SI-DEP dataset, dept-filtered if possible."""
    resp = requests.get(
        f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/", timeout=15
    )
    resp.raise_for_status()
    resources = resp.json().get("resources", [])
    csvs = [
        r for r in resources
        if r.get("format", "").lower() == "csv" or r["url"].endswith(".csv")
    ]
    dep_csvs = [r for r in csvs if "dep" in r["title"].lower()]
    return dep_csvs if dep_csvs else csvs


def load_sidep_dataset(dataset_id: str, label: str) -> pd.DataFrame:
    """
    Load SI-DEP département CSV.
    Priority: daily files (jour/quot) over weekly (7j/heb/ti-tp/hebdo).
    """
    candidates = _get_sidep_candidates(dataset_id)
    if not candidates:
        raise ValueError(f"No CSV resource found in dataset {dataset_id}")

    candidates.sort(key=lambda r: r.get("last_modified", ""), reverse=True)

    daily_kw  = ["jour", "quot"]
    weekly_kw = ["7j", "heb", "ti-tp", "hebdo"]
    daily_cands   = [r for r in candidates if any(kw in r["title"].lower() for kw in daily_kw)]
    nweekly_cands = [r for r in candidates
                     if not any(kw in r["title"].lower() for kw in weekly_kw)]
    best = (daily_cands or nweekly_cands or candidates)
    best.sort(key=lambda r: r.get("last_modified", ""), reverse=True)

    print(f"  ℹ️  [{label}] Candidates (daily preferred):")
    for r in best[:3]:
        print(f"       • {r['title']}")

    print(f"  ✓ Loading: {best[0]['title']}")
    _raw = pd.read_csv(best[0]["url"], sep=";", dtype=str, low_memory=False)
    return _fix_comma_decimals(_raw)


def load_sidep_noage(dataset_id: str) -> pd.DataFrame:
    """Fallback: load non-age-stratified daily dep file (no 'cage' in title)."""
    candidates = _get_sidep_candidates(dataset_id)
    if not candidates:
        raise ValueError(f"No fallback CSV found in {dataset_id}")

    candidates.sort(key=lambda r: r.get("last_modified", ""), reverse=True)
    daily_noage = [r for r in candidates
                   if "jour" in r["title"].lower() and "cage" not in r["title"].lower()]
    daily_any   = [r for r in candidates if "jour" in r["title"].lower()]
    best = (daily_noage or daily_any or candidates)
    best.sort(key=lambda r: r.get("last_modified", ""), reverse=True)

    print(f"  ↩️  Fallback loading: {best[0]['title']}")
    _raw = pd.read_csv(best[0]["url"], sep=";", dtype=str, low_memory=False)
    return _fix_comma_decimals(_raw)


def apply_age_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to all-ages rows.
    Priority: cl_age90 / clage / cl_age == '0'
    Fallback: cage10 == '00'
    Last resort: return full DataFrame unfiltered.
    """
    primary_col = next(
        (c for c in df.columns if c.lower() in ["cl_age90", "clage", "cl_age"]), None
    )
    if primary_col:
        filtered = df[df[primary_col].astype(str).str.strip() == "0"].copy()
        if len(filtered) > 0:
            print(f"  ℹ️  Age filter on '{primary_col}' == '0': {len(filtered):,} rows kept")
            return filtered
        cage10_col = next((c for c in df.columns if c.lower() == "cage10"), None)
        if cage10_col:
            filtered2 = df[df[cage10_col].astype(str).str.strip() == "00"].copy()
            if len(filtered2) > 0:
                print(f"  ℹ️  Age filter on '{cage10_col}' == '00': {len(filtered2):,} rows kept")
                return filtered2
        print("  ⚠️  Age filter returned 0 rows — keeping all rows unfiltered")
        return df
    print("  ℹ️  No age column found — keeping all rows")
    return df


def _normalize_dep_col(df: pd.DataFrame) -> pd.DataFrame:
    """Detect, normalize and rename the département code column to 'dep'."""
    col = next(
        (c for c in df.columns if c.lower() in ["dep", "dep_code", "departement"]), None
    )
    if col:
        df[col] = normalize_dep(df[col])
        df = df.rename(columns={col: "dep"})
    return df


def load_finess_dataset() -> pd.DataFrame:
    """
    Load FINESS geolocalized CSV from data.gouv.fr.
    Tries multiple encodings; handles accented French resource names.
    """
    finess_id = "finess-extraction-du-fichier-des-etablissements"
    resp = requests.get(
        f"https://www.data.gouv.fr/api/1/datasets/{finess_id}/", timeout=15
    )
    resp.raise_for_status()
    resources = resp.json().get("resources", [])

    print("  ℹ️  FINESS resources available:")
    for r in resources:
        print(f"       • [{r.get('format','?'):6}] {r['title']}")

    geo_kw = [
        "géolocalis", "geolocalis", "géoloc", "geoloc",
        "géographi",  "geographi",  "coordonn", "gps",
        "etalab",     "cs1100507",
    ]
    csvs = [
        r for r in resources
        if r.get("format", "").lower() in ["csv", ""] or r["url"].endswith(".csv")
    ]
    geo_csvs   = [r for r in csvs if any(kw in r["title"].lower() for kw in geo_kw)]
    candidates = geo_csvs if geo_csvs else csvs
    candidates.sort(key=lambda r: r.get("last_modified", ""), reverse=True)

    if not candidates:
        raise ValueError("No CSV resource found in FINESS dataset")

    url = candidates[0]["url"]
    print(f"  ✓ Loading: {candidates[0]['title']}")

    for enc in ["utf-8", "latin1", "utf-8-sig", "cp1252"]:
        for sep in [";", "|", ",", "\t"]:
            try:
                _df = pd.read_csv(url, sep=sep, encoding=enc, dtype=str, low_memory=False)
                if len(_df.columns) > 3:
                    print(f"  ℹ️  FINESS parsed with sep={sep!r} enc={enc}")
                    return _df
            except (UnicodeDecodeError, Exception):  # noqa: BLE001
                continue
    raise ValueError("Could not decode FINESS file with any encoding or separator")


def load_covidtracker_variants() -> pd.DataFrame:
    """
    Try to load département-level variant data from Guillaume Rozier's GitHub repo.
    Returns empty DataFrame if no file is found (non-blocking).
    """
    variant_urls = [
        "https://raw.githubusercontent.com/rozierguillaume/covid-19/master/data/france/variants.csv",
        "https://raw.githubusercontent.com/rozierguillaume/covid-19/master/data/france/donnees-variants.csv",
        "https://raw.githubusercontent.com/rozierguillaume/covid-19/master/data/france/covid-variants.csv",
    ]
    for url in variant_urls:
        try:
            df = pd.read_csv(url, sep=";", dtype=str, low_memory=False)
            if len(df.columns) == 1:
                df = pd.read_csv(url, sep=",", dtype=str, low_memory=False)
            return df
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame()


def _fix_comma_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace French-locale comma decimal separators (e.g. '129,00') with dots."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(200)
            if sample.str.match(r"^-?\d+,\d+$").any():
                df[col] = df[col].str.replace(",", ".", regex=False)
    return df


# ============================================================
# SOURCE 1 — Geographic Boundaries
# ============================================================

print("📦 [1] Geographic boundaries...")

gdf_dept = gpd.read_file(
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/"
    "departements-version-simplifiee.geojson"
).to_crs("EPSG:4326")

gdf_reg = gpd.read_file(
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/"
    "regions-version-simplifiee.geojson"
).to_crs("EPSG:4326")

gdf_dept["code"] = normalize_dep(gdf_dept["code"])
gdf_reg["code"]  = normalize_dep(gdf_reg["code"])

corsica = gdf_dept[gdf_dept["code"].isin(["2A", "2B"])][["code", "nom"]].values.tolist()
print(f"  ✓ Corsica: {corsica}")
print(f"  ✓ Départements: {len(gdf_dept)} | Régions: {len(gdf_reg)}")

gdf_dept.to_file("data/raw/departements.geojson", driver="GeoJSON")
gdf_reg.to_file("data/raw/regions.geojson", driver="GeoJSON")


# ============================================================
# SOURCE 2 — Kalisio COVID GeoJSON (opendatasoft mirror)
# day_hosp / day_intcare = direct daily new hospitalization counts
# ============================================================

print("\n🗺️  [2] Kalisio COVID GeoJSON (opendatasoft)...")

try:  # noqa: W0718
    gdf_kalisio = gpd.read_file(
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "donnees-hospitalieres-covid-19-dep-france/exports/geojson"
        "?limit=-1&timezone=UTC&use_labels=false&epsg=4326"
    )
    gdf_kalisio.to_file("data/raw/kalisio_dept.geojson", driver="GeoJSON")
    print(f"  ✓ {len(gdf_kalisio):,} rows | Columns: {list(gdf_kalisio.columns[:6])}")

    # Supplement metropolitan GeoJSON with DOM polygons from Kalisio
    try:
        _dom_codes = {"971", "972", "973", "974", "976"}
        _dom_names = {"971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
                      "974": "La Réunion", "976": "Mayotte"}
        _dep_norm = normalize_dep(gdf_kalisio["dep_code"].astype(str))
        _gdf_dom = (
            gdf_kalisio.assign(_dep_norm=_dep_norm)
            .loc[lambda d: d["_dep_norm"].isin(_dom_codes)]
            .drop_duplicates("_dep_norm")[["_dep_norm", "geometry"]]
            .rename(columns={"_dep_norm": "code"})
            .to_crs("EPSG:4326")
        )
        _gdf_dom["nom"] = _gdf_dom["code"].map(_dom_names)
        import geopandas as _gpd
        gdf_dept = _gpd.GeoDataFrame(
            pd.concat([gdf_dept, _gdf_dom[["code", "nom", "geometry"]]], ignore_index=True),
            crs="EPSG:4326",
        )
        gdf_dept.to_file("data/raw/departements.geojson", driver="GeoJSON")
        print(f"  ✓ GeoJSON complété: {len(gdf_dept)} depts (métropole + DOM)")
    except Exception as _err_dom:  # noqa: W0718
        print(f"  ⚠️  DOM supplement: {_err_dom}")

except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Kalisio: {err}")


# ============================================================
# SOURCE 3 — Hospitalization Stock (SPF)
# hosp / rea / dc / rad — cumulative stock; sexe==0 = all sexes
# ============================================================

print("\n🏥 [3] Hospitalization stock (SPF)...")

df_hosp = pd.read_csv(
    "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7",
    sep=";", parse_dates=["jour"], dtype={"dep": str}
)
df_hosp = df_hosp[df_hosp["sexe"] == 0].copy()
df_hosp["dep"] = normalize_dep(df_hosp["dep"])

print(f"  ✓ {len(df_hosp):,} rows | {df_hosp['jour'].min().date()} → {df_hosp['jour'].max().date()}")
corsica_hosp = sorted(df_hosp[df_hosp["dep"].isin(["2A", "2B", "20"])]["dep"].unique())
print(f"  ✓ Corsica codes present: {corsica_hosp}")
df_hosp.to_csv("data/raw/hospitalisations.csv", index=False)


# SOURCE 3b — daily incidence: covered by Sources 2 / 3 (.diff()) / 10
print("\n📈 [3b] Daily incidence → covered by Sources 2, 3 and 10 ✓")


# ============================================================
# SOURCE 3c — Hospitalization by Age Group (region-level)
# ============================================================

print("\n👥 [3c] Hospitalization by age group (region-level)...")

try:  # noqa: W0718
    df_hosp_age = pd.read_csv(
        "https://www.data.gouv.fr/fr/datasets/r/08c18e08-6780-452d-9b8c-ae244ad529b3",
        sep=";", dtype=str, low_memory=False
    )
    hosp_age_date_col = "jour" if "jour" in df_hosp_age.columns else df_hosp_age.columns[0]
    df_hosp_age[hosp_age_date_col] = pd.to_datetime(
        df_hosp_age[hosp_age_date_col], errors="coerce", format="%Y-%m-%d"
    )
    print(f"  ✓ {len(df_hosp_age):,} rows | Columns: {list(df_hosp_age.columns)}")
    print("  ℹ️  Region-level (col 'reg') — no département breakdown available")
    df_hosp_age.to_csv("data/raw/hospitalisations_age_region.csv", index=False)
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Hosp age region: {err}")


# ============================================================
# SOURCE 4 — OpenCovid19-FR (early 2020 gap filler)
# Coverage: 2020-01-24 → 2021-08-12
# ============================================================

print("\n🔓 [4] OpenCovid19-FR (early 2020 gap filler)...")

try:  # noqa: W0718
    df_oc = pd.read_csv(
        "https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/chiffres-cles.csv",
        parse_dates=["date"], dtype={"maille_code": str}, low_memory=False
    )
    df_oc_dept = df_oc[df_oc["granularite"] == "departement"].copy()
    df_oc_dept["dep"] = normalize_dep(df_oc_dept["maille_code"])
    oc_min = pd.to_datetime(df_oc_dept["date"], errors="coerce").min()
    oc_max = pd.to_datetime(df_oc_dept["date"], errors="coerce").max()
    print(f"  ✓ {len(df_oc_dept):,} dept rows | {oc_min.date()} → {oc_max.date()}")
    df_oc_dept.to_csv("data/raw/opencovid19_departements.csv", index=False)
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  OpenCovid19-FR: {err}")


# ============================================================
# SOURCE 5 — Vaccination by Département (VAC-SI)
# couv_complet = % fully vaccinated | clage_vacsi == 0 = all ages
# ============================================================

print("\n💉 [5] Vaccination by département (VAC-SI)...")

df_vaccin = pd.DataFrame()
try:  # noqa: W0718
    url_vaccin = fetch_latest_datagouv_url(
        dataset_id="donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1",
        resource_name_contains="vacsi-dep-"
    )
    print(f"  ✓ Latest URL: {url_vaccin}")
    df_vaccin = pd.read_csv(
        url_vaccin, sep=";", parse_dates=["jour"], dtype={"dep": str}, low_memory=False
    )
    df_vaccin["dep"] = normalize_dep(df_vaccin["dep"])
    if "clage_vacsi" in df_vaccin.columns:
        df_vaccin = df_vaccin[df_vaccin["clage_vacsi"] == 0].copy()
    print(f"  ✓ {len(df_vaccin):,} rows | {df_vaccin['jour'].min().date()} → {df_vaccin['jour'].max().date()}")
    df_vaccin.to_csv("data/vaccines/vaccination_dept.csv", index=False)
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Vaccination: {err}")


# ============================================================
# SOURCE 6 — Vaccination by Age + Département
# clage_vacsi: 0=tous, 4=<5, 9=5-9, 11=10-11 … 80=80+
# ============================================================

print("\n👶 [6] Vaccination by age + département...")

try:  # noqa: W0718
    url_vaccin_age = fetch_latest_datagouv_url(
        dataset_id="donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1",
        resource_name_contains="vacsi-a-dep-"
    )
    df_vaccin_age = pd.read_csv(
        url_vaccin_age, sep=";", parse_dates=["jour"], dtype={"dep": str}, low_memory=False
    )
    df_vaccin_age["dep"] = normalize_dep(df_vaccin_age["dep"])
    print(f"  ✓ {len(df_vaccin_age):,} rows | Ages: {sorted(df_vaccin_age['clage_vacsi'].unique())}")
    df_vaccin_age.to_csv("data/vaccines/vaccination_age_dept.csv", index=False)
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Vaccination by age: {err}")


# ============================================================
# SOURCE 7 — INSEE Population by Département (hardcoded 2020 census)
# ============================================================

print("\n👪 [7] INSEE population by département (hardcoded 2020)...")

_POP_DATA = {
    "01": 643350,  "02": 527261,  "03": 337988,  "04": 163915,  "05": 141284,
    "06": 1094068, "07": 330854,  "08": 273534,  "09": 154157,  "10": 310020,
    "11": 370260,  "12": 279206,  "13": 2043110, "14": 694002,  "15": 144692,
    "16": 352335,  "17": 644391,  "18": 304256,  "19": 241464,
    "2A": 157249,  "2B": 349465,
    "21": 534670,  "22": 598814,  "23": 111664,  "24": 415574,  "25": 542085,
    "26": 518361,  "27": 601843,  "28": 433233,  "29": 909028,  "30": 757620,
    "31": 1389767, "32": 191091,  "33": 1623111, "34": 1144892, "35": 1069657,
    "36": 222232,  "37": 616327,  "38": 1273068, "39": 263808,  "40": 413936,
    "41": 336367,  "42": 762941,  "43": 227983,  "44": 1430682, "45": 685346,
    "46": 174219,  "47": 436484,  "48": 66197,   "49": 824000,  "50": 502924,
    "51": 571534,  "52": 184130,  "53": 306977,  "54": 733481,  "55": 186703,
    "56": 753030,  "57": 1043522, "58": 212105,  "59": 2604361, "60": 824503,
    "61": 283372,  "62": 1477260, "63": 654606,  "64": 682372,  "65": 232964,
    "66": 477784,  "67": 1125559, "68": 770087,  "69": 1843270, "70": 240164,
    "71": 555413,  "72": 566506,  "73": 430692,  "74": 418769,  "75": 2187526,
    "76": 1254767, "77": 1444464, "78": 1437487, "79": 375952,  "80": 384922,
    "81": 387890,  "82": 261466,  "83": 1061390, "84": 550890,  "85": 681671,
    "86": 442370,  "87": 375770,  "88": 473946,  "89": 353000,  "90": 164302,
    "91": 1296130, "92": 1609306, "93": 1623111, "94": 1387926, "95": 1228618,
    "971": 413850, "972": 376078, "973": 290691, "974": 882895, "976": 327897,
}

df_pop = pd.DataFrame(list(_POP_DATA.items()), columns=["dep", "population"])
assert len(df_pop) == 101, f"Expected 101 rows, got {len(df_pop)}"
assert df_pop["population"].isna().sum() == 0, "Found NaN populations"

print(f"  ✓ {len(df_pop)} départements (19 + 2A/2B + 75 metro + 5 DOM)")
print(f"  ✓ Total: {df_pop['population'].sum():,.0f}")
print(f"  ✓ Largest:  {df_pop.loc[df_pop['population'].idxmax(), 'dep']} "
      f"({df_pop['population'].max():,.0f})")
print(f"  ✓ Smallest: {df_pop.loc[df_pop['population'].idxmin(), 'dep']} "
      f"({df_pop['population'].min():,.0f})")
df_pop.to_csv("data/raw/population_departements.csv", index=False)


# ============================================================
# SOURCE 8 — SI-DEP Old Format (2020-05-13 → 2022-05-17)
# Weekly aggregates only: dep, semaine_glissante, P (positifs), T (testés)
# ============================================================

print("\n🔬 [8] SI-DEP lab tests — old format (2020–2022)...")

df_sidep_old = pd.DataFrame()
try:  # noqa: W0718
    df_sidep_old = load_sidep_dataset(
        "old-donnees-relatives-aux-resultats-des-tests-virologiques-covid-19",
        "SI-DEP old"
    )
    df_sidep_old = _normalize_dep_col(df_sidep_old)

    sidep_old_age = next(
        (c for c in df_sidep_old.columns
         if c.lower() in ["cl_age90", "cage10", "clage", "cl_age"]), None
    )
    if sidep_old_age:
        df_sidep_old = df_sidep_old[
            df_sidep_old[sidep_old_age].astype(str).str.strip() == "0"
        ].copy()
        print(f"  ℹ️  Age filter on '{sidep_old_age}' == '0': {len(df_sidep_old):,} rows kept")

    if "P" in df_sidep_old.columns and "T" in df_sidep_old.columns:
        df_sidep_old["P"] = pd.to_numeric(df_sidep_old["P"], errors="coerce")
        df_sidep_old["T"] = pd.to_numeric(df_sidep_old["T"], errors="coerce")
        df_sidep_old["tx_pos"] = (df_sidep_old["P"] / df_sidep_old["T"] * 100).round(2)
        print(f"  ✓ tx_pos: min={df_sidep_old['tx_pos'].min():.1f}% "
              f"max={df_sidep_old['tx_pos'].max():.1f}%")

    df_sidep_old.to_csv("data/labs/sidep_old.csv", index=False)
    print(f"  ✓ {len(df_sidep_old):,} rows | Columns: {list(df_sidep_old.columns[:8])}")

except Exception as err:  # noqa: W0718
    print(f"  ⚠️  SI-DEP old: {err}")
    try:  # noqa: W0718
        debug_dataset_resources(
            "old-donnees-relatives-aux-resultats-des-tests-virologiques-covid-19"
        )
    except Exception as err2:  # noqa: W0718
        print(f"  ⚠️  Debug failed: {err2}")


# ============================================================
# SOURCE 9 — SI-DEP New Formats (2022-05-18 → end)
# cage10 uses label codes ("00","09"…) — not the "0" sentinel
# Priority: cl_age90 → cage10=="00" → no-age fallback file
# ============================================================

print("\n🔬 [9] SI-DEP lab tests — new formats (2022–2023)...")

df_sidep_new = pd.DataFrame()

_SIDEP_NEW_DATASETS = [
    ("donnees-de-laboratoires-pour-le-depistage-a-compter-du-28-09-2023-si-dep", "2023-09 → end"),
    ("donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep", "2022-05 → 2023-09"),
]

for _sidep_id, _period in _SIDEP_NEW_DATASETS:
    try:  # noqa: W0718
        _df = load_sidep_dataset(_sidep_id, _period)
        _df = _normalize_dep_col(_df)
        _df = apply_age_filter(_df)

        if len(_df) == 0:
            print("  ⚠️  Still 0 rows — trying non-age-grouped fallback file...")
            _df = load_sidep_noage(_sidep_id)
            _df = _normalize_dep_col(_df)
            _df = apply_age_filter(_df)

        df_sidep_new = (
            pd.concat([df_sidep_new, _df], ignore_index=True)
            if not df_sidep_new.empty else _df
        )
        print(f"  ✓ {_period}: {len(_df):,} rows | Columns: {list(_df.columns[:6])}")

    except Exception as err:  # noqa: W0718
        print(f"  ⚠️  {_period}: {err}")
        try:  # noqa: W0718
            debug_dataset_resources(_sidep_id)
        except Exception as err2:  # noqa: W0718
            print(f"  ⚠️  Debug failed: {err2}")

if not df_sidep_new.empty:
    # Derive jour from sg (sliding window "YYYY-MM-DD-YYYY-MM-DD") when jour is NaN
    if "jour" in df_sidep_new.columns and "sg" in df_sidep_new.columns:
        _null_jour = df_sidep_new["jour"].isna()
        if _null_jour.any():
            df_sidep_new.loc[_null_jour, "jour"] = df_sidep_new.loc[_null_jour, "sg"].str[:10]
            print(f"  ℹ️  {_null_jour.sum():,} lignes: 'jour' complété depuis 'sg'")
    df_sidep_new.to_csv("data/labs/sidep_new.csv", index=False)
    print(f"  ✓ Total SI-DEP new: {len(df_sidep_new):,} rows saved")
else:
    print("  ⚠️  All SI-DEP new periods failed — non-critical for Phase 2")


# ============================================================
# SOURCE 10 — Indicateurs de Suivi Épidémique
# incidence_rate = Ti/100k | r0 = Rt | sae_rate = ICU tension %
# ============================================================

print("\n📉 [10] Indicateurs de suivi épidémique (tx_incid, R0, TO)...")

df_indicateurs = pd.DataFrame()
try:  # noqa: W0718
    df_indicateurs = pd.read_csv(
        "https://data.ampmetropole.fr/api/explore/v2.1/catalog/datasets/"
        "fed-indicateurs-de-suivi-de-lepidemie-de-covid-19-en-france/exports/csv"
        "?limit=-1&timezone=UTC&use_labels=false",
        sep=";", dtype=str, low_memory=False
    )
    indic_dep = next(
        (c for c in df_indicateurs.columns
         if c.lower() in ["dep", "dep_code", "departement", "code_dep"]), None
    )
    if indic_dep:
        df_indicateurs[indic_dep] = normalize_dep(df_indicateurs[indic_dep])
        df_indicateurs = df_indicateurs.rename(columns={indic_dep: "dep"})

    indic_date = next(
        (c for c in df_indicateurs.columns
         if c.lower() in ["jour", "date", "extract_date"]), None
    )
    if indic_date:
        df_indicateurs[indic_date] = pd.to_datetime(df_indicateurs[indic_date], errors="coerce")
        df_indicateurs = df_indicateurs.rename(columns={indic_date: "jour"})

    df_indicateurs.to_csv("data/labs/indicateurs_suivi.csv", index=False)
    print(f"  ✓ {len(df_indicateurs):,} rows | Columns: {list(df_indicateurs.columns[:8])}")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Indicateurs: {err}")


# ============================================================
# SOURCE 11 — Vaccination Centers GPS (opendatasoft)
# ============================================================

print("\n🏥 [11] Vaccination centers GPS...")

try:  # noqa: W0718
    df_centres_vac = pd.read_csv(
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "covid19-france-lieu-vaccination/exports/csv"
        "?limit=-1&timezone=UTC&use_labels=false",
        sep=";", dtype=str, low_memory=False
    )
    centres_lat = next((c for c in df_centres_vac.columns if "lat" in c.lower()), None)
    centres_lon = next((c for c in df_centres_vac.columns if "lon" in c.lower()), None)
    centres_dep = next((c for c in df_centres_vac.columns if "dep" in c.lower()), None)

    # Fallback: parse geo_point_2d ("lat, lon" string) when no dedicated lat/lon columns
    if not centres_lat:
        _geo_col = next((c for c in df_centres_vac.columns if "geo_point" in c.lower()), None)
        if _geo_col:
            _coords = df_centres_vac[_geo_col].str.split(",", expand=True)
            df_centres_vac["latitude"]  = pd.to_numeric(_coords[0], errors="coerce")
            df_centres_vac["longitude"] = pd.to_numeric(_coords[1].str.strip(), errors="coerce")
            centres_lat, centres_lon = "latitude", "longitude"
            print(f"  ℹ️  GPS extrait depuis '{_geo_col}'")

    if centres_lat and centres_lon:
        df_centres_vac[centres_lat] = pd.to_numeric(df_centres_vac[centres_lat], errors="coerce")
        df_centres_vac[centres_lon] = pd.to_numeric(df_centres_vac[centres_lon], errors="coerce")

    df_centres_vac.to_csv("data/locations/centres_vaccination.csv", index=False)

    if centres_dep:
        df_centres_vac[centres_dep] = normalize_dep(df_centres_vac[centres_dep])
        centres_per_dep = (
            df_centres_vac.groupby(centres_dep)
            .size().reset_index(name="nb_centres_vac")
        )
        centres_per_dep.columns = ["dep", "nb_centres_vac"]
        centres_per_dep.to_csv("data/locations/centres_vac_par_dep.csv", index=False)
        print(f"  ✓ {len(df_centres_vac):,} centres | {len(centres_per_dep)} départements")
    else:
        print(f"  ✓ {len(df_centres_vac):,} centres loaded (no dept column found)")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Centres vaccination: {err}")


# ============================================================
# SOURCE 12 — Vaccination by Location Type (Ameli — national)
# 1=Centre, 2=Cabinet/Officine, 3=EHPAD, 4=Domicile, 5=Entreprise
# ============================================================

print("\n💊 [12] Vaccination par lieu (Ameli — national)...")

try:  # noqa: W0718
    df_lieu_vac = pd.read_csv(
        "https://datavaccin-covid.ameli.fr/api/explore/v2.1/catalog/datasets/"
        "donnees-vaccination-lieu-de-vaccination/exports/csv"
        "?limit=-1&timezone=UTC&use_labels=false",
        sep=";", dtype=str, low_memory=False
    )
    df_lieu_vac.to_csv("data/vaccines/vaccination_lieu.csv", index=False)
    print(f"  ✓ {len(df_lieu_vac):,} rows | Columns: {list(df_lieu_vac.columns)}")
    if "categorie" in df_lieu_vac.columns:
        print(f"  ✓ Location types: {df_lieu_vac['categorie'].unique().tolist()}")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Lieu vaccination: {err}")


# ============================================================
# SOURCE 13 — Vaccination by Injector Type (Ameli — national)
# médecin libéral / pharmacien / infirmier / centre / EHPAD
# ============================================================

print("\n💉 [13] Vaccination par type d'injecteur (Ameli — national)...")

try:  # noqa: W0718
    df_injecteur = pd.read_csv(
        "https://datavaccin-covid.ameli.fr/api/explore/v2.1/catalog/datasets/"
        "donnees-de-vaccination-type-dinjecteur/exports/csv"
        "?limit=-1&timezone=UTC&use_labels=false",
        sep=";", dtype=str, low_memory=False
    )
    df_injecteur.to_csv("data/vaccines/vaccination_injecteur.csv", index=False)
    print(f"  ✓ {len(df_injecteur):,} rows | Columns: {list(df_injecteur.columns)}")
    print("  ℹ️  National level only — no département breakdown available")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Injecteur: {err}")


# ============================================================
# SOURCE 14 — Vaccination by Vaccine Brand (VAC-SI vacsi-v-dep)
# vaccin: 0=tous, 1=Pfizer, 2=Moderna, 3=AstraZeneca, 4=Janssen, 5=Novavax
# ============================================================

print("\n🧪 [14] Vaccination par marque de vaccin (VAC-SI)...")

try:  # noqa: W0718
    url_marque = fetch_latest_datagouv_url(
        dataset_id="donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1",
        resource_name_contains="vacsi-v-dep"
    )
    df_marque = pd.read_csv(url_marque, sep=";", dtype={"dep": str}, low_memory=False)
    df_marque["dep"] = normalize_dep(df_marque["dep"])
    df_marque.to_csv("data/vaccines/vaccination_marque.csv", index=False)
    vac_col = next(
        (c for c in df_marque.columns if c in ["vaccin", "vac_lib", "vaccin_lib"]), None
    )
    print(f"  ✓ {len(df_marque):,} rows | Columns: {list(df_marque.columns)}")
    if vac_col:
        print(f"  ✓ Brand codes: {sorted(df_marque[vac_col].unique())}")
        print("  ℹ️  0=tous, 1=Pfizer, 2=Moderna, 3=AstraZeneca, 4=Janssen, 5=Novavax")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Marque: {err}")


# ============================================================
# SOURCE 15 — COVID Variants (DREES national appariement)
# vac_statut × nb_pcr × hosp counts — national level only
# ============================================================

print("\n🧫 [15] Variants COVID (DREES — national)...")

df_variants = pd.DataFrame()
try:  # noqa: W0718
    df_variants = pd.read_csv(
        "https://data.smartidf.services/api/explore/v2.1/catalog/datasets/"
        "covid-19-resultats-issus-des-appariements-entre-si-vic-si-dep-et-vac-si/exports/csv"
        "?limit=-1&timezone=UTC&use_labels=false",
        sep=";", dtype=str, low_memory=False
    )
    df_variants.to_csv("data/labs/variants.csv", index=False)
    print(f"  ✓ {len(df_variants):,} rows | Columns: {list(df_variants.columns[:8])}")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Variants: {err}")


# ============================================================
# SOURCE 15b — CovidTracker variant data (Rozier — optional)
# Potential département-level breakdown not available in DREES
# ============================================================

print("\n🧫 [15b] CovidTracker variant data (Rozier — optional)...")

df_variants_dept = load_covidtracker_variants()
if not df_variants_dept.empty:
    df_variants_dept = _normalize_dep_col(df_variants_dept)
    if "dep" in df_variants_dept.columns:
        df_variants_dept.to_csv("data/labs/variants_dept.csv", index=False)
        print(f"  ✓ {len(df_variants_dept):,} rows | Columns: {list(df_variants_dept.columns[:8])}")
    else:
        print("  ℹ️  No département column found — national only, skipping")
else:
    print("  ℹ️  No CovidTracker variant file found — Source 15 (DREES national) sufficient")


# ============================================================
# SOURCE 16 — FINESS Establishments GPS (Atlasanté)
# Hospitals, labs, pharmacies, clinics — GPS coordinates
# ============================================================

print("\n📍 [16] FINESS establishments GPS (Atlasanté)...")

try:  # noqa: W0718
    df_finess = load_finess_dataset()
    df_finess.columns = [c.lower().strip() for c in df_finess.columns]
    print(f"  ✓ Raw columns: {list(df_finess.columns[:10])}")

    finess_lat = next((c for c in df_finess.columns if "lat" in c), None)
    finess_lon = next((c for c in df_finess.columns if "lon" in c), None)
    finess_dep = next((c for c in df_finess.columns if "dep" in c), None)
    finess_id  = next((c for c in df_finess.columns if "finess" in c or "nofiness" in c), None)
    finess_cat = next((c for c in df_finess.columns if "categ" in c), None)
    finess_nm  = next((c for c in df_finess.columns if c in ["rs", "raison", "libelle"]), None)

    keep = [c for c in [finess_id, finess_nm, finess_dep, finess_lat, finess_lon, finess_cat] if c]
    df_finess = df_finess[keep].copy()

    if finess_lat and finess_lon:
        df_finess[finess_lat] = pd.to_numeric(df_finess[finess_lat], errors="coerce")
        df_finess[finess_lon] = pd.to_numeric(df_finess[finess_lon], errors="coerce")
        df_finess = df_finess.dropna(subset=[finess_lat, finess_lon])
        df_finess = df_finess.rename(columns={finess_lat: "latitude", finess_lon: "longitude"})
    if finess_dep:
        df_finess[finess_dep] = normalize_dep(df_finess[finess_dep])
        df_finess = df_finess.rename(columns={finess_dep: "dep"})

    df_finess.to_csv("data/raw/finess_coords.csv", index=False)
    print(f"  ✓ {len(df_finess):,} establishments with GPS")
    if finess_cat and finess_cat in df_finess.columns:
        print(f"  ✓ Top categories: {df_finess[finess_cat].value_counts().head(5).to_dict()}")

except Exception as err:  # noqa: W0718
    df_finess = pd.DataFrame()
    print(f"  ⚠️  FINESS: {err} — skipped, non-critical for Phase 2")


# ============================================================
# SOURCE 17 — SI-VIC via DREES (hosp by vaccination status)
# Regional: hosp / ICU / deaths split vaccinated vs unvaccinated
# ============================================================

print("\n🏢 [17] SI-VIC via DREES (hosp par statut vaccinal — régional)...")

df_sivic = pd.DataFrame()
try:  # noqa: W0718
    df_sivic = pd.read_csv(
        "https://data.smartidf.services/api/explore/v2.1/catalog/datasets/"
        "covid-19-resultats-regionaux-issus-des-appariements-entre-si-vic-si-dep-et-vac-s/exports/csv"
        "?limit=-1&timezone=UTC&use_labels=false",
        sep=";", dtype=str, low_memory=False
    )
    df_sivic.to_csv("data/raw/sivic_regional.csv", index=False)
    print(f"  ✓ {len(df_sivic):,} rows | Columns: {list(df_sivic.columns[:8])}")
    print("  ℹ️  Regional — hosp/ICU split vaccinated vs unvaccinated")
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  SI-VIC DREES: {err} — non-critical for Phase 2")


# ============================================================
# POST-PROCESSING A — Normalization: Rates per 100K
# ============================================================

print("\n📊 Computing normalized rates per 100K...")

try:  # noqa: W0718
    df_latest = df_hosp.sort_values("jour").groupby("dep").last().reset_index()
    df_latest = df_latest.merge(df_pop[["dep", "population"]], on="dep", how="left")
    for _col in ["hosp", "rea", "dc", "rad"]:
        if _col in df_latest.columns:
            df_latest[f"{_col}_rate_100k"] = (
                pd.to_numeric(df_latest[_col], errors="coerce")
                / df_latest["population"] * 100_000
            ).round(2)
    rate_cols = [c for c in df_latest.columns if "100k" in c]
    print(f"  ✓ Rate columns computed: {rate_cols}")
    df_latest.to_csv("data/processed/hospitalisations_normalized.csv", index=False)
except Exception as err:  # noqa: W0718
    print(f"  ⚠️  Normalization: {err}")


# ============================================================
# POST-PROCESSING B — Vaccine 14-Day Lag
# couv_complet_lag14: vaccination coverage 14 days prior (immune delay)
# ============================================================

if not df_vaccin.empty and "couv_complet" in df_vaccin.columns:
    print("\n⏱️  Pre-computing 14-day vaccination lag...")
    df_vaccin_lag = df_vaccin.copy().sort_values(["dep", "jour"])
    df_vaccin_lag["couv_complet_lag14"] = (
        df_vaccin_lag.groupby("dep")["couv_complet"].shift(14)
    )
    df_vaccin_lag.to_csv("data/vaccines/vaccination_lag14.csv", index=False)
    print("  ✓ couv_complet_lag14 added")
else:
    print("\n⏱️  Vaccination lag skipped — df_vaccin empty or missing couv_complet")


# ============================================================
# MERGE READINESS CHECK
# ============================================================

print("\n" + "=" * 60)
print("✅ MERGE READINESS CHECK")
print("=" * 60)

geo_codes  = set(gdf_dept["code"].unique())
hosp_codes = set(df_hosp["dep"].unique())
pop_codes  = set(df_pop["dep"].unique())

print(f"  GeoJSON dept codes : {len(geo_codes)}")
print(f"  Hosp CSV dep codes : {len(hosp_codes)}")
print(f"  Pop  CSV dep codes : {len(pop_codes)}")
print(f"  GeoJSON ∩ Hosp     : {len(geo_codes & hosp_codes)}")
print(f"  GeoJSON ∩ Pop      : {len(geo_codes & pop_codes)}")

unmatched_geo_hosp = geo_codes - hosp_codes
unmatched_geo_pop  = geo_codes - pop_codes

if unmatched_geo_hosp:
    print(f"  ⚠️  GeoJSON codes not in Hosp: {sorted(unmatched_geo_hosp)}")
else:
    print("  ✓ GeoJSON ↔ Hosp: all codes match")

if unmatched_geo_pop:
    print(f"  ⚠️  GeoJSON codes not in Pop: {sorted(unmatched_geo_pop)}")
else:
    print("  ✓ GeoJSON ↔ Pop:  all codes match")


# ============================================================
# DATA FOLDER SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("📁 FILES SAVED")
print("=" * 60)

total_kb = 0
for _folder in ["data/raw", "data/processed", "data/vaccines", "data/labs", "data/locations"]:
    _files = sorted(os.listdir(_folder)) if os.path.exists(_folder) else []
    print(f"\n  📂 {_folder}/")
    for _f in _files:
        _path = os.path.join(_folder, _f)
        _kb = os.path.getsize(_path) / 1024
        total_kb += _kb
        print(f"     ├─ {_f:<52} {_kb:>9.1f} KB")

print(f"\n  💾 Total size on disk: {total_kb / 1024:.1f} MB")
print("\n🚀 Phase 1 complete — ready for Phase 2: Preprocessing")