# ============================================================
# CARTE DYNAMIQUE — Phase 5 v2: Carte Interactive (Folium)
# ============================================================
#
# INPUT (produits par les phases précédentes) :
#   - data/processed/master_dept_geo.geojson   → polygones + métriques (Phase 2)
#   - data/processed/master_dept_latest.csv    → snapshot par dept (Phase 2)
#   - data/clustering/global_cluster_labels.csv → clusters (Phase 3)
#   - data/locations/centres_vaccination.csv   → GPS centres (Phase 1)
#   - data/raw/finess_coords.csv               → GPS établissements (Phase 1, optionnel)
#
# OUTPUT :
#   - outputs/carte_covid_france.html
#
# COUCHES DE LA CARTE :
#   ① Choroplèthe — Taux d'hospitalisation /100K   (défaut)
#   ② Choroplèthe — Taux de positivité %
#   ③ Choroplèthe — Couverture vaccinale %
#   ④ Choroplèthe — Létalité hospitalière %
#   ⑤ Choroplèthe — Clusters épidémiques (Phase 3)
#   ⑥ Marqueurs   — Centres de vaccination
#   ⑦ Marqueurs   — Établissements FINESS (si disponible)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap
import branca.colormap as cm

os.makedirs("outputs", exist_ok=True)

# ── Chemins ────────────────────────────────────────────────────
GEO_FILE    = "data/processed/master_dept_geo.geojson"
LATEST_CSV  = "data/processed/master_dept_latest.csv"
CLUSTER_CSV = "data/clustering/global_cluster_labels.csv"
VACC_CSV    = "data/locations/centres_vaccination.csv"
FINESS_CSV  = "data/raw/finess_coords.csv"
OUT_MAP     = "outputs/carte_covid_france.html"

# ── Helpers ────────────────────────────────────────────────────

def normalize_dep(code: str) -> str:
    code = str(code).strip().upper().replace("DEP-", "")
    if code in ("2A", "2B"):
        return code
    try:
        return str(int(float(code))).zfill(2)
    except (ValueError, OverflowError):
        return code.zfill(2)


def safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 2)
    except Exception:
        return None


def fmt(val, suffix="", decimals=1) -> str:
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except Exception:
        return "N/D"


# ── Couleurs clusters ──────────────────────────────────────────
CLUSTER_COLORS = {
    0: "#e74c3c",   # rouge — forte sévérité
    1: "#3498db",   # bleu — diffusion large
    2: "#2ecc71",
    3: "#f39c12",
    -2: "#95a5a6",  # DOM (marker spécial)
}
CLUSTER_LABELS = {
    0: "Cluster 0 — forte sévérité hospitalière",
    1: "Cluster 1 — diffusion large, moindre gravité",
    -2: "DOM",
}

# ── Métriques choroplèthe ──────────────────────────────────────
METRICS = [
    {
        "key":     "hosp_rate",
        "label":   "🏥 Taux d'hospitalisation (/100K)",
        "unit":    "/100K",
        "colors":  ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        "show":    True,
    },
    {
        "key":     "tx_pos",
        "label":   "🔬 Taux de positivité (%)",
        "unit":    "%",
        "colors":  ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        "show":    False,
    },
    {
        "key":     "couv_complet",
        "label":   "💉 Couverture vaccinale (%)",
        "unit":    "%",
        "colors":  ["#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c"],
        "show":    False,
    },
    {
        "key":     "cfr",
        "label":   "💀 Létalité hospitalière (%)",
        "unit":    "%",
        "colors":  ["#fff5f0", "#fca082", "#fb6a4a", "#de2d26", "#a50f15"],
        "show":    False,
    },
]

# ── Colonnes pour les tooltips / popups ────────────────────────
TOOLTIP_FIELDS   = ["dep", "nom", "hosp_rate", "tx_pos", "couv_complet",
                    "cfr", "icu_pressure", "kmeans_cluster"]
TOOLTIP_ALIASES  = ["Dép.", "Département", "Hosp./100K", "Positivité%",
                    "Vaccinés%", "Létalité%", "Pression réa%", "Cluster"]
POPUP_FIELDS     = ["dep", "nom", "hosp_rate", "new_hosp_rate", "tx_pos",
                    "couv_complet", "couv_dose1", "cfr", "icu_pressure",
                    "hosp", "rea", "dc", "kmeans_cluster"]
POPUP_ALIASES    = ["Dép.", "Département", "Hosp./100K", "Nouv.hosp/100K",
                    "Positivité%", "Couv.complète%", "Couv.dose1%",
                    "Létalité hosp.%", "Pression réa%",
                    "Hosp.(stock)", "Réa.(stock)", "Décès cumulés", "Cluster"]


# ============================================================
# STEP 1 — Chargement des données
# ============================================================

print("=" * 60)
print("📂 STEP 1 — Chargement des données")
print("=" * 60)

# GeoJSON Phase 2
if not os.path.exists(GEO_FILE):
    raise FileNotFoundError(
        f"{GEO_FILE} introuvable. Lancez d'abord : python phase2_preprocessing.py"
    )
with open(GEO_FILE, encoding="utf-8") as _f:
    geo = json.load(_f)
print(f"  ✓ GeoJSON : {len(geo['features'])} départements")

# Métriques latest Phase 2
df_latest = pd.read_csv(LATEST_CSV, dtype={"dep": str})
df_latest["dep"] = df_latest["dep"].apply(normalize_dep)
print(f"  ✓ Métriques latest : {len(df_latest)} depts × {len(df_latest.columns)} colonnes")

# Clusters Phase 3 (optionnel)
df_cl = pd.DataFrame()
if os.path.exists(CLUSTER_CSV):
    df_cl = pd.read_csv(CLUSTER_CSV, dtype={"dep": str})
    df_cl["dep"] = df_cl["dep"].apply(normalize_dep)
    print(f"  ✓ Clusters Phase 3 : {df_cl['dep'].nunique()} depts")
else:
    print(f"  ℹ️  {CLUSTER_CSV} absent — clusters désactivés")

# Merge clusters → latest
if not df_cl.empty and "kmeans_cluster" in df_cl.columns:
    df_latest = df_latest.merge(
        df_cl[["dep", "kmeans_cluster"]].drop_duplicates("dep"),
        on="dep", how="left"
    )
    df_latest["kmeans_cluster"] = df_latest["kmeans_cluster"].fillna(-1).astype(int)
elif "kmeans_cluster" not in df_latest.columns:
    df_latest["kmeans_cluster"] = 0

lookup = df_latest.set_index("dep").to_dict("index")

# Centres de vaccination
df_vacc = pd.DataFrame()
if os.path.exists(VACC_CSV):
    _vacc_raw = pd.read_csv(VACC_CSV, dtype=str, low_memory=False)
    # Cherche lat/lon directs, sinon parse geo_point_2d
    _lat = next((c for c in _vacc_raw.columns if c.lower() == "latitude"), None)
    _lon = next((c for c in _vacc_raw.columns if c.lower() == "longitude"), None)
    if not _lat:
        _lat = next((c for c in _vacc_raw.columns if "lat" in c.lower()), None)
    if not _lon:
        _lon = next((c for c in _vacc_raw.columns if "lon" in c.lower()), None)
    if not (_lat and _lon):
        _geo_col = next((c for c in _vacc_raw.columns if "geo_point" in c.lower()), None)
        if _geo_col:
            _coords = _vacc_raw[_geo_col].str.split(",", expand=True)
            _vacc_raw["latitude"]  = pd.to_numeric(_coords[0], errors="coerce")
            _vacc_raw["longitude"] = pd.to_numeric(_coords[1].str.strip(), errors="coerce")
            _lat, _lon = "latitude", "longitude"
    if _lat and _lon:
        df_vacc = _vacc_raw.copy()
        df_vacc["lat"] = pd.to_numeric(df_vacc[_lat], errors="coerce")
        df_vacc["lon"] = pd.to_numeric(df_vacc[_lon], errors="coerce")
        df_vacc = df_vacc.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        print(f"  ✓ Centres vaccination : {len(df_vacc):,} avec GPS")
    else:
        print("  ℹ️  centres_vaccination.csv : pas de GPS exploitable")

# FINESS (optionnel — non bloquant)
df_finess = pd.DataFrame()
if os.path.exists(FINESS_CSV):
    _fin_raw = pd.read_csv(FINESS_CSV, dtype=str, low_memory=False)
    _fin_raw.columns = [c.lower().strip() for c in _fin_raw.columns]
    _flat = next((c for c in _fin_raw.columns if "lat" in c), None)
    _flon = next((c for c in _fin_raw.columns if "lon" in c), None)
    if _flat and _flon and len(_fin_raw.columns) > 2:
        df_finess = _fin_raw.copy()
        df_finess["lat"] = pd.to_numeric(df_finess[_flat], errors="coerce")
        df_finess["lon"] = pd.to_numeric(df_finess[_flon], errors="coerce")
        df_finess = df_finess.dropna(subset=["lat", "lon"]).head(8000)
        print(f"  ✓ FINESS : {len(df_finess):,} établissements avec GPS")
    else:
        print("  ℹ️  FINESS : pas de GPS exploitable — couche ignorée (non bloquant)")


# ============================================================
# STEP 2 — Enrichissement du GeoJSON
# ============================================================

print("\n" + "=" * 60)
print("🔗 STEP 2 — Enrichissement du GeoJSON")
print("=" * 60)

EMBED_COLS = ["hosp_rate", "new_hosp_rate", "tx_pos", "couv_complet", "couv_dose1",
              "cfr", "icu_pressure", "hosp", "rea", "dc", "kmeans_cluster",
              "population", "wave", "is_dom"]

enriched = 0
for feat in geo["features"]:
    dep = normalize_dep(feat["properties"].get("dep", ""))
    feat["properties"]["dep"] = dep
    row = lookup.get(dep, {})
    if row:
        enriched += 1
    for col in EMBED_COLS:
        v = row.get(col)
        if col == "kmeans_cluster":
            feat["properties"][col] = int(v) if v is not None else -1
        else:
            feat["properties"][col] = safe_float(v)
    # Assure nom présent
    if not feat["properties"].get("nom"):
        feat["properties"]["nom"] = dep

print(f"  ✓ {enriched}/{len(geo['features'])} features enrichies avec métriques")


# ============================================================
# STEP 3 — Construction de la carte
# ============================================================

print("\n" + "=" * 60)
print("🗺️  STEP 3 — Construction de la carte")
print("=" * 60)

m = folium.Map(
    location=[46.6, 2.3],
    zoom_start=6,
    tiles="CartoDB positron",
    prefer_canvas=True,
    min_zoom=4,
    max_zoom=13,
)

Fullscreen(position="topleft").add_to(m)
MiniMap(toggle_display=True, position="bottomright").add_to(m)


# ── Choroplèthes ───────────────────────────────────────────────

def make_style_fn(metric_key: str, colormap_fn) -> callable:
    """Retourne une style_function fermée sur la colormap pour un métrique donné."""
    _cache: dict[str, str] = {}
    for feat in geo["features"]:
        dep = feat["properties"]["dep"]
        v   = feat["properties"].get(metric_key)
        _cache[dep] = colormap_fn(v) if v is not None else "#cccccc"

    def _style(feature):
        return {
            "fillColor":   _cache.get(feature["properties"]["dep"], "#cccccc"),
            "fillOpacity": 0.72,
            "color":       "#ffffff",
            "weight":      0.6,
        }
    return _style


for m_cfg in METRICS:
    key = m_cfg["key"]
    vals = [f["properties"][key] for f in geo["features"]
            if f["properties"].get(key) is not None]
    if not vals:
        print(f"  ⚠️  {key} : aucune valeur — couche ignorée")
        continue

    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))

    colormap = cm.LinearColormap(
        colors=m_cfg["colors"],
        vmin=vmin,
        vmax=vmax,
        caption=f"{m_cfg['label']} ({m_cfg['unit']})",
    )
    colormap.add_to(m)

    fg = folium.FeatureGroup(name=m_cfg["label"], show=m_cfg["show"])
    folium.GeoJson(
        geo,
        style_function=make_style_fn(key, colormap),
        highlight_function=lambda _: {"weight": 2, "color": "#333", "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=TOOLTIP_FIELDS,
            aliases=TOOLTIP_ALIASES,
            localize=True,
            sticky=True,
            style="font-size:12px;",
        ),
        popup=folium.GeoJsonPopup(
            fields=POPUP_FIELDS,
            aliases=POPUP_ALIASES,
            max_width=320,
        ),
    ).add_to(fg)
    fg.add_to(m)
    print(f"  ✓ Choroplèthe [{key}]  vmin={vmin:.1f}  vmax={vmax:.1f}")


# ── Couche clusters ────────────────────────────────────────────

def _cluster_style(feature):
    cl = feature["properties"].get("kmeans_cluster", -1)
    return {
        "fillColor":   CLUSTER_COLORS.get(int(cl) if cl is not None else -1, "#aaaaaa"),
        "fillOpacity": 0.65,
        "color":       "#ffffff",
        "weight":      0.6,
    }


fg_cl = folium.FeatureGroup(name="🔵 Clusters épidémiques (Phase 3)", show=False)
folium.GeoJson(
    geo,
    style_function=_cluster_style,
    highlight_function=lambda _: {"weight": 2, "color": "#333", "fillOpacity": 0.85},
    tooltip=folium.GeoJsonTooltip(
        fields=["dep", "nom", "kmeans_cluster", "hosp_rate", "tx_pos", "couv_complet"],
        aliases=["Dép.", "Département", "Cluster", "Hosp./100K", "Positivité%", "Vaccinés%"],
        sticky=True,
    ),
).add_to(fg_cl)
fg_cl.add_to(m)
print("  ✓ Couche clusters épidémiques")

# Légende clusters (HTML fixe)
_legend_items = "".join(
    f'<span style="color:{c};font-size:18px;line-height:1.4">■</span> '
    f'{CLUSTER_LABELS.get(cl, f"Cluster {cl}")}<br>\n'
    for cl, c in CLUSTER_COLORS.items() if cl != -2
)
m.get_root().html.add_child(folium.Element(f"""
<div id="cluster-legend"
     style="position:fixed;bottom:60px;left:10px;z-index:1000;
            background:rgba(255,255,255,0.93);padding:10px 14px;
            border-radius:8px;border:1px solid #ccc;
            font-family:Arial,sans-serif;font-size:12px;line-height:1.6">
  <b>Clusters épidémiques</b><br>
  {_legend_items}
</div>"""))


# ── Centres de vaccination ─────────────────────────────────────

if not df_vacc.empty:
    fg_vacc = folium.FeatureGroup(name="💉 Centres de vaccination", show=False)
    mc = MarkerCluster(
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 10}
    ).add_to(fg_vacc)

    _name_col = next((c for c in df_vacc.columns
                      if any(k in c.lower() for k in ["name","nom"])), None)
    _dep_col  = next((c for c in df_vacc.columns
                      if c.lower() in ["dep_code", "dep_name", "dep"]), None)

    for _, row in df_vacc.iterrows():
        _nm  = str(row[_name_col]).strip() if _name_col else "Centre de vaccination"
        _dep = str(row[_dep_col]).strip()  if _dep_col  else ""
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(
                f"<b>💉 {_nm}</b><br>Dép.&nbsp;: <b>{_dep}</b>",
                max_width=260
            ),
            tooltip=f"💉 {_nm}",
            icon=folium.Icon(color="green", icon="plus-sign", prefix="glyphicon"),
        ).add_to(mc)

    fg_vacc.add_to(m)
    print(f"  ✓ Centres vaccination : {len(df_vacc):,} marqueurs")


# ── Établissements FINESS (optionnel) ─────────────────────────

if not df_finess.empty:
    fg_fin = folium.FeatureGroup(name="🏥 Établissements FINESS", show=False)
    mc_fin = MarkerCluster(
        options={"maxClusterRadius": 30, "disableClusteringAtZoom": 13}
    ).add_to(fg_fin)

    _nm_col  = next((c for c in df_finess.columns
                     if c in ("rs", "rslongue", "libelle", "raison")), None)
    _cat_col = next((c for c in df_finess.columns if "categ" in c), None)
    _dep_col = "dep" if "dep" in df_finess.columns else None

    for _, row in df_finess.iterrows():
        _nm  = str(row[_nm_col]).title()[:60]  if _nm_col  else "Établissement"
        _cat = str(row[_cat_col])[:6]           if _cat_col else ""
        _dep = str(row[_dep_col])               if _dep_col else ""
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#c0392b",
            fill=True,
            fill_opacity=0.65,
            popup=f"<b>{_nm}</b><br>Cat.&nbsp;: {_cat} | Dép.&nbsp;: {_dep}",
            tooltip=_nm,
        ).add_to(mc_fin)

    fg_fin.add_to(m)
    print(f"  ✓ FINESS : {len(df_finess):,} établissements")


# ── LayerControl ───────────────────────────────────────────────

folium.LayerControl(collapsed=False, position="topright").add_to(m)


# ── Titre ──────────────────────────────────────────────────────

m.get_root().html.add_child(folium.Element("""
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
     z-index:1000;background:rgba(255,255,255,0.93);
     padding:8px 22px;border-radius:8px;border:1px solid #bbb;
     font-family:Arial,sans-serif;text-align:center;pointer-events:none">
  <span style="font-size:15px;font-weight:bold">
    🗺️ Carte COVID-19 France — Impact par département
  </span><br>
  <span style="font-size:10px;color:#666">
    Sources : SPF · VAC-SI · SI-DEP · INSEE · Clustering Phase 3
    &nbsp;|&nbsp; Données : 2020-2023
  </span>
</div>"""))


# ============================================================
# STEP 4 — Sauvegarde
# ============================================================

print("\n" + "=" * 60)
print("💾 STEP 4 — Sauvegarde")
print("=" * 60)

m.save(OUT_MAP)

_size_kb = os.path.getsize(OUT_MAP) / 1024
print(f"  ✓ Carte sauvegardée → {OUT_MAP}  ({_size_kb:.0f} KB)")
print()
print("  Couches disponibles dans la carte :")
print("    ① Taux d'hospitalisation /100K     [défaut]")
print("    ② Taux de positivité %")
print("    ③ Couverture vaccinale %")
print("    ④ Létalité hospitalière %")
print("    ⑤ Clusters épidémiques (Phase 3)")
if not df_vacc.empty:
    print(f"    ⑥ Centres de vaccination ({len(df_vacc):,} marqueurs)")
if not df_finess.empty:
    print(f"    ⑦ Établissements FINESS ({len(df_finess):,} marqueurs)")

print()
print("🚀 Phase 5 terminée — ouvrez outputs/carte_covid_france.html dans votre navigateur")
