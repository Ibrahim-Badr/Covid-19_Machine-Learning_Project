#!/usr/bin/env python3
"""
Phase 5 — Tableau de bord interactif COVID-19 France
Génère outputs/dashboard_covid.html  (HTML autonome, sans serveur)

Contrôles :
  - Curseur de date (hebdomadaire, jan 2020 → mars 2023)
  - Sélecteur d'indicateur (carte + classement)
  - Clic dept sur la carte ou le classement → courbe détaillée

Sections :
  ① Barre de contrôle (curseur, sélecteur, badge vague)
  ② Cartes KPI (hospitalisés, réa, décès, positivité, vaccination)
  ③ Ligne 1 : évolution nationale | carte choroplèthe
  ④ Ligne 2 : top-15 depts (barres) | scatter positivité vs hosp.
  ⑤ Détail département (affiché au clic)
"""
import os, json
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

DAILY_CSV   = "data/processed/master_dept_daily.csv"
GEO_FILE    = "data/processed/master_dept_geo.geojson"
CLUSTER_CSV = "data/clustering/global_cluster_labels.csv"
WAVE_CSV    = "data/processed/wave_periods.csv"
OUT_FILE    = "outputs/dashboard_covid.html"


def normalize_dep(code: str) -> str:
    code = str(code).strip().upper().replace("DEP-", "")
    if code in ("2A", "2B"):
        return code
    try:
        return str(int(float(code))).zfill(2)
    except Exception:
        return code.zfill(2)


# ─────────────────────────── LOAD ────────────────────────────
print("=" * 60)
print("STEP 1 — Chargement des données")
print("=" * 60)

df = pd.read_csv(DAILY_CSV, dtype={"dep": str}, parse_dates=["jour"])
df["dep"] = df["dep"].apply(normalize_dep)
df = df.dropna(subset=["jour"]).sort_values(["dep", "jour"])
print(f"  ✓ {len(df):,} lignes | {df['dep'].nunique()} depts | "
      f"{df['jour'].min().date()} → {df['jour'].max().date()}")

with open(GEO_FILE, encoding="utf-8") as f:
    geo_raw = json.load(f)

# Géo allégé : seulement dep + nom + géométrie
geo = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "dep": normalize_dep(feat["properties"].get("dep", "")),
                "nom": feat["properties"].get("nom", feat["properties"].get("dep", "")),
            },
            "geometry": feat["geometry"],
        }
        for feat in geo_raw["features"]
    ],
}
print(f"  ✓ GeoJSON allégé : {len(geo['features'])} features")

clusters: dict[str, int] = {}
if os.path.exists(CLUSTER_CSV):
    df_cl = pd.read_csv(CLUSTER_CSV, dtype={"dep": str})
    df_cl["dep"] = df_cl["dep"].apply(normalize_dep)
    if "kmeans_cluster" in df_cl.columns:
        clusters = {
            k: int(v)
            for k, v in df_cl.set_index("dep")["kmeans_cluster"]
            .fillna(-1).astype(int).to_dict().items()
        }
    print(f"  ✓ Clusters : {len(clusters)} depts")

df_waves = pd.read_csv(WAVE_CSV, parse_dates=["start", "end"])
waves = [
    {
        "start": r["start"].strftime("%Y-%m-%d"),
        "end":   r["end"].strftime("%Y-%m-%d"),
        "label": r["label"],
        "wave":  int(r["wave"]),
    }
    for _, r in df_waves.iterrows()
]
print(f"  ✓ {len(waves)} vagues")

# ─────────────────────────── MÉTRIQUES ───────────────────────
METRICS = [
    {"key": "hosp_rate",     "label": "Hosp. /100K",         "unit": "/100K", "cs": "Reds"},
    {"key": "new_hosp_rate", "label": "Nouv. hosp. /100K",   "unit": "/100K", "cs": "Oranges"},
    {"key": "tx_pos_7j",     "label": "Positivité 7j (%)",   "unit": "%",     "cs": "YlOrRd"},
    {"key": "couv_complet",  "label": "Couv. vaccinale (%)", "unit": "%",     "cs": "Greens"},
    {"key": "icu_pressure",  "label": "Pression réa (%)",    "unit": "%",     "cs": "Purples"},
    {"key": "cfr",           "label": "Létalité hosp. (%)",  "unit": "%",     "cs": "Greys"},
]
METRIC_KEYS = [m["key"] for m in METRICS]

# ─────────────────────────── NATIONAL ────────────────────────
print("\nSTEP 2 — Agrégation nationale quotidienne")
nat_agg: dict[str, str] = {}
for c in ["hosp_rate", "new_hosp_rate", "tx_pos_7j", "couv_complet", "icu_pressure", "cfr"]:
    if c in df.columns:
        nat_agg[c] = "mean"
for c in ["hosp", "rea", "dc", "new_hosp_7j"]:
    if c in df.columns:
        nat_agg[c] = "sum"

nat = df.groupby("jour").agg(nat_agg).reset_index().sort_values("jour")
nat["jour_str"] = nat["jour"].dt.strftime("%Y-%m-%d")

nat_data: dict = {"dates": nat["jour_str"].tolist()}
for col in nat_agg:
    if col in nat.columns:
        nat_data[col] = [
            None if pd.isna(v) else round(float(v), 2)
            for v in nat[col]
        ]
print(f"  ✓ {len(nat_data['dates'])} dates × {len(nat_agg)} métriques")

# ─────────────────────────── DEPT HEBDO ──────────────────────
print("\nSTEP 3 — Données départementales hebdomadaires")
avail = [k for k in METRIC_KEYS if k in df.columns]
df_sub = df[["dep", "jour"] + avail].copy()
df_sub = (
    df_sub
    .set_index("jour")
    .groupby("dep")
    .resample("W")
    [[k for k in METRIC_KEYS if k in df_sub.columns]]
    .mean()
    .reset_index()
)
df_sub["jour_str"] = df_sub["jour"].dt.strftime("%Y-%m-%d")
weekly_dates: list[str] = sorted(df_sub["jour_str"].unique().tolist())

# {dep: {metric: [value_par_semaine]}}
dept_ts: dict[str, dict[str, list]] = {}
for dep, grp in df_sub.groupby("dep"):
    grp_idx = grp.set_index("jour_str")
    dept_ts[dep] = {}
    for k in METRIC_KEYS:
        if k in grp_idx.columns:
            s = grp_idx[k].reindex(weekly_dates)
            dept_ts[dep][k] = [
                None if pd.isna(v) else round(float(v), 2) for v in s
            ]
        else:
            dept_ts[dep][k] = [None] * len(weekly_dates)

print(f"  ✓ {len(dept_ts)} depts × {len(weekly_dates)} semaines × {len(METRIC_KEYS)} métriques")

# ─────────────────────────── NOMS & POP ──────────────────────
dept_names = {
    feat["properties"]["dep"]: feat["properties"].get("nom", feat["properties"]["dep"])
    for feat in geo["features"]
}

dept_pop: dict[str, int] = {}
if "population" in df.columns:
    for dep, v in df.groupby("dep")["population"].last().items():
        if not pd.isna(v):
            dept_pop[dep] = int(v)

# ─────────────────────────── JSON ────────────────────────────
print("\nSTEP 4 — Sérialisation JSON")

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

j_nat      = to_json(nat_data)
j_dept     = to_json(dept_ts)
j_geo      = to_json(geo)
j_waves    = to_json(waves)
j_clusters = to_json(clusters)
j_dnames   = to_json(dept_names)
j_metrics  = to_json(METRICS)
j_wdates   = to_json(weekly_dates)
j_pop      = to_json(dept_pop)

for name, val in [("nat_data", j_nat), ("dept_ts", j_dept), ("geo", j_geo)]:
    print(f"  ✓ {name} : {len(val)/1024:.0f} KB")

# ─────────────────────────── HTML ────────────────────────────
print("\nSTEP 5 — Génération du tableau de bord")

HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tableau de Bord COVID-19 France</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:#eef2f7;color:#1a2b4a;min-height:100vh}

/* HEADER */
#hdr{background:linear-gradient(135deg,#1a2b4a 0%,#2d4a7a 100%);color:#fff;
     padding:14px 24px;display:flex;align-items:center;justify-content:space-between;
     box-shadow:0 2px 8px rgba(0,0,0,.2)}
#hdr h1{font-size:1.2rem;font-weight:700;letter-spacing:-.3px}
#hdr .sub{font-size:.75rem;opacity:.7;margin-top:2px}
#hdr .src{font-size:.7rem;opacity:.55;text-align:right;line-height:1.5}

/* CONTROLS */
#ctrl{background:#fff;padding:12px 24px;display:flex;align-items:center;
      gap:20px;box-shadow:0 1px 4px rgba(0,0,0,.08);flex-wrap:wrap;position:sticky;top:0;z-index:100}
.slider-wrap{flex:1;min-width:260px;display:flex;flex-direction:column;gap:5px}
.sl-labels{display:flex;justify-content:space-between;font-size:.7rem;color:#94a3b8}
#date-slider{width:100%;accent-color:#3b82f6;cursor:pointer;height:4px}
#date-disp{font-size:1rem;font-weight:700;color:#1a2b4a;min-width:160px;text-align:center;
           background:#f0f4f8;padding:7px 12px;border-radius:8px;white-space:nowrap}
.cg{display:flex;flex-direction:column;gap:3px}
.cg label{font-size:.68rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px}
#metric-sel{padding:6px 10px;border:1px solid #d1d9e0;border-radius:6px;
            font-size:.85rem;background:#fff;cursor:pointer;color:#1a2b4a}
#wave-badge{padding:5px 13px;border-radius:20px;font-size:.75rem;font-weight:600;
            background:#e2e8f0;color:#64748b;white-space:nowrap;transition:all .3s}
#wave-badge.on{background:#dbeafe;color:#1d4ed8}

/* KPI */
#kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:12px 24px}
.kpi{background:#fff;border-radius:10px;padding:12px 16px;
     box-shadow:0 1px 4px rgba(0,0,0,.07);display:flex;flex-direction:column;gap:3px}
.kpi-lbl{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8}
.kpi-val{font-size:1.5rem;font-weight:800;line-height:1.1;color:#1a2b4a}
.kpi-trnd{font-size:.73rem;color:#94a3b8}
.kpi-trnd.up{color:#ef4444}.kpi-trnd.dn{color:#22c55e}.kpi-trnd.dn-bad{color:#ef4444}.kpi-trnd.up-good{color:#22c55e}

/* CHARTS */
.row{display:grid;gap:10px;padding:0 24px 10px}
.row1{grid-template-columns:3fr 2fr}
.row2{grid-template-columns:1fr 1fr}
.card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.07);
      padding:12px;overflow:hidden}
.card-title{font-size:.82rem;font-weight:700;color:#1a2b4a;margin-bottom:6px}

/* DEPT DETAIL */
#detail{margin:0 24px 20px;background:#fff;border-radius:10px;
        box-shadow:0 1px 4px rgba(0,0,0,.07);padding:12px;display:none}
#detail-title{font-size:.85rem;font-weight:700;color:#1a2b4a;margin-bottom:6px}
#close-btn{float:right;cursor:pointer;background:#f0f4f8;border:none;
           border-radius:6px;padding:4px 10px;font-size:.78rem;color:#64748b}
#close-btn:hover{background:#e2e8f0}

/* FOOTER */
footer{text-align:center;padding:10px;font-size:.68rem;color:#94a3b8}

@media(max-width:900px){
  #kpis{grid-template-columns:repeat(3,1fr)}
  .row1,.row2{grid-template-columns:1fr}
}
@media(max-width:580px){#kpis{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>

<div id="hdr">
  <div>
    <h1>&#x1F4CA; Tableau de Bord COVID-19 France</h1>
    <div class="sub">101 d&eacute;partements &middot; Janvier 2020 &ndash; Mars 2023</div>
  </div>
  <div class="src">Sources&nbsp;: SPF &middot; VAC-SI &middot; SI-DEP &middot; INSEE<br>
    Phases&nbsp;1&ndash;4&nbsp;: acquisition, pr&eacute;traitement, clustering, ML</div>
</div>

<div id="ctrl">
  <div class="slider-wrap">
    <div class="sl-labels">
      <span>Jan&nbsp;2020</span><span>2021</span><span>2022</span><span>Mars&nbsp;2023</span>
    </div>
    <input type="range" id="date-slider" min="0" step="1" value="0">
  </div>
  <div id="date-disp">&mdash;</div>
  <div class="cg">
    <label>Indicateur carte &amp; classement</label>
    <select id="metric-sel"></select>
  </div>
  <div id="wave-badge">Hors p&eacute;riode de vague</div>
</div>

<div id="kpis">
  <div class="kpi">
    <div class="kpi-lbl">Hospitalis&eacute;s</div>
    <div class="kpi-val" id="kv-hosp">&mdash;</div>
    <div class="kpi-trnd" id="kt-hosp"></div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">R&eacute;animation</div>
    <div class="kpi-val" id="kv-rea">&mdash;</div>
    <div class="kpi-trnd" id="kt-rea"></div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">D&eacute;c&egrave;s cumul&eacute;s</div>
    <div class="kpi-val" id="kv-dc">&mdash;</div>
    <div class="kpi-trnd" id="kt-dc"></div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Positivit&eacute; 7j</div>
    <div class="kpi-val" id="kv-txpos">&mdash;</div>
    <div class="kpi-trnd" id="kt-txpos"></div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Couv. vaccinale</div>
    <div class="kpi-val" id="kv-vacc">&mdash;</div>
    <div class="kpi-trnd" id="kt-vacc"></div>
  </div>
</div>

<div class="row row1">
  <div class="card">
    <div class="card-title">&#x1F4C8; &Eacute;volution nationale &mdash; Hospitalis&eacute;s &amp; R&eacute;animation</div>
    <div id="ch-nat"></div>
  </div>
  <div class="card">
    <div class="card-title">&#x1F5FA;&#xFE0F; Carte France &mdash; <span id="map-lbl">Hosp. /100K</span></div>
    <div id="ch-map"></div>
  </div>
</div>

<div class="row row2">
  <div class="card">
    <div class="card-title">&#x1F3C6; Top 15 d&eacute;partements &mdash; <span id="bar-lbl">Hosp. /100K</span></div>
    <div id="ch-bar"></div>
  </div>
  <div class="card">
    <div class="card-title">&#x1F50D; Positivit&eacute; 7j vs Hospitalisation par d&eacute;partement</div>
    <div id="ch-scatter"></div>
  </div>
</div>

<div id="detail">
  <button id="close-btn" onclick="closeDetail()">&#x2715; Fermer</button>
  <div id="detail-title">&Eacute;volution &mdash; d&eacute;partement</div>
  <div id="ch-dept"></div>
</div>

<footer>Phase 5 &middot; Dashboard COVID-19 France &middot; Donn&eacute;es 2020&ndash;2023</footer>

<script>
// ── DONNÉES ──────────────────────────────────────────────────
const NAT      = __NAT__;
const DEPT     = __DEPT__;
const GEO      = __GEO__;
const WAVES    = __WAVES__;
const CLUSTERS = __CLUSTERS__;
const DNAMES   = __DNAMES__;
const METRICS  = __METRICS__;
const WDATES   = __WDATES__;
const POP      = __POP__;

// ── ÉTAT ─────────────────────────────────────────────────────
let curIdx    = WDATES.length - 1;
let curMetric = METRICS[0].key;

// ── UTILITAIRES ───────────────────────────────────────────────
function fmtNum(v, dec=0, suf='') {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const n = parseFloat(v);
  if (dec === 0 && Math.abs(n) >= 1000)
    return n.toLocaleString('fr-FR', {maximumFractionDigits:0}) + suf;
  return n.toLocaleString('fr-FR', {minimumFractionDigits:dec, maximumFractionDigits:dec}) + suf;
}

function closestNatIdx(dateStr) {
  const exact = NAT.dates.indexOf(dateStr);
  if (exact >= 0) return exact;
  let lo = 0, hi = NAT.dates.length - 1;
  while (lo < hi) { const m = (lo+hi)>>1; if (NAT.dates[m] < dateStr) lo=m+1; else hi=m; }
  return Math.min(lo, NAT.dates.length - 1);
}

function getWave(dateStr) {
  for (const w of WAVES) if (dateStr >= w.start && dateStr <= w.end) return w;
  return null;
}

function getMetric(key) { return METRICS.find(m => m.key === key) || METRICS[0]; }

function clColor(cl) {
  const palette = ['#3b82f6','#ef4444','#22c55e','#f59e0b','#8b5cf6','#06b6d4'];
  return cl < 0 ? '#94a3b8' : palette[cl % palette.length];
}

function getDeptVals(dateIdx, metricKey) {
  const locs=[], vals=[], names=[], cls=[];
  for (const [dep, data] of Object.entries(DEPT)) {
    const arr = data[metricKey];
    const v   = arr ? arr[dateIdx] : null;
    if (v !== null && v !== undefined) {
      locs.push(dep); vals.push(v);
      names.push(DNAMES[dep] || dep);
      cls.push(CLUSTERS[dep] !== undefined ? CLUSTERS[dep] : -1);
    }
  }
  return {locs, vals, names, cls};
}

function trendText(curr, prev, higherIsBad=true) {
  if (curr === null || prev === null || prev === 0) return {txt:'', cls:''};
  const d = curr - prev, pct = Math.abs(d / prev * 100);
  if (pct < 0.5) return {txt:'→ stable', cls:''};
  const arrow = d > 0 ? '▲' : '▼';
  const txt = `${arrow} ${pct.toFixed(1)}% vs sem. préc.`;
  let cls = '';
  if (d > 0) cls = higherIsBad ? 'up' : 'up-good';
  else       cls = higherIsBad ? 'dn' : 'dn-bad';
  return {txt, cls};
}

// ── WAVE COLORS (shapes) ─────────────────────────────────────
const WAVE_FILLS = ['#dbeafe','#fee2e2','#dcfce7','#fef3c7','#f3e8ff','#ffedd5','#e0f2fe'];
function waveShapes() {
  return WAVES.map((w,i) => ({
    type:'rect', xref:'x', yref:'paper',
    x0:w.start, x1:w.end, y0:0, y1:1,
    fillcolor: WAVE_FILLS[i % WAVE_FILLS.length],
    opacity:0.22, line:{width:0}, layer:'below',
  }));
}

// ── CONTRÔLES ─────────────────────────────────────────────────
function initControls() {
  const sl = document.getElementById('date-slider');
  sl.max = WDATES.length - 1;
  sl.value = curIdx;

  const sel = document.getElementById('metric-sel');
  METRICS.forEach(m => {
    const o = document.createElement('option');
    o.value = m.key; o.textContent = m.label;
    sel.appendChild(o);
  });
  sel.value = curMetric;

  sl.addEventListener('input', e => { curIdx = +e.target.value; updateAll(); });
  sel.addEventListener('change', e => {
    curMetric = e.target.value;
    const cfg = getMetric(curMetric);
    document.getElementById('map-lbl').textContent = cfg.label;
    document.getElementById('bar-lbl').textContent = cfg.label;
    updateMap();
    updateBar();
  });
}

// ── DATE DISPLAY ──────────────────────────────────────────────
function updateDateDisplay() {
  const d = new Date(WDATES[curIdx] + 'T00:00:00');
  document.getElementById('date-disp').textContent =
    d.toLocaleDateString('fr-FR', {year:'numeric', month:'long', day:'numeric'});
}

// ── WAVE BADGE ─────────────────────────────────────────────────
function updateWaveBadge() {
  const w = getWave(WDATES[curIdx]);
  const el = document.getElementById('wave-badge');
  if (w) {
    el.textContent = w.label;
    el.classList.add('on');
  } else {
    el.textContent = 'Hors période de vague';
    el.classList.remove('on');
  }
}

// ── KPI ──────────────────────────────────────────────────────
function updateKPIs() {
  const ni  = closestNatIdx(WDATES[curIdx]);
  const ni7 = Math.max(0, ni - 7);

  function set(id, curr, prev, dec, suf, higherIsBad=true) {
    const ve = document.getElementById('kv-' + id);
    const te = document.getElementById('kt-' + id);
    if (ve) ve.textContent = fmtNum(curr, dec, suf);
    if (te) {
      const {txt, cls} = trendText(curr, prev, higherIsBad);
      te.textContent = txt; te.className = 'kpi-trnd ' + cls;
    }
  }

  set('hosp',  NAT.hosp?.[ni],       NAT.hosp?.[ni7],       0, '');
  set('rea',   NAT.rea?.[ni],        NAT.rea?.[ni7],         0, '');
  set('dc',    NAT.dc?.[ni],         NAT.dc?.[ni7],          0, '');
  set('txpos', NAT.tx_pos_7j?.[ni],  NAT.tx_pos_7j?.[ni7],  1, '%');
  set('vacc',  NAT.couv_complet?.[ni], NAT.couv_complet?.[ni7], 1, '%', false);
}

// ── GRAPHIQUE NATIONAL ────────────────────────────────────────
function initNatChart() {
  const traces = [];
  if (NAT.hosp) traces.push({
    x:NAT.dates, y:NAT.hosp, type:'scatter', mode:'lines', name:'Hospitalisés',
    line:{color:'#3b82f6', width:2},
    fill:'tozeroy', fillcolor:'rgba(59,130,246,.10)',
    hovertemplate:'%{y:,.0f}<extra>Hospitalisés</extra>',
  });
  if (NAT.rea) traces.push({
    x:NAT.dates, y:NAT.rea, type:'scatter', mode:'lines', name:'Réanimation',
    line:{color:'#ef4444', width:2},
    fill:'tozeroy', fillcolor:'rgba(239,68,68,.12)',
    hovertemplate:'%{y:,.0f}<extra>Réanimation</extra>',
  });
  if (NAT.new_hosp_7j) traces.push({
    x:NAT.dates, y:NAT.new_hosp_7j, type:'scatter', mode:'lines', name:'Nouv. hosp. 7j',
    line:{color:'#f59e0b', width:1.5, dash:'dot'},
    hovertemplate:'%{y:,.0f}<extra>Nouv. hosp. 7j</extra>',
  });
  if (NAT.tx_pos_7j) traces.push({
    x:NAT.dates, y:NAT.tx_pos_7j, type:'scatter', mode:'lines', name:'Positivité 7j (%)',
    line:{color:'#8b5cf6', width:1.5, dash:'dash'},
    yaxis:'y2', hovertemplate:'%{y:.1f}%<extra>Positivité 7j</extra>',
  });

  const cursor = {
    type:'line', xref:'x', yref:'paper',
    x0:WDATES[curIdx], x1:WDATES[curIdx], y0:0, y1:1,
    line:{color:'#1a2b4a', width:1.8, dash:'dot'},
  };

  const layout = {
    height:290, margin:{l:52,r:52,t:6,b:42},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    legend:{orientation:'h', y:-0.18, x:0, font:{size:10}},
    xaxis:{showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:10}},
    yaxis:{title:{text:'Personnes', font:{size:10}}, showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:10}},
    yaxis2:{title:{text:'Positivité (%)', font:{size:10}}, overlaying:'y', side:'right',
            showgrid:false, tickfont:{size:10}, range:[0,50]},
    shapes:[...waveShapes(), cursor],
    hovermode:'x unified',
  };
  Plotly.newPlot('ch-nat', traces, layout, {responsive:true, displayModeBar:false});
}

function updateNatCursor() {
  const si = WAVES.length; // cursor is the last shape
  Plotly.relayout('ch-nat', {
    [`shapes[${si}].x0`]: WDATES[curIdx],
    [`shapes[${si}].x1`]: WDATES[curIdx],
  });
}

// ── CARTE ────────────────────────────────────────────────────
function initMap() {
  const {locs, vals, names} = getDeptVals(curIdx, curMetric);
  const cfg = getMetric(curMetric);
  const trace = {
    type:'choropleth', geojson:GEO,
    locations:locs, z:vals, featureidkey:'properties.dep',
    colorscale:cfg.cs, text:names,
    colorbar:{thickness:11, len:.65, title:{text:cfg.unit, font:{size:10}}, tickfont:{size:9}},
    marker:{line:{color:'#fff', width:.5}},
    hovertemplate:'<b>%{text}</b><br>' + cfg.label + ': %{z:.2f} ' + cfg.unit + '<extra></extra>',
  };
  const layout = {
    height:290, margin:{l:0,r:0,t:0,b:0},
    paper_bgcolor:'#fff',
    geo:{fitbounds:'locations', visible:false, bgcolor:'#fff', projection:{type:'mercator'}},
  };
  Plotly.newPlot('ch-map', [trace], layout, {responsive:true, displayModeBar:false});

  document.getElementById('ch-map').on('plotly_click', d => {
    if (d.points.length > 0) showDeptDetail(d.points[0].location);
  });
}

function updateMap() {
  const {locs, vals, names} = getDeptVals(curIdx, curMetric);
  const cfg = getMetric(curMetric);
  Plotly.restyle('ch-map', {
    locations:[locs], z:[vals], text:[names], colorscale:cfg.cs,
    hovertemplate:['<b>%{text}</b><br>' + cfg.label + ': %{z:.2f} ' + cfg.unit + '<extra></extra>'],
  }, 0);
}

// ── CLASSEMENT BARRES ────────────────────────────────────────
function buildBarData(dateIdx, metricKey) {
  const {locs, vals, names, cls} = getDeptVals(dateIdx, metricKey);
  return locs
    .map((d,i) => ({dep:d, val:vals[i], name:names[i], cl:cls[i]}))
    .filter(x => x.val !== null)
    .sort((a,b) => b.val - a.val)
    .slice(0, 15);
}

function initBar() {
  const sorted = buildBarData(curIdx, curMetric);
  const cfg = getMetric(curMetric);
  const trace = {
    type:'bar', orientation:'h',
    x:sorted.map(x => x.val),
    y:sorted.map(x => `${x.dep} — ${x.name}`),
    marker:{color:sorted.map(x => clColor(x.cl))},
    text:sorted.map(x => x.val.toFixed(1)),
    textposition:'outside', textfont:{size:9},
    hovertemplate:'<b>%{y}</b><br>' + cfg.label + ': %{x:.2f}<extra></extra>',
  };
  const layout = {
    height:290, margin:{l:130,r:55,t:6,b:35},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    xaxis:{title:{text:cfg.unit, font:{size:10}}, showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:9}},
    yaxis:{tickfont:{size:9}, autorange:'reversed'},
    clickmode:'event',
  };
  Plotly.newPlot('ch-bar', [trace], layout, {responsive:true, displayModeBar:false});

  document.getElementById('ch-bar').on('plotly_click', d => {
    if (d.points.length > 0) {
      const label = d.points[0].y;
      const dep = label.split(' — ')[0].trim();
      showDeptDetail(dep);
    }
  });
}

function updateBar() {
  const sorted = buildBarData(curIdx, curMetric);
  const cfg = getMetric(curMetric);
  Plotly.restyle('ch-bar', {
    x:[sorted.map(x => x.val)],
    y:[sorted.map(x => `${x.dep} — ${x.name}`)],
    'marker.color':[sorted.map(x => clColor(x.cl))],
    text:[sorted.map(x => x.val.toFixed(1))],
    hovertemplate:['<b>%{y}</b><br>' + cfg.label + ': %{x:.2f}<extra></extra>'],
  }, 0);
  Plotly.relayout('ch-bar', {'xaxis.title.text': cfg.unit});
}

// ── SCATTER ───────────────────────────────────────────────────
function buildScatterData(dateIdx) {
  const h = getDeptVals(dateIdx, 'hosp_rate');
  const t = getDeptVals(dateIdx, 'tx_pos_7j');
  const hMap = {};
  h.locs.forEach((d,i) => hMap[d] = {hosp:h.vals[i], name:h.names[i], cl:h.cls[i]});
  t.locs.forEach((d,i) => { if (hMap[d]) hMap[d].txpos = t.vals[i]; });
  return Object.entries(hMap)
    .filter(([,p]) => p.hosp !== undefined && p.txpos !== undefined)
    .map(([dep, p]) => ({dep, ...p}));
}

function initScatter() {
  const pts = buildScatterData(curIdx);
  const trace = {
    type:'scatter', mode:'markers',
    x:pts.map(p => p.txpos), y:pts.map(p => p.hosp),
    text:pts.map(p => `${p.dep} — ${p.name}`),
    marker:{size:9, color:pts.map(p => clColor(p.cl)), opacity:.8, line:{width:1, color:'#fff'}},
    hovertemplate:'<b>%{text}</b><br>Positivité 7j: %{x:.1f}%<br>Hosp./100K: %{y:.2f}<extra></extra>',
  };
  const layout = {
    height:290, margin:{l:52,r:20,t:6,b:42},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    xaxis:{title:{text:'Taux positivité 7j (%)', font:{size:10}},
           showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:10}},
    yaxis:{title:{text:'Hosp. /100K', font:{size:10}},
           showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:10}},
    showlegend:false,
  };
  Plotly.newPlot('ch-scatter', [trace], layout, {responsive:true, displayModeBar:false});
}

function updateScatter() {
  const pts = buildScatterData(curIdx);
  Plotly.restyle('ch-scatter', {
    x:[pts.map(p => p.txpos)], y:[pts.map(p => p.hosp)],
    text:[pts.map(p => `${p.dep} — ${p.name}`)],
    'marker.color':[pts.map(p => clColor(p.cl))],
  }, 0);
}

// ── DÉTAIL DÉPARTEMENT ────────────────────────────────────────
function showDeptDetail(dep) {
  const name = DNAMES[dep] || dep;
  document.getElementById('detail-title').textContent =
    `Évolution — ${dep} ${name}`;
  document.getElementById('detail').style.display = 'block';

  const data = DEPT[dep];
  if (!data) return;

  const addTrace = (key, label, color, dash='solid', yax='y') => {
    const vals = data[key];
    if (!vals) return null;
    return {
      x:WDATES, y:vals, type:'scatter', mode:'lines', name:label,
      line:{color, width:2, dash}, yaxis:yax,
      hovertemplate:`%{y:.2f}<extra>${label}</extra>`,
    };
  };

  const traces = [
    addTrace('hosp_rate',     'Hosp. /100K',         '#3b82f6'),
    addTrace('new_hosp_rate', 'Nouv. hosp. /100K',   '#f59e0b', 'dot'),
    addTrace('icu_pressure',  'Pression réa (%)',     '#ef4444', 'dash'),
    addTrace('tx_pos_7j',     'Positivité 7j (%)',    '#8b5cf6', 'solid', 'y2'),
    addTrace('couv_complet',  'Couv. vaccinale (%)',  '#22c55e', 'dash',  'y2'),
  ].filter(Boolean);

  const layout = {
    height:290, margin:{l:52,r:52,t:6,b:52},
    paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    legend:{orientation:'h', y:-0.22, font:{size:10}},
    xaxis:{showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:10}},
    yaxis:{title:{text:'Valeur', font:{size:10}}, showgrid:true, gridcolor:'#f0f4f8', tickfont:{size:10}},
    yaxis2:{title:{text:'%', font:{size:10}}, overlaying:'y', side:'right',
            showgrid:false, tickfont:{size:10}, range:[0,100]},
    shapes:waveShapes(),
    hovermode:'x unified',
  };

  Plotly.newPlot('ch-dept', traces, layout, {responsive:true, displayModeBar:false});
  document.getElementById('detail').scrollIntoView({behavior:'smooth', block:'nearest'});
}

function closeDetail() {
  document.getElementById('detail').style.display = 'none';
}

// ── MISE À JOUR GLOBALE ────────────────────────────────────────
function updateAll() {
  updateDateDisplay();
  updateWaveBadge();
  updateKPIs();
  updateNatCursor();
  updateMap();
  updateBar();
  updateScatter();
}

// ── INITIALISATION ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initControls();
  initNatChart();
  initMap();
  initBar();
  initScatter();
  updateAll();
});
</script>
</body>
</html>
"""

# Injection des données
HTML = HTML.replace("__NAT__",      j_nat)
HTML = HTML.replace("__DEPT__",     j_dept)
HTML = HTML.replace("__GEO__",      j_geo)
HTML = HTML.replace("__WAVES__",    j_waves)
HTML = HTML.replace("__CLUSTERS__", j_clusters)
HTML = HTML.replace("__DNAMES__",   j_dnames)
HTML = HTML.replace("__METRICS__",  j_metrics)
HTML = HTML.replace("__WDATES__",   j_wdates)
HTML = HTML.replace("__POP__",      j_pop)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(HTML)

size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
print(f"\n  ✓ Tableau de bord → {OUT_FILE}  ({size_mb:.1f} MB)")
print()
print("  Fonctionnalités :")
print("    ① Curseur de date (hebdomadaire, 2020-2023)")
print("    ② Cartes KPI (hosp, réa, décès, positivité, vaccination)")
print("    ③ Évolution nationale avec fond de vagues colorées")
print("    ④ Carte choroplèthe (6 indicateurs au choix)")
print("    ⑤ Classement top-15 départements (colorés par cluster)")
print("    ⑥ Scatter positivité vs hospitalisation")
print("    ⑦ Courbe détaillée au clic sur carte ou classement")
print()
print("  Ouvrez outputs/dashboard_covid.html dans votre navigateur")
