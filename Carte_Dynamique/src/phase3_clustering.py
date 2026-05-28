# ============================================================
# CARTE DYNAMIQUE — Phase 3: Clustering
# ============================================================
#
# INPUT  (data/processed/):
#   - dept_clusters_features.csv   → 101 depts × 19 features (Phase 2)
#   - master_dept_latest.csv       → for cluster annotation
#   - wave_periods.csv             → wave labels
#
# OUTPUT (data/processed/, data/clustering/):
#   - pca_components.csv           → 2D/3D PCA coordinates per dept
#   - kmeans_labels.csv            → K-Means cluster assignment
#   - dbscan_labels.csv            → DBSCAN cluster + outlier flags
#   - cluster_profiles.csv         → per-cluster mean feature profile
#   - clustering_report.txt        → full interpretability report
#
# STEPS:
#   1.  Load & validate feature matrix
#   2.  Feature scaling (RobustScaler — outlier-resistant)
#   3.  PCA — variance explained + 2D/3D projections
#   4.  K-Means — elbow + silhouette → optimal k
#   5.  K-Means — final fit
#   6.  DBSCAN — parameter search (eps, min_samples)
#   7.  Cluster profiles — mean features per cluster
#   8.  DOM analysis — are overseas depts isolated?
#   9.  Wave-level clustering — did cluster membership change per wave?
#   10. Outputs + interpretability report
#
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

os.makedirs("data/clustering", exist_ok=True)

# ── Constants ────────────────────────────────────────────────
RANDOM_STATE  = 42
K_RANGE       = range(2, 11)        # k candidates for elbow/silhouette
DBSCAN_EPS    = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
DBSCAN_MIN_S  = [2, 3, 4, 5]
DOM_DEPS      = {"971", "972", "973", "974", "976"}


# ============================================================
# STEP 1 — Load & Validate Feature Matrix
# ============================================================

print("=" * 60)
print("📂 STEP 1 — Load & validate feature matrix")
print("=" * 60)

df_feat = pd.read_csv("data/processed/dept_clusters_features.csv", dtype={"dep": str})
df_latest = pd.read_csv("data/processed/master_dept_latest.csv",   dtype={"dep": str})

df_feat_dom   = df_feat[df_feat["dep"].isin(DOM_DEPS)].copy()
df_feat       = df_feat[~df_feat["dep"].isin(DOM_DEPS)].copy()
print(f"  ✓ Metropolitan France : {len(df_feat)} depts (clustering target)")
print(f"  ✓ DOM set aside       : {len(df_feat_dom)} depts (analysed separately in Step 8)")

# Attach dept name for readability (both metro and DOM)
if "nom" in df_latest.columns:
    _nom_map = df_latest[["dep", "nom"]].drop_duplicates()
    df_feat     = df_feat.merge(_nom_map, on="dep", how="left")
    df_feat_dom = df_feat_dom.merge(_nom_map, on="dep", how="left")

print(f"  ✓ Feature matrix  : {df_feat.shape[0]} depts × {df_feat.shape[1]} cols")

# Separate identifiers from numeric features
ID_COLS      = ["dep", "nom"] if "nom" in df_feat.columns else ["dep"]
STATIC_EXCL  = ["is_dom", "population", "R0_mean", "R0_max"]  # R0: 9.5% fill, quasi-constant
FEATURE_COLS = [
    c for c in df_feat.columns
    if c not in ID_COLS + STATIC_EXCL
    and pd.api.types.is_numeric_dtype(df_feat[c])
]

print(f"  ✓ Feature columns : {FEATURE_COLS}")
print(f"  ✓ Excluded (static): {STATIC_EXCL}")

X_raw = df_feat[FEATURE_COLS].copy()

# Sanity: no remaining NaN after Phase 2 median-fill
nan_count = X_raw.isnull().sum().sum()
if nan_count > 0:
    print(f"  ⚠️  {nan_count} NaN found — filling with column median")
    X_raw = X_raw.fillna(X_raw.median())
else:
    print(f"  ✓ No NaN — feature matrix clean")

print(f"\n  Feature summary:")
for _c in FEATURE_COLS:
    print(f"    {_c:<30} min={X_raw[_c].min():8.2f}  "
          f"max={X_raw[_c].max():8.2f}  mean={X_raw[_c].mean():8.2f}")


# ============================================================
# STEP 2 — Feature Scaling (RobustScaler)
# RobustScaler uses median + IQR → resistant to outlier depts
# (Paris, Hauts-de-Seine, DOM extreme values)
# ============================================================

print("\n" + "=" * 60)
print("⚖️  STEP 2 — Feature scaling (RobustScaler)")
print("=" * 60)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"  ✓ Scaled shape : {X_scaled.shape}")
print(f"  ✓ Post-scale   : mean={X_scaled.mean():.4f}  std={X_scaled.std():.4f}")
print(f"  ✓ Range        : [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")


# ============================================================
# STEP 3 — PCA: Variance Explained + 2D/3D Projections
# ============================================================

print("\n" + "=" * 60)
print("📐 STEP 3 — PCA: variance explained + projections")
print("=" * 60)

# Full PCA to inspect cumulative variance
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95   = int(np.argmax(cumvar >= 0.95)) + 1
n_90   = int(np.argmax(cumvar >= 0.90)) + 1
n_80   = int(np.argmax(cumvar >= 0.80)) + 1

print(f"  ✓ Components to explain 80% variance : {n_80}")
print(f"  ✓ Components to explain 90% variance : {n_90}")
print(f"  ✓ Components to explain 95% variance : {n_95}")
print(f"\n  Per-component variance explained:")
for i, v in enumerate(pca_full.explained_variance_ratio_[:8]):
    bar = "█" * round(v * 100 / 2) + "░" * (50 - round(v * 100 / 2))
    print(f"    PC{i+1:2d}  [{bar}] {v*100:5.1f}%")

# 2D projection (for map + scatter plots in Phase 5)
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d   = pca_2d.fit_transform(X_scaled)

# 3D projection (for interactive Plotly in Phase 5)
pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
X_3d   = pca_3d.fit_transform(X_scaled)

# Save PCA coordinates
df_pca = df_feat[ID_COLS].copy()
df_pca["pc1"] = X_2d[:, 0]
df_pca["pc2"] = X_2d[:, 1]
df_pca["pc3"] = X_3d[:, 2]
df_pca["is_dom"] = df_feat["dep"].isin(DOM_DEPS).astype(int)
df_pca.to_csv("data/clustering/pca_components.csv", index=False)

print(f"\n  ✓ 2D PCA: PC1={pca_2d.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca_2d.explained_variance_ratio_[1]*100:.1f}%  "
      f"(total={sum(pca_2d.explained_variance_ratio_)*100:.1f}%)")
print(f"  ✓ 3D PCA: +PC3={pca_3d.explained_variance_ratio_[2]*100:.1f}%  "
      f"(total={sum(pca_3d.explained_variance_ratio_)*100:.1f}%)")

# PCA loadings — which features drive each component?
df_loadings = pd.DataFrame(
    pca_2d.components_.T,
    index=FEATURE_COLS,
    columns=["PC1", "PC2"]
).round(4)
df_loadings["abs_pc1"] = df_loadings["PC1"].abs()
df_loadings = df_loadings.sort_values("abs_pc1", ascending=False).drop(columns="abs_pc1")
df_loadings.to_csv("data/clustering/pca_loadings.csv")
print(f"\n  📊 PCA Loadings (top drivers):")
print(f"  {'Feature':<30} {'PC1':>8}  {'PC2':>8}")
print(f"  {'-'*50}")
for feat, row in df_loadings.head(8).iterrows():
    print(f"  {feat:<30} {row['PC1']:>8.4f}  {row['PC2']:>8.4f}")


# ============================================================
# STEP 4 — K-Means: Elbow + Silhouette → Optimal k
# ============================================================

print("\n" + "=" * 60)
print("📊 STEP 4 — K-Means: elbow + silhouette search")
print("=" * 60)

inertias, silhouettes, ch_scores, db_scores = [], [], [], []

print(f"\n  {'k':>3}  {'Inertia':>10}  {'Silhouette':>12}  "
      f"{'Calinski-H':>12}  {'Davies-B':>10}")
print(f"  {'-'*55}")

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    ine = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    inertias.append(ine)
    silhouettes.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)
    print(f"  {k:>3}  {ine:>10.1f}  {sil:>12.4f}  {ch:>12.1f}  {db:>10.4f}")

# Save metrics for Phase 5 charting
df_kmeans_metrics = pd.DataFrame({
    "k":          list(K_RANGE),
    "inertia":    inertias,
    "silhouette": silhouettes,
    "calinski_h": ch_scores,
    "davies_b":   db_scores,
})
df_kmeans_metrics.to_csv("data/clustering/kmeans_metrics.csv", index=False)

# Elbow: largest drop in inertia delta
inertia_deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
elbow_k = list(K_RANGE)[inertia_deltas.index(max(inertia_deltas))]

# Best silhouette
sil_k = list(K_RANGE)[silhouettes.index(max(silhouettes))]

# Best Calinski-Harabasz
ch_k = list(K_RANGE)[ch_scores.index(max(ch_scores))]

print(f"\n  ✓ Elbow method suggests          k = {elbow_k}")
print(f"  ✓ Best silhouette score suggests  k = {sil_k}  ({max(silhouettes):.4f})")
print(f"  ✓ Best Calinski-Harabasz suggests k = {ch_k}")

# Consensus: majority vote among the 3 criteria
votes = [elbow_k, sil_k, ch_k]
OPTIMAL_K = max(set(votes), key=votes.count)
# Tiebreak: prefer silhouette (most interpretable for geographic clustering)
if len(set(votes)) == len(votes):
    OPTIMAL_K = sil_k

print(f"\n  🎯 OPTIMAL K = {OPTIMAL_K}  "
      f"(votes: elbow={elbow_k}, silhouette={sil_k}, CH={ch_k})")


# ============================================================
# STEP 5 — K-Means: Final Fit with Optimal k
# ============================================================

print("\n" + "=" * 60)
print(f"🎯 STEP 5 — K-Means final fit (k={OPTIMAL_K})")
print("=" * 60)

km_final = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=RANDOM_STATE,
    n_init=50,       # more initialisations for stable result
    max_iter=1000
)
km_labels = km_final.fit_predict(X_scaled)

df_feat["kmeans_cluster"] = km_labels

sil_final = silhouette_score(X_scaled, km_labels)
ch_final  = calinski_harabasz_score(X_scaled, km_labels)
db_final  = davies_bouldin_score(X_scaled, km_labels)

print(f"  ✓ Silhouette score      : {sil_final:.4f}  (>0.5 = strong, >0.25 = reasonable)")
print(f"  ✓ Calinski-Harabasz     : {ch_final:.1f}   (higher = better)")
print(f"  ✓ Davies-Bouldin        : {db_final:.4f}  (lower = better)")
print(f"  ✓ Inertia               : {km_final.inertia_:.1f}")

print(f"\n  Cluster sizes:")
for _cl in sorted(np.unique(km_labels)):
    _depts = df_feat[df_feat["kmeans_cluster"] == _cl]["dep"].tolist()
    _doms  = [d for d in _depts if d in DOM_DEPS]
    _names = (
        df_feat[df_feat["kmeans_cluster"] == _cl]["nom"]
        .dropna()
        .astype(str)
        .tolist()
        if "nom" in df_feat.columns else _depts
    )
    print(f"    Cluster {_cl}: {len(_depts):>3} depts"
          + (f"  [DOM: {_doms}]" if _doms else ""))
    print(f"             → {', '.join(sorted(_names[:8]))}"
          + (" ..." if len(_names) > 8 else ""))

# Save K-Means labels
df_kmeans_out = df_feat[ID_COLS + ["kmeans_cluster"]].copy()
df_kmeans_out["is_dom"] = df_feat["dep"].isin(DOM_DEPS).astype(int)
df_kmeans_out = df_kmeans_out.merge(df_pca[["dep", "pc1", "pc2", "pc3"]],
                                     on="dep", how="left")
df_kmeans_out.to_csv("data/clustering/kmeans_labels.csv", index=False)
print(f"\n  ✓ K-Means labels saved → data/clustering/kmeans_labels.csv")

# Assign DOM to nearest metropolitan cluster centroid (predict with fitted model)
print("\n  📍 DOM → nearest metropolitan centroid:")
if not df_feat_dom.empty:
    _feat_cols_dom = [c for c in FEATURE_COLS if c in df_feat_dom.columns]
    _X_dom = df_feat_dom[_feat_cols_dom].copy()
    # Fill missing features with metropolitan median
    for _mc in [c for c in FEATURE_COLS if c not in df_feat_dom.columns]:
        _X_dom[_mc] = X_raw[_mc].median()
    _X_dom = _X_dom[FEATURE_COLS].apply(
        lambda col: col.fillna(X_raw[col.name].median()), axis=0
    )
    _X_dom_scaled = scaler.transform(_X_dom)
    df_feat_dom = df_feat_dom.copy()
    df_feat_dom["kmeans_cluster"] = km_final.predict(_X_dom_scaled)
    df_feat_dom["dbscan_cluster"] = -2  # DOM marker: excluded from density clustering
    for _, _r in df_feat_dom.iterrows():
        _nm = _r.get("nom", _r["dep"])
        print(f"       {_r['dep']} {str(_nm):<20} → Cluster {int(_r['kmeans_cluster'])}")


# ============================================================
# STEP 6 — DBSCAN: Parameter Search
# Finds natural density-based clusters + outlier depts
# ============================================================

print("\n" + "=" * 60)
print("🔍 STEP 6 — DBSCAN: parameter search (eps × min_samples)")
print("=" * 60)

# Use PCA-reduced space (n_95 components) for DBSCAN — avoids curse of dimensionality
pca_db = PCA(n_components=min(n_90, X_scaled.shape[1]), random_state=RANDOM_STATE)
X_db   = pca_db.fit_transform(X_scaled)
print(f"  ✓ DBSCAN input: {X_db.shape[1]} PCA components "
      f"({pca_db.explained_variance_ratio_.sum()*100:.1f}% variance)")

best_db   = None
best_sil  = -1
best_params = {}

print(f"\n  {'eps':>6}  {'min_s':>6}  {'n_clust':>8}  "
      f"{'n_outlier':>10}  {'silhouette':>12}")
print(f"  {'-'*50}")

for eps in DBSCAN_EPS:
    for min_s in DBSCAN_MIN_S:
        db  = DBSCAN(eps=eps, min_samples=min_s)
        lbl = db.fit_predict(X_db)
        n_clusters  = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_outliers  = (lbl == -1).sum()
        if n_clusters >= 2 and n_outliers < len(lbl) * 0.3:
            sil = silhouette_score(X_db, lbl)
            flag = " ← best" if sil > best_sil else ""
            print(f"  {eps:>6.1f}  {min_s:>6}  {n_clusters:>8}  "
                  f"{n_outliers:>10}  {sil:>12.4f}{flag}")
            if sil > best_sil:
                best_sil    = sil
                best_db     = lbl
                best_params = {"eps": eps, "min_samples": min_s,
                               "n_clusters": n_clusters, "n_outliers": n_outliers}
        else:
            print(f"  {eps:>6.1f}  {min_s:>6}  {n_clusters:>8}  "
                  f"{n_outliers:>10}  {'—':>12}  (skipped)")

if best_db is not None:
    df_feat["dbscan_cluster"] = best_db
    df_dbscan_out = df_feat[ID_COLS + ["dbscan_cluster"]].copy()
    df_dbscan_out["is_outlier"] = (best_db == -1).astype(int)
    df_dbscan_out["is_dom"]     = df_feat["dep"].isin(DOM_DEPS).astype(int)
    df_dbscan_out.to_csv("data/clustering/dbscan_labels.csv", index=False)
    print(f"\n  🎯 Best DBSCAN: eps={best_params['eps']}, "
          f"min_samples={best_params['min_samples']}, "
          f"clusters={best_params['n_clusters']}, "
          f"outliers={best_params['n_outliers']}, "
          f"silhouette={best_sil:.4f}")

    outlier_depts = df_feat[df_feat["dbscan_cluster"] == -1]["dep"].tolist()
    if "nom" in df_feat.columns:
        outlier_names = df_feat[df_feat["dbscan_cluster"] == -1]["nom"].tolist()
    else:
        outlier_names = outlier_depts
    print(f"  ✓ Outlier depts: {', '.join(sorted(outlier_names))}")
else:
    print("  ⚠️  No valid DBSCAN config found — try wider eps range")
    df_feat["dbscan_cluster"] = -1


# ============================================================
# STEP 7 — Cluster Profiles: Mean Feature per Cluster
# ============================================================

print("\n" + "=" * 60)
print("📋 STEP 7 — Cluster profiles (mean features per cluster)")
print("=" * 60)

profile_cols = FEATURE_COLS + ["population", "is_dom", "kmeans_cluster"]
df_profiles  = (
    df_feat[profile_cols]
    .groupby("kmeans_cluster")
    .agg(["mean", "std", "count"])
)
df_profiles.columns = ["_".join(c) for c in df_profiles.columns]
df_profiles = df_profiles.reset_index()
df_profiles.to_csv("data/clustering/cluster_profiles.csv", index=False)

# Human-readable profile table
print(f"\n  {'Feature':<30}", end="")
for _cl in sorted(df_feat["kmeans_cluster"].unique()):
    print(f"  Cluster {_cl:>2}", end="")
print()
print(f"  {'-'*70}")

for feat in FEATURE_COLS:
    print(f"  {feat:<30}", end="")
    for _cl in sorted(df_feat["kmeans_cluster"].unique()):
        _sub = df_feat[df_feat["kmeans_cluster"] == _cl][feat]
        print(f"  {_sub.mean():>10.2f}", end="")
    print()

print(f"\n  ✓ Profiles saved → data/clustering/cluster_profiles.csv")


# ============================================================
# STEP 8 — DOM Analysis
# Are overseas départements isolated in their own cluster?
# ============================================================

print("\n" + "=" * 60)
print("🌴 STEP 8 — DOM analysis")
print("=" * 60)

print(f"  DOM département cluster assignments (K-Means k={OPTIMAL_K}):")
if not df_feat_dom.empty and "kmeans_cluster" in df_feat_dom.columns:
    _dom_disp_cols = [c for c in ID_COLS + ["kmeans_cluster", "dbscan_cluster"]
                      if c in df_feat_dom.columns]
    df_dom = df_feat_dom[_dom_disp_cols].copy()
    for _, row in df_dom.iterrows():
        nm = row.get("nom", row["dep"])
        db_cl = int(row["dbscan_cluster"]) if "dbscan_cluster" in row.index else "N/A"
        print(f"    {row['dep']} {str(nm):<20}  K-Means={int(row['kmeans_cluster'])}  "
              f"DBSCAN={db_cl} {'(DOM — exclu density)' if db_cl==-2 else ''}")

    dom_clusters = df_dom["kmeans_cluster"].unique()
    metro_only   = df_feat["kmeans_cluster"].unique()
    dom_isolated = set(dom_clusters.tolist()) - set(metro_only.tolist())

    if dom_isolated:
        print(f"\n  ✓ DOM depts form their own cluster(s): {dom_isolated}")
    else:
        print(f"\n  ℹ️  DOM depts assigned to metro clusters: {sorted(int(x) for x in dom_clusters)}")
        print("  ✓ Nearest-centroid assignment — epidemiologically comparable profile")
else:
    print("  ⚠️  DOM features unavailable — skipped")


# ============================================================
# STEP 9 — Wave-Level Cluster Stability
# Did a dept's epidemic profile change across waves?
# ============================================================

print("\n" + "=" * 60)
print("🌊 STEP 9 — Wave-level cluster stability")
print("=" * 60)

df_master = pd.read_csv(
    "data/processed/master_dept_daily.csv",
    dtype={"dep": str}, low_memory=False
)
df_waves = pd.read_csv("data/processed/wave_periods.csv")

wave_cluster_rows = []

for _, wrow in df_waves.iterrows():
    _wnum  = int(wrow["wave"])
    _wlbl  = wrow["label"]
    _wdf   = df_master[df_master["wave"] == _wnum].copy()
    if _wdf.empty:
        continue

    # Aggregate same features per dept for this wave
    _agg = {}
    for _c in ["hosp_rate", "new_hosp_rate", "rea", "icu_pressure",
               "cfr", "tx_pos", "couv_complet", "R0"]:
        if _c in _wdf.columns:
            _agg[_c] = "mean"
    if not _agg:
        continue

    _wave_feat = _wdf.groupby("dep").agg(_agg).reset_index()
    _wave_feat = _wave_feat.dropna(thresh=max(2, len(_agg)//2))

    if len(_wave_feat) < 10:
        print(f"  ⚠️  Wave {_wnum}: only {len(_wave_feat)} depts with data — skipped")
        continue

    _wave_X   = _wave_feat.drop(columns=["dep"]).fillna(0)
    _wave_X_s = RobustScaler().fit_transform(_wave_X)

    # Use same OPTIMAL_K as global clustering
    _km_w = KMeans(n_clusters=OPTIMAL_K, random_state=RANDOM_STATE,
                   n_init=20, max_iter=500)
    _wave_labels = _km_w.fit_predict(_wave_X_s)
    _sil_w = silhouette_score(_wave_X_s, _wave_labels) if len(set(_wave_labels)) > 1 else 0

    _wave_feat["wave"]          = _wnum
    _wave_feat["wave_label"]    = _wlbl
    _wave_feat["wave_cluster"]  = _wave_labels
    wave_cluster_rows.append(_wave_feat[["dep", "wave", "wave_label", "wave_cluster"]])

    print(f"  Wave {_wnum}: {len(_wave_feat):>3} depts | "
          f"silhouette={_sil_w:.4f} | label={_wlbl}")

if wave_cluster_rows:
    df_wave_clusters = pd.concat(wave_cluster_rows, ignore_index=True)
    df_wave_clusters.to_csv("data/clustering/wave_cluster_labels.csv", index=False)

    # Cluster stability: how many depts stayed in the same relative cluster across waves?
    df_pivot = df_wave_clusters.pivot(index="dep", columns="wave", values="wave_cluster")
    df_pivot = df_pivot.dropna()
    # Measure consistency as mode == all values
    stable = df_pivot.apply(lambda r: r.nunique() == 1, axis=1).sum()
    print(f"\n  ✓ Wave-stable depts (same cluster all waves): "
          f"{stable}/{len(df_pivot)} ({stable/len(df_pivot)*100:.1f}%)")
    df_pivot.to_csv("data/clustering/wave_cluster_pivot.csv")


# ============================================================
# STEP 10 — Save Global Labels + Interpretability Report
# ============================================================

print("\n" + "=" * 60)
print("💾 STEP 10 — Save global labels + interpretability report")
print("=" * 60)

# Master clustering output — one row per dept, all cluster labels (metro + DOM)
df_global = df_feat[ID_COLS + ["kmeans_cluster", "dbscan_cluster"]].copy()
df_global["is_dom"] = 0

# Re-add DOM with their nearest-centroid assignment
if not df_feat_dom.empty and "kmeans_cluster" in df_feat_dom.columns:
    _dom_id_cols = [c for c in ID_COLS if c in df_feat_dom.columns]
    _dom_rows = df_feat_dom[_dom_id_cols + ["kmeans_cluster", "dbscan_cluster"]].copy()
    _dom_rows["is_dom"] = 1
    df_global = pd.concat([df_global, _dom_rows], ignore_index=True)
    print(f"  ✓ DOM réintégrés: {len(_dom_rows)} depts ajoutés")

df_global = df_global.merge(df_pca[["dep", "pc1", "pc2", "pc3"]], on="dep", how="left")

# Attach key profile metrics for Phase 5 map tooltip (metro + DOM)
_feat_all = pd.concat(
    [df_feat, df_feat_dom] if not df_feat_dom.empty else [df_feat],
    ignore_index=True
)
for _col in ["hosp_rate_mean", "tx_pos_mean", "couv_complet_max", "cfr_mean"]:
    if _col in _feat_all.columns:
        _metric_map = _feat_all.set_index("dep")[_col]
        df_global[_col] = df_global["dep"].map(_metric_map)

df_global.to_csv("data/clustering/global_cluster_labels.csv", index=False)
print(f"  ✓ Global labels saved → data/clustering/global_cluster_labels.csv")

# ── Interpretability Report ──────────────────────────────────
report_lines = [
    "=" * 60,
    "PHASE 3 CLUSTERING — INTERPRETABILITY REPORT",
    "=" * 60,
    f"Feature matrix     : 101 depts × {len(FEATURE_COLS)} features",
    f"Scaler             : RobustScaler (median + IQR)",
    f"PCA 90% variance   : {n_90} components",
    f"PCA 95% variance   : {n_95} components",
    "",
    "K-MEANS",
    f"  Optimal k        : {OPTIMAL_K}",
    f"  Silhouette       : {sil_final:.4f}",
    f"  Calinski-Harabász: {ch_final:.1f}",
    f"  Davies-Bouldin   : {db_final:.4f}",
    "",
    "DBSCAN",
    f"  Best eps         : {best_params.get('eps', 'N/A')}",
    f"  Best min_samples : {best_params.get('min_samples', 'N/A')}",
    f"  Clusters found   : {best_params.get('n_clusters', 'N/A')}",
    f"  Outliers (noise) : {best_params.get('n_outliers', 'N/A')}",
    f"  Silhouette       : {best_sil:.4f}",
    "",
    "PCA LOADINGS (top 5 PC1 drivers):",
]
for feat, row in df_loadings.head(5).iterrows():
    report_lines.append(f"  {feat:<30} PC1={row['PC1']:+.4f}  PC2={row['PC2']:+.4f}")

report_lines += [
    "",
    "CLUSTER PROFILES (K-Means means):",
]
for _cl in sorted(df_feat["kmeans_cluster"].unique()):
    _sub = df_feat[df_feat["kmeans_cluster"] == _cl]
    _n   = len(_sub)
    _doms = [d for d in _sub["dep"].tolist() if d in DOM_DEPS]
    report_lines.append(f"  Cluster {_cl} ({_n} depts{', includes DOM' if _doms else ''}):")
    for feat in ["hosp_rate_mean", "tx_pos_mean", "couv_complet_max", "cfr_mean", "R0_mean"]:
        if feat in df_feat.columns:
            report_lines.append(f"    {feat:<30} {_sub[feat].mean():.4f}")

report_lines += ["", "=" * 60,
                 "🚀 Phase 3 complete — ready for Phase 4: ML Models"]

report_txt = "\n".join(report_lines)
with open("data/clustering/clustering_report.txt", "w", encoding="utf-8") as _f:
    _f.write(report_txt)

print(report_txt)

# ── File Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📁 PHASE 3 OUTPUT FILES")
print("=" * 60)

_P3_FILES = [
    "data/clustering/pca_components.csv",
    "data/clustering/pca_loadings.csv",
    "data/clustering/kmeans_metrics.csv",
    "data/clustering/kmeans_labels.csv",
    "data/clustering/dbscan_labels.csv",
    "data/clustering/cluster_profiles.csv",
    "data/clustering/global_cluster_labels.csv",
    "data/clustering/wave_cluster_labels.csv",
    "data/clustering/wave_cluster_pivot.csv",
    "data/clustering/clustering_report.txt",
]
total_kb = 0
for _f in _P3_FILES:
    if os.path.exists(_f):
        _kb = os.path.getsize(_f) / 1024
        total_kb += _kb
        print(f"  ✓ {_f:<55} {_kb:>7.1f} KB")
    else:
        print(f"  ✗ {_f:<55} NOT CREATED")

print(f"\n  💾 Phase 3 output size: {total_kb/1024:.2f} MB")
print("\n🚀 Phase 3 complete — ready for Phase 4: ML Predictive Models")