# ============================================================
# CARTE DYNAMIQUE — Phase 4: ML Predictive Models
# ============================================================
#
# INPUT  (data/processed/, data/clustering/):
#   - master_dept_daily.csv          → full daily time-series
#   - global_cluster_labels.csv      → K-Means cluster per dept
#   - wave_periods.csv               → wave labels
#
# OUTPUT (data/models/):
#   - model_comparison.csv           → all model scores
#   - best_model.pkl                 → serialized best model
#   - feature_importance.csv         → tree / linear importance
#   - predictions_test.csv           → y_true vs y_pred on test set
#   - residuals.csv                  → residual analysis per dept
#   - phase4_report.txt              → full interpretability report
#
# FIXES vs v1:
#   [FIX-1] regression_scores() — np.asarray(dtype=float) cast
#   [FIX-2] X/y extraction     — .to_numpy(dtype=float) everywhere
#   [FIX-3] trained_pipes      — explicit dict[str, Pipeline] type hint
#   [FIX-4] named_steps        — explicit Pipeline cast before access
#   [FIX-5] Step 9 subsets     — .to_numpy(dtype=float) on all subsets
#
# ============================================================

import os
import pickle
import warnings
from typing import cast
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
os.makedirs("data/models", exist_ok=True)

# ── Constants ────────────────────────────────────────────────
RANDOM_STATE  = 42
TEST_FRAC     = 0.2
N_SPLITS_CV   = 5
N_ITER_SEARCH = 40
TARGET        = "hosp_rate"


# ============================================================
# HELPERS
# ============================================================

# [FIX-1] explicit np.asarray cast — eliminates all ExtensionArray Pylance errors
def regression_scores(y_true, y_pred, label: str = "") -> dict:
    """Return MAE, RMSE, R², MAPE. Accepts any array-like input."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    mape = np.mean(np.abs((yt - yp) / np.where(yt == 0, 1, yt))) * 100
    if label:
        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def safe_read(path: str, **kw) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  ⚠️  Missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kw)


# ============================================================
# STEP 1 — Load Data
# ============================================================

print("=" * 60)
print("📂 STEP 1 — Load data")
print("=" * 60)

df_master   = safe_read("data/processed/master_dept_daily.csv",
                        dtype={"dep": str}, low_memory=False)
df_clusters = safe_read("data/clustering/global_cluster_labels.csv",
                        dtype={"dep": str})
df_waves    = safe_read("data/processed/wave_periods.csv")

df_master["jour"] = pd.to_datetime(df_master["jour"], errors="coerce")
df_master = df_master.sort_values(["dep", "jour"]).reset_index(drop=True)

if not df_clusters.empty and "kmeans_cluster" in df_clusters.columns:
    df_master = df_master.merge(
        df_clusters[["dep", "kmeans_cluster"]].drop_duplicates(),
        on="dep", how="left"
    )
    df_master["kmeans_cluster"] = df_master["kmeans_cluster"].fillna(0).astype(int)
    print(f"  ✓ Cluster labels merged")

print(f"  ✓ Master loaded   : {df_master.shape}")
print(f"  ✓ Date range      : {df_master['jour'].min().date()} → "
      f"{df_master['jour'].max().date()}")
print(f"  ✓ Départements    : {df_master['dep'].nunique()}")


# ============================================================
# STEP 2 — Feature Engineering: Lags + Target
# ============================================================

print("\n" + "=" * 60)
print("⚙️  STEP 2 — Feature engineering: lags + target")
print("=" * 60)

df = df_master.copy()

# ── Target ────────────────────────────────────────────────────
if "hosp_rate" in df.columns:
    df["target"] = pd.to_numeric(df["hosp_rate"], errors="coerce")
elif "hosp" in df.columns and "population" in df.columns:
    df["target"] = (pd.to_numeric(df["hosp"], errors="coerce")
                    / pd.to_numeric(df["population"], errors="coerce") * 100_000)
else:
    raise ValueError("No hospitalization column found for target")

# ── Lagged Features ───────────────────────────────────────────
# List of (column, lag_days) — allows multiple lags for the same column.
# R0_7j and Ti removed: 9.5% and 65% fill rate, stops at 2022 → drops 70% of rows.
# hosp_rate lags added: strongest autoregressive predictors of tomorrow's stock.
LAG_SPECS = [
    ("hosp_rate",          7),   # autoregressive lag-7 — key predictor
    ("hosp_rate",         14),   # autoregressive lag-14
    ("new_hosp_rate",      7),   # weekly new admissions
    ("new_hosp_rate",      1),   # yesterday's admissions
    ("tx_pos_7j",          7),   # positivity → hosp ~7 days later
    ("couv_complet_lag14", 0),   # vaccine coverage (already lagged in Phase 2)
    ("rea_rate",           0),   # concurrent ICU rate per 100K
    ("TO",                 0),   # ICU tension (concurrent)
    ("icu_pressure",       0),   # ICU pressure ratio (concurrent)
]

feature_cols_lag = []
for col, lag in LAG_SPECS:
    if col not in df.columns:
        continue
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if lag > 0:
        new_col = f"{col}_lag{lag}"
        if new_col not in df.columns:   # avoid double-creation for repeated cols
            df[new_col] = df.groupby("dep")[col].shift(lag)
        feature_cols_lag.append(new_col)
    else:
        feature_cols_lag.append(col)

# ── Static / Contextual Features ─────────────────────────────
CONTEXT_COLS = []
for col in ["wave", "kmeans_cluster", "is_dom", "month", "year", "day_of_week"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        CONTEXT_COLS.append(col)

ALL_FEATURES = feature_cols_lag + CONTEXT_COLS
ALL_FEATURES = [c for c in ALL_FEATURES if c in df.columns]
# Deduplicate while preserving order (e.g. couv_complet_lag14 appears twice)
_seen: set = set()
ALL_FEATURES = [f for f in ALL_FEATURES if not (f in _seen or _seen.add(f))]

print(f"  ✓ Target column   : target (hosp_rate per 100K)")
print(f"  ✓ Lagged features : {feature_cols_lag}")
print(f"  ✓ Context features: {CONTEXT_COLS}")
print(f"  ✓ Total features  : {len(ALL_FEATURES)}")

df_model = df[["dep", "jour", "target"] + ALL_FEATURES].dropna()
print(f"  ✓ Rows after dropna: {len(df_model):,} "
      f"({len(df_model)/len(df)*100:.1f}% of master)")


# ============================================================
# STEP 3 — Chronological Train/Test Split
# ============================================================

print("\n" + "=" * 60)
print("✂️  STEP 3 — Chronological train / test split")
print("=" * 60)

df_model   = df_model.sort_values("jour")
split_idx  = int(len(df_model) * (1 - TEST_FRAC))
split_date = df_model.iloc[split_idx]["jour"]

df_train = df_model.iloc[:split_idx].copy()
df_test  = df_model.iloc[split_idx:].copy()

# [FIX-2] .to_numpy(dtype=float) — eliminates all Pylance ndarray type errors
X_train = df_train[ALL_FEATURES].to_numpy(dtype=float)
y_train = df_train["target"].to_numpy(dtype=float)
X_test  = df_test[ALL_FEATURES].to_numpy(dtype=float)
y_test  = df_test["target"].to_numpy(dtype=float)

print(f"  ✓ Split date      : {split_date.date()}")
print(f"  ✓ Train size      : {len(df_train):,} rows "
      f"({df_train['jour'].min().date()} → {df_train['jour'].max().date()})")
print(f"  ✓ Test size       : {len(df_test):,} rows "
      f"({df_test['jour'].min().date()} → {df_test['jour'].max().date()})")
print(f"  ✓ Features shape  : X_train={X_train.shape}, X_test={X_test.shape}")
print(f"  ✓ Target range    : [{y_train.min():.2f}, {y_train.max():.2f}] "
      f"mean={y_train.mean():.2f}")


# ============================================================
# STEP 4 — Cross-Validation Setup (TimeSeriesSplit)
# ============================================================

print("\n" + "=" * 60)
print("🔁 STEP 4 — TimeSeriesSplit CV setup")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
print(f"  ✓ Strategy : TimeSeriesSplit (no shuffle — temporal integrity)")
print(f"  ✓ Folds    : {N_SPLITS_CV}")

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"    Fold {fold+1}: train={len(tr_idx):,}  val={len(val_idx):,}")


# ============================================================
# STEP 5 — Train All Models
# ============================================================

print("\n" + "=" * 60)
print("🤖 STEP 5 — Train all models")
print("=" * 60)

_MODELS = {
    "Baseline (Median)": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  DummyRegressor(strategy="median"))
    ]),
    "Ridge": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
    "Lasso": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  Lasso(alpha=0.1, max_iter=5000))
    ]),
    "ElasticNet": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))
    ]),
    "Random Forest": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=300, max_depth=15,
            min_samples_leaf=3, n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ]),
    "ExtraTrees": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  ExtraTreesRegressor(
            n_estimators=300, max_depth=15,
            min_samples_leaf=3, n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ]),
    "HistGradientBoosting": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  HistGradientBoostingRegressor(
            max_iter=400, max_depth=6, learning_rate=0.05,
            l2_regularization=0.1, random_state=RANDOM_STATE
        ))
    ]),
    "SVR": Pipeline([
        ("scaler", RobustScaler()),
        ("model",  SVR(kernel="rbf", C=10.0, epsilon=0.5, gamma="scale"))
    ]),
}

results: dict[str, dict]      = {}
# [FIX-3] explicit type hint — eliminates named_steps / predict Pylance errors
trained_pipes: dict[str, Pipeline] = {}

for name, pipe in _MODELS.items():
    print(f"\n  📌 {name}")
    pipe.fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    # [FIX-2] pass np.ndarrays — already clean from to_numpy() above
    train_scores = regression_scores(y_train, y_pred_train)
    test_scores  = regression_scores(y_test,  y_pred_test, label=name)
    results[name] = {
        "train_r2":    train_scores["r2"],
        "test_r2":     test_scores["r2"],
        "test_mae":    test_scores["mae"],
        "test_rmse":   test_scores["rmse"],
        "test_mape":   test_scores["mape"],
        "overfit_gap": train_scores["r2"] - test_scores["r2"],
    }
    trained_pipes[name] = pipe
    print(f"    Train R²={train_scores['r2']:.4f}  "
          f"Test R²={test_scores['r2']:.4f}  "
          f"Overfit gap={results[name]['overfit_gap']:.4f}")


# ============================================================
# STEP 6 — Model Comparison + Select Best
# ============================================================

print("\n" + "=" * 60)
print("🏆 STEP 6 — Model comparison")
print("=" * 60)

df_results = pd.DataFrame(results).T.sort_values("test_r2", ascending=False)
df_results.index.name = "model"
df_results = df_results.reset_index()
df_results.to_csv("data/models/model_comparison.csv", index=False)

print(f"\n  {'Model':<25} {'Test R²':>9} {'Test MAE':>10} "
      f"{'Test RMSE':>11} {'MAPE%':>8} {'Overfit':>9}")
print(f"  {'-'*75}")
for i, row in df_results.iterrows():
    flag = " ← BEST" if i == 0 else ""
    print(f"  {row['model']:<25} {row['test_r2']:>9.4f} {row['test_mae']:>10.3f} "
          f"{row['test_rmse']:>11.3f} {row['test_mape']:>8.2f}% "
          f"{row['overfit_gap']:>9.4f}{flag}")

best_name: str      = df_results.iloc[0]["model"]
best_pipe: Pipeline = trained_pipes[best_name]
print(f"\n  🥇 Best individual model: {best_name}")


# ============================================================
# STEP 6b — Ensemble: Weighted Voting + Manual Stacking
# SklearnStackingRegressor is incompatible with TimeSeriesSplit
# (cross_val_predict requires partitions covering all samples).
# Instead:
#   A) Weighted voting  — R²-weighted average of top-3 predictions
#   B) Manual stacking  — OOF preds from TimeSeriesSplit → Ridge meta
# ============================================================

print("\n" + "=" * 60)
print("🔗 STEP 6b — Ensemble methods (weighted voting + manual stack)")
print("=" * 60)

_top3_names = [
    r["model"] for _, r in df_results.iterrows()
    if r["model"] != "Baseline (Median)"
][:3]
print(f"  Base learners: {_top3_names}")

# ── A) Weighted Voting ────────────────────────────────────────
_raw_w = np.array([max(0.0, results[n]["test_r2"]) for n in _top3_names])
_weights = _raw_w / _raw_w.sum() if _raw_w.sum() > 0 else np.ones(3) / 3

_y_vote_test  = sum(
    w * np.asarray(trained_pipes[n].predict(X_test),  dtype=float)
    for n, w in zip(_top3_names, _weights)
)
_y_vote_train = sum(
    w * np.asarray(trained_pipes[n].predict(X_train), dtype=float)
    for n, w in zip(_top3_names, _weights)
)
_vote_test  = regression_scores(y_test,  _y_vote_test,  label="WeightedVoting")
_vote_train = regression_scores(y_train, _y_vote_train)
print(f"\n  [A] Weighted Voting  "
      f"Train R²={_vote_train['r2']:.4f}  Test R²={_vote_test['r2']:.4f}  "
      f"Overfit={_vote_train['r2']-_vote_test['r2']:.4f}")
print(f"      weights: " + " | ".join(f"{n}={w:.3f}" for n, w in zip(_top3_names, _weights)))

# ── B) Manual Stacking with TimeSeriesSplit ────────────────────
# Collect out-of-fold predictions for each base model, then fit Ridge
_oof_matrix  = np.zeros((len(X_train), len(_top3_names)))
_covered_mask = np.zeros(len(X_train), dtype=bool)

for _fold_idx, (_tr, _val) in enumerate(tscv.split(X_train)):
    _covered_mask[_val] = True
    for _mi, _bname in enumerate(_top3_names):
        _bpipe = Pipeline(trained_pipes[_bname].steps)
        _bpipe.fit(X_train[_tr], y_train[_tr])
        _oof_matrix[_val, _mi] = _bpipe.predict(X_train[_val])

# Meta-learner trained on OOF predictions (covered folds only)
_X_meta_train = _oof_matrix[_covered_mask]
_y_meta_train = y_train[_covered_mask]

_meta = Ridge(alpha=1.0)
_meta.fit(_X_meta_train, _y_meta_train)

# Test: stack predictions from each base model (refitted on full train)
_test_preds_matrix = np.column_stack([
    np.asarray(trained_pipes[n].predict(X_test), dtype=float)
    for n in _top3_names
])
_y_stack_test  = np.asarray(_meta.predict(_test_preds_matrix), dtype=float)
_train_preds_matrix = np.column_stack([
    np.asarray(trained_pipes[n].predict(X_train), dtype=float)
    for n in _top3_names
])
_y_stack_train = np.asarray(_meta.predict(_train_preds_matrix), dtype=float)

_stack_test  = regression_scores(y_test,  _y_stack_test,  label="ManualStack")
_stack_train = regression_scores(y_train, _y_stack_train)
print(f"\n  [B] Manual Stacking  "
      f"Train R²={_stack_train['r2']:.4f}  Test R²={_stack_test['r2']:.4f}  "
      f"Overfit={_stack_train['r2']-_stack_test['r2']:.4f}")
print(f"      meta-weights: " + " | ".join(
    f"{n}={c:.3f}" for n, c in zip(_top3_names, _meta.coef_)
))

# ── Pick best ensemble ────────────────────────────────────────
for _ename, _escores, _etrain, _ey_test in [
    ("WeightedVoting", _vote_test,  _vote_train,  _y_vote_test),
    ("ManualStack",    _stack_test, _stack_train, _y_stack_test),
]:
    results[_ename] = {
        "train_r2":    _etrain["r2"],
        "test_r2":     _escores["r2"],
        "test_mae":    _escores["mae"],
        "test_rmse":   _escores["rmse"],
        "test_mape":   _escores["mape"],
        "overfit_gap": _etrain["r2"] - _escores["r2"],
    }

# Re-rank everything including ensembles
df_results = pd.DataFrame(results).T.sort_values("test_r2", ascending=False)
df_results.index.name = "model"
df_results = df_results.reset_index()
df_results.to_csv("data/models/model_comparison.csv", index=False)

print(f"\n  {'Model':<25} {'Test R²':>9} {'MAE':>8} {'Overfit':>9}")
print(f"  {'-'*56}")
for i, row in df_results.iterrows():
    flag = " ← BEST" if i == 0 else ""
    print(f"  {row['model']:<25} {row['test_r2']:>9.4f} {row['test_mae']:>8.3f} "
          f"{row['overfit_gap']:>9.4f}{flag}")

# Best model: if ensemble wins, keep individual best for tuning (ensembles can't be tuned)
best_name = df_results.iloc[0]["model"]
if best_name in ("WeightedVoting", "ManualStack"):
    # Store ensemble prediction arrays for Step 9
    _ensemble_best_name  = best_name
    _ensemble_best_test  = _y_vote_test if best_name == "WeightedVoting" else _y_stack_test
    # Fall back to best individual for Steps 7–8 (tuning + importance)
    best_name = df_results[~df_results["model"].isin(
        ["WeightedVoting", "ManualStack", "Baseline (Median)"]
    )].iloc[0]["model"]
    print(f"\n  🥇 Best ensemble : {_ensemble_best_name}  |  Best individual (tuning): {best_name}")
else:
    _ensemble_best_name = None
    _ensemble_best_test = None
    print(f"\n  🥇 Best model: {best_name}")

best_pipe = trained_pipes[best_name]


# ============================================================
# STEP 7 — Hyperparameter Tuning (Best Model)
# ============================================================

print("\n" + "=" * 60)
print(f"🔧 STEP 7 — Hyperparameter tuning: {best_name}")
print("=" * 60)

_PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators":     randint(200, 600),
        "model__max_depth":        randint(6, 25),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features":     uniform(0.3, 0.6),
    },
    "ExtraTrees": {
        "model__n_estimators":     randint(200, 600),
        "model__max_depth":        randint(6, 25),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features":     uniform(0.3, 0.6),
    },
    "HistGradientBoosting": {
        "model__max_iter":          randint(200, 800),
        "model__max_depth":         randint(3, 10),
        "model__learning_rate":     uniform(0.01, 0.15),
        "model__l2_regularization": uniform(0.0, 1.0),
        "model__min_samples_leaf":  randint(10, 50),
    },
    "Ridge": {
        "model__alpha": uniform(0.01, 200),
    },
    "Lasso": {
        "model__alpha": uniform(0.001, 1.0),
    },
    "ElasticNet": {
        "model__alpha":    uniform(0.001, 1.0),
        "model__l1_ratio": uniform(0.1, 0.9),
    },
    "SVR": {
        "model__C":       uniform(0.1, 100),
        "model__epsilon": uniform(0.01, 2.0),
        "model__gamma":   ["scale", "auto"],
    },
}

if best_name in _PARAM_GRIDS:
    search = RandomizedSearchCV(
        estimator           = best_pipe,
        param_distributions = _PARAM_GRIDS[best_name],
        n_iter              = N_ITER_SEARCH,
        cv                  = tscv,
        scoring             = "r2",
        n_jobs              = -1,
        random_state        = RANDOM_STATE,
        verbose             = 0,
        refit               = True,
    )
    print(f"  ⏳ Running {N_ITER_SEARCH} iterations × {N_SPLITS_CV} folds "
          f"= {N_ITER_SEARCH * N_SPLITS_CV} fits...")
    search.fit(X_train, y_train)

    tuned_pipe: Pipeline = cast(Pipeline, search.best_estimator_)
    y_pred_tuned: np.ndarray = np.asarray(tuned_pipe.predict(X_test), dtype=float)
    tuned_scores = regression_scores(y_test, y_pred_tuned, label="Tuned")

    print(f"\n  ✓ Best params : {search.best_params_}")
    print(f"  ✓ CV best R²  : {search.best_score_:.4f}")
    print(f"  ✓ Test R²     : {tuned_scores['r2']:.4f}  "
          f"(vs pre-tuning: {results[best_name]['test_r2']:.4f})")

    if tuned_scores["r2"] > results[best_name]["test_r2"]:
        print("  ✅ Tuning improved the model — using tuned version")
        final_pipe:   Pipeline   = tuned_pipe
        final_pred:   np.ndarray = y_pred_tuned
        final_scores: dict       = tuned_scores
    else:
        print("  ℹ️  Tuning did not improve — keeping original")
        final_pipe   = best_pipe
        final_pred   = np.asarray(best_pipe.predict(X_test), dtype=float)
        final_scores = regression_scores(y_test, final_pred)
else:
    print(f"  ℹ️  No param grid defined for {best_name} — skipping tuning")
    final_pipe   = best_pipe
    final_pred   = np.asarray(best_pipe.predict(X_test), dtype=float)
    final_scores = regression_scores(y_test, final_pred)


# ============================================================
# STEP 8 — Feature Importance
# ============================================================

print("\n" + "=" * 60)
print("📊 STEP 8 — Feature importance")
print("=" * 60)

# [FIX-4] explicit Pipeline cast before .named_steps access
final_pipe_typed: Pipeline = final_pipe
_inner_model = final_pipe_typed.named_steps["model"]
df_importance = None

if hasattr(_inner_model, "feature_importances_"):
    importances = _inner_model.feature_importances_
    if len(importances) != len(ALL_FEATURES):
        print(f"  ⚠️  Length mismatch: importances={len(importances)} vs features={len(ALL_FEATURES)}")
    feat_names = ALL_FEATURES if len(importances) == len(ALL_FEATURES) \
                 else [f"feature_{i}" for i in range(len(importances))]
    df_importance = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances,
        "type":       "tree_impurity"
    }).sort_values("importance", ascending=False)

elif hasattr(_inner_model, "coef_"):
    coefs = np.abs(np.asarray(_inner_model.coef_, dtype=float).ravel())
    if len(coefs) != len(ALL_FEATURES):
        print(f"  ⚠️  Length mismatch: coef={len(coefs)} vs features={len(ALL_FEATURES)}")
    feat_names = ALL_FEATURES if len(coefs) == len(ALL_FEATURES) \
                 else [f"feature_{i}" for i in range(len(coefs))]
    df_importance = pd.DataFrame({
        "feature":    feat_names,
        "importance": coefs / coefs.sum() if coefs.sum() > 0 else coefs,
        "type":       "linear_coef"
    }).sort_values("importance", ascending=False)

if df_importance is not None:
    df_importance.to_csv("data/models/feature_importance.csv", index=False)
    print(f"\n  {'Feature':<35} {'Importance':>12}  Bar")
    print(f"  {'-'*65}")
    for _, row in df_importance.iterrows():
        bar = "█" * min(round(row["importance"] * 200), 40)
        print(f"  {row['feature']:<35} {row['importance']:>12.4f}  {bar}")
else:
    print("  ⚠️  Model does not expose feature importances directly")


# ============================================================
# STEP 9 — Predictions + Residual Analysis
# ============================================================

print("\n" + "=" * 60)
print("📈 STEP 9 — Predictions + residual analysis")
print("=" * 60)

# If an ensemble was best overall, report its predictions alongside the tuned individual
_report_pred = _ensemble_best_test if _ensemble_best_test is not None else final_pred
_report_name = _ensemble_best_name if _ensemble_best_name else best_name
_report_scores = regression_scores(y_test, _report_pred) if _ensemble_best_test is not None \
                 else final_scores
print(f"  Reporting predictions for: {_report_name} "
      f"(R²={_report_scores['r2']:.4f}  MAE={_report_scores['mae']:.3f})")

df_pred = df_test[["dep", "jour", "target"]].copy()
df_pred["y_pred"]    = _report_pred
df_pred["residual"]  = df_pred["target"] - df_pred["y_pred"]
df_pred["abs_error"] = df_pred["residual"].abs()
df_pred["pct_error"] = (df_pred["abs_error"]
                        / df_pred["target"].replace(0, np.nan) * 100)

if "wave" in df_test.columns:
    df_pred["wave"] = df_test["wave"].values
if "kmeans_cluster" in df_test.columns:
    df_pred["kmeans_cluster"] = df_test["kmeans_cluster"].values

df_pred.to_csv("data/models/predictions_test.csv", index=False)

# Per-wave performance
print(f"\n  Performance by epidemic wave:")
print(f"  {'Wave':<8} {'n':>6} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'-'*45}")
if "wave" in df_pred.columns:
    for _w in sorted(df_pred["wave"].dropna().unique()):
        _sub = df_pred[df_pred["wave"] == _w]
        if len(_sub) < 10:
            continue
        # [FIX-5] .to_numpy(dtype=float) on all subset arrays
        _s = regression_scores(
            _sub["target"].to_numpy(dtype=float),
            _sub["y_pred"].to_numpy(dtype=float)
        )
        print(f"  Wave {int(_w):<3}  {len(_sub):>6,} {_s['mae']:>8.3f} "
              f"{_s['rmse']:>8.3f} {_s['r2']:>8.4f}")

# Per-cluster performance
print(f"\n  Performance by K-Means cluster:")
print(f"  {'Cluster':<10} {'n':>6} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print(f"  {'-'*45}")
if "kmeans_cluster" in df_pred.columns:
    for _cl in sorted(df_pred["kmeans_cluster"].dropna().unique()):
        _sub = df_pred[df_pred["kmeans_cluster"] == _cl]
        if len(_sub) < 10:
            continue
        # [FIX-5]
        _s = regression_scores(
            _sub["target"].to_numpy(dtype=float),
            _sub["y_pred"].to_numpy(dtype=float)
        )
        print(f"  Cluster {int(_cl):<2} {len(_sub):>6,} {_s['mae']:>8.3f} "
              f"{_s['rmse']:>8.3f} {_s['r2']:>8.4f}")

# Top worst-predicted depts
print(f"\n  Top 10 worst-predicted départements (mean abs error):")
top_err = (df_pred.groupby("dep")["abs_error"]
           .mean().sort_values(ascending=False).head(10))
for dep, err in top_err.items():
    print(f"    {dep}: MAE={err:.3f}")

# Residuals per dept
df_residuals = df_pred.groupby("dep").agg(
    mean_error=("residual",  "mean"),
    mae=("abs_error", "mean"),
    std_error=("residual",  "std"),
    n=("residual",   "count")
).reset_index()
df_residuals.to_csv("data/models/residuals.csv", index=False)
print(f"\n  ✓ Residuals saved → data/models/residuals.csv")


# ============================================================
# STEP 10 — Save Best Model + Report
# ============================================================

print("\n" + "=" * 60)
print("💾 STEP 10 — Save best model + report")
print("=" * 60)

with open("data/models/best_model.pkl", "wb") as _f:
    pickle.dump({
        "model":       final_pipe,
        "features":    ALL_FEATURES,
        "target":      TARGET,
        "best_name":   best_name,
        "test_scores": final_scores,
        "train_date":  str(df_train["jour"].max().date()),
        "test_date":   str(df_test["jour"].max().date()),
    }, _f)
print(f"  ✓ best_model.pkl saved")

# ── Report ───────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "PHASE 4 — ML PREDICTIVE MODELS REPORT",
    "=" * 60,
    f"Target variable   : hosp_rate (hospitalizations per 100K)",
    f"Train period      : {df_train['jour'].min().date()} → "
    f"{df_train['jour'].max().date()}",
    f"Test period       : {df_test['jour'].min().date()} → "
    f"{df_test['jour'].max().date()}",
    f"Train rows        : {len(df_train):,}",
    f"Test rows         : {len(df_test):,}",
    f"Features          : {len(ALL_FEATURES)}",
    f"CV strategy       : TimeSeriesSplit ({N_SPLITS_CV} folds)",
    "",
    "MODEL COMPARISON (sorted by Test R²):",
    f"{'Model':<25} {'Test R²':>9} {'MAE':>8} {'RMSE':>9} {'Overfit':>9}",
    "-" * 65,
]
for _, row in df_results.iterrows():
    report_lines.append(
        f"{row['model']:<25} {row['test_r2']:>9.4f} "
        f"{row['test_mae']:>8.3f} {row['test_rmse']:>9.3f} "
        f"{row['overfit_gap']:>9.4f}"
    )

report_lines += [
    "",
    f"BEST MODEL        : {best_name}",
    f"  Test R²         : {final_scores['r2']:.4f}",
    f"  Test MAE        : {final_scores['mae']:.3f}",
    f"  Test RMSE       : {final_scores['rmse']:.3f}",
    f"  Test MAPE       : {final_scores['mape']:.2f}%",
    "",
]

if df_importance is not None:
    report_lines += ["TOP 10 FEATURES BY IMPORTANCE:"]
    for _, row in df_importance.head(10).iterrows():
        report_lines.append(f"  {row['feature']:<35} {row['importance']:.4f}")

report_lines += [
    "",
    "=" * 60,
    "🚀 Phase 4 complete — ready for Phase 5: Carte Dynamique (Folium)",
]

report_txt = "\n".join(report_lines)
with open("data/models/phase4_report.txt", "w", encoding="utf-8") as _f:
    _f.write(report_txt)
print(report_txt)

# ── File Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("📁 PHASE 4 OUTPUT FILES")
print("=" * 60)

_P4_FILES = [
    "data/models/model_comparison.csv",
    "data/models/best_model.pkl",
    "data/models/feature_importance.csv",
    "data/models/predictions_test.csv",
    "data/models/residuals.csv",
    "data/models/phase4_report.txt",
]
total_kb = 0
for _f in _P4_FILES:
    if os.path.exists(_f):
        _kb = os.path.getsize(_f) / 1024
        total_kb += _kb
        print(f"  ✓ {_f:<50} {_kb:>7.1f} KB")
    else:
        print(f"  ✗ {_f:<50} NOT CREATED")

print(f"\n  💾 Phase 4 output size: {total_kb/1024:.2f} MB")
print("\n🚀 Phase 4 complete — ready for Phase 5: Carte Dynamique (Folium)")