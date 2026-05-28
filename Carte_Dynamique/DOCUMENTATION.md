# Documentation technique — Pipeline COVID-19 France

> Pipeline d'analyse épidémiologique pour les 101 départements français (métropole + DOM),
> couvrant la période janvier 2020 – mars 2023.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Structure des fichiers](#2-structure-des-fichiers)
3. [Phase 1 — Acquisition des données](#3-phase-1--acquisition-des-données)
4. [Phase 2 — Prétraitement et feature engineering](#4-phase-2--prétraitement-et-feature-engineering)
5. [Phase 3 — Clustering épidémiologique](#5-phase-3--clustering-épidémiologique)
6. [Phase 4 — Modèles de Machine Learning](#6-phase-4--modèles-de-machine-learning)
7. [Phase 5 — Tableau de bord interactif](#7-phase-5--tableau-de-bord-interactif)
8. [Conventions et décisions de conception](#8-conventions-et-décisions-de-conception)
9. [Dépendances](#9-dépendances)

---

## 1. Vue d'ensemble

### Objectif

Construire un pipeline end-to-end pour :
1. Collecter les données brutes COVID-19 depuis 17 sources officielles françaises
2. Produire une série temporelle quotidienne nettoyée par département
3. Identifier des groupes de départements au comportement épidémique similaire (clustering)
4. Prédire le taux d'hospitalisation à partir d'indicateurs épidémiques décalés (ML)
5. Visualiser l'ensemble dans un tableau de bord interactif piloté par curseur de date

### Périmètre géographique

- **96 départements** de France métropolitaine (01–95, incluant 2A/2B Corse)
- **5 DOM** : Guadeloupe (971), Martinique (972), Guyane (973), La Réunion (974), Mayotte (976)
- **Total : 101 départements**

### Période couverte

Janvier 2020 → mars 2023 (1 149 jours calendaires)

### Architecture séquentielle

```
Phase 1          Phase 2          Phase 3           Phase 4           Phase 5
─────────        ─────────        ─────────         ─────────         ─────────
17 sources   →   master_dept  →   PCA +         →   Régression    →   Dashboard
réseau           _daily.csv       K-Means           hosp_rate         HTML
                                  DBSCAN            prédictive        interactif
```

---

## 2. Structure des fichiers

```
src/
├── phase1_data_acquisition 1.py   # Acquisition réseau (~10-20 min)
├── phase2_preprocessing.py        # Feature engineering
├── phase3_clustering.py           # PCA + clustering
├── phase4_Ml-models.py            # Modèles ML + tuning
├── phase5_dashboard.py            # Tableau de bord HTML
└── data/
    ├── raw/                       # Données géographiques + brutes
    │   ├── departements.geojson   # Polygones métropole + DOM
    │   ├── regions.geojson        # Polygones régions
    │   ├── hospitalisations.csv   # Stock hosp/réa/dc/rad (SPF)
    │   ├── population_departements.csv  # INSEE 2020 (101 depts)
    │   ├── hospitalisations_age_region.csv
    │   ├── opencovid19_departements.csv # Comblement 2020
    │   ├── sivic_regional.csv     # Hosp × statut vaccinal (régional)
    │   └── finess_coords.csv      # GPS établissements de santé
    ├── labs/
    │   ├── sidep_old.csv          # SI-DEP hebdomadaire 2020–2022
    │   ├── sidep_new.csv          # SI-DEP quotidien 2022–2023
    │   ├── indicateurs_suivi.csv  # R0, Ti/100K, TO
    │   ├── variants.csv           # Variants COVID (DREES national)
    │   └── variants_dept.csv      # Variants par dept (optionnel)
    ├── vaccines/
    │   ├── vaccination_dept.csv   # VAC-SI couverture par dept
    │   ├── vaccination_age_dept.csv
    │   ├── vaccination_lag14.csv  # couv_complet décalée 14j
    │   ├── vaccination_lieu.csv   # Répartition par lieu
    │   ├── vaccination_injecteur.csv
    │   └── vaccination_marque.csv # Par marque de vaccin
    ├── locations/
    │   ├── centres_vaccination.csv
    │   └── centres_vac_par_dep.csv
    ├── processed/
    │   ├── master_dept_daily.csv      # Série principale (113 174 lignes)
    │   ├── master_dept_latest.csv     # Snapshot dernier point par dept
    │   ├── master_dept_geo.geojson    # GeoJSON enrichi
    │   ├── wave_periods.csv           # 7 vagues épidémiques
    │   ├── dept_clusters_features.csv # Matrice 101 depts × 15 features
    │   └── hospitalisations_normalized.csv
    ├── clustering/
    │   ├── pca_components.csv         # Coordonnées 2D/3D PCA
    │   ├── pca_loadings.csv           # Contributions features/composantes
    │   ├── kmeans_metrics.csv         # Coude + silhouette pour k=2..10
    │   ├── kmeans_labels.csv          # Label K-Means par dept
    │   ├── dbscan_labels.csv          # Label DBSCAN + outliers
    │   ├── cluster_profiles.csv       # Profil moyen par cluster
    │   ├── global_cluster_labels.csv  # Labels unifiés (métro + DOM)
    │   ├── wave_cluster_labels.csv    # Clustering par vague
    │   ├── wave_cluster_pivot.csv     # Stabilité inter-vagues
    │   └── clustering_report.txt      # Rapport interprétatif
    └── models/
        ├── best_model.pkl             # Pipeline sérialisé (pickle)
        ├── model_comparison.csv       # Scores tous modèles
        ├── feature_importance.csv     # Importances features
        ├── predictions_test.csv       # y_true vs y_pred
        ├── residuals.csv              # Résidus par département
        └── phase4_report.txt          # Rapport interprétatif
```

> **Répertoire de travail :** tous les scripts utilisent des chemins relatifs depuis `src/`.
> Exécuter depuis `src/` avec `python <script>.py`.

---

## 3. Phase 1 — Acquisition des données

**Script :** `phase1_data_acquisition 1.py`
**Durée estimée :** 10–20 minutes (réseau-intensif)

### 3.1 Fonctions utilitaires clés

#### `normalize_dep(series)`
Normalise les codes département en chaînes à 2 caractères avec zéro de tête.

| Entrée | Sortie | Cas particuliers |
|--------|--------|-----------------|
| `"1"`, `1`, `"01"` | `"01"` | — |
| `"2A"`, `"2B"` | `"2A"`, `"2B"` | Corse — conservés tels quels |
| `"971"` | `"971"` | DOM — 3 chiffres conservés |
| `"DEP-75"` | `"75"` | Préfixe `DEP-` supprimé |

#### `fetch_latest_datagouv_url(dataset_id, resource_name_contains)`
Interroge l'API data.gouv.fr pour résoudre l'URL la plus récente d'une ressource correspondant à un fragment de nom. Évite les URLs hardcodées qui deviennent obsolètes.

#### `apply_age_filter(df)`
Filtre les datasets SI-DEP contenant des données stratifiées par âge pour ne garder que les lignes "toutes tranches d'âge" :
- Priorité 1 : colonne `cl_age90 == "0"` (ou `clage`, `cl_age`)
- Priorité 2 : colonne `cage10 == "00"`
- Fallback : toutes les lignes si aucune colonne d'âge trouvée

#### `load_sidep_dataset(dataset_id, label)`
Sélectionne intelligemment le meilleur fichier CSV dans un dataset SI-DEP :
- Préfère les fichiers quotidiens (contenant "jour", "quot") aux fichiers hebdomadaires ("7j", "hebdo")
- Trie par date de modification décroissante

### 3.2 Les 17 sources de données

| # | Nom | URL / Provider | Granularité | Variables principales |
|---|-----|----------------|-------------|----------------------|
| 1 | Frontières géographiques | gregoiredavid/france-geojson (GitHub) | Département / Région | Polygones GeoJSON |
| 2 | Kalisio COVID GeoJSON | opendatasoft | Département × Jour | `day_hosp`, `day_intcare` (nouvelles hosp. quotidiennes) + polygones DOM |
| 3 | Hospitalisations stock | SPF / data.gouv.fr | Département × Jour | `hosp`, `rea`, `dc`, `rad` (sexe=0 = tous sexes) |
| 3b | Incidence quotidienne | — | — | Couvert par sources 2, 3, 10 |
| 3c | Hosp. par âge | SPF / data.gouv.fr | Région × Âge × Jour | Hospitalisations stratifiées par tranche d'âge |
| 4 | OpenCovid19-FR | github.com/opencovid19-fr | Département × Jour | `hospitalises`, `reanimation`, `deces`, `gueris` (combleur 2020 pré-SPF) |
| 5 | Vaccination par dept | VAC-SI / data.gouv.fr | Département × Jour | `couv_dose1`, `couv_complet`, `n_cum_dose1`, `n_cum_complet` (`clage_vacsi=0` = tous âges) |
| 6 | Vaccination par âge + dept | VAC-SI / data.gouv.fr | Département × Âge × Jour | Même variables, stratifiées (clage_vacsi: 0=tous, 4=<5, 9=5-9…) |
| 7 | Population INSEE 2020 | Recensement 2020 (hardcodé) | Département | `population` (101 depts, total ≈ 67M) |
| 8 | SI-DEP ancien format | data.gouv.fr | Département × Semaine | `P` (positifs), `T` (testés), `tx_pos` = P/T × 100 |
| 9 | SI-DEP nouveau format | data.gouv.fr | Département × Jour | `Tp` → `tx_pos`, `Ti` = incidence/100K |
| 10 | Indicateurs de suivi | AMP Métropole (mirror) | Département × Jour | `R0` (taux de reproduction), `Ti` (incidence/100K), `TO` (taux d'occupation réa) |
| 11 | Centres vaccination GPS | opendatasoft | Point GPS | Lat/Lon, nom, département |
| 12 | Vaccination par lieu | Ameli | National × Jour | Répartition par type (centre, cabinet, EHPAD, domicile, entreprise) |
| 13 | Vaccination par injecteur | Ameli | National × Jour | Médecin, pharmacien, infirmier, EHPAD… |
| 14 | Vaccination par marque | VAC-SI / data.gouv.fr | Département × Vaccin × Jour | 0=tous, 1=Pfizer, 2=Moderna, 3=AstraZeneca, 4=Janssen, 5=Novavax |
| 15 | Variants COVID | DREES (national) | National | Appariement SI-VIC × SI-DEP × VAC-SI |
| 15b | Variants dept | CovidTracker / G. Rozier (GitHub) | Département (optionnel) | Tentative dept-level, non garanti |
| 16 | FINESS établissements | Atlasanté / data.gouv.fr | Point GPS | Hôpitaux, labos, pharmacies, cliniques |
| 17 | SI-VIC (statut vaccinal) | DREES | Région × Statut vaccinal | Hosp/réa/décès split vaccinés vs non-vaccinés |

### 3.3 Post-traitements Phase 1

**A) Normalisation en taux pour 100 000 habitants**
```python
df[f"{col}_rate_100k"] = pd.to_numeric(df[col]) / population * 100_000
```
Appliqué à `hosp`, `rea`, `dc`, `rad` sur le snapshot le plus récent par département. Résultat → `data/processed/hospitalisations_normalized.csv`.

**B) Décalage vaccinal 14 jours (`couv_complet_lag14`)**
```python
df_vaccin_lag["couv_complet_lag14"] = df_vaccin_lag.groupby("dep")["couv_complet"].shift(14)
```
Modélise le délai de 14 jours pour l'acquisition de l'immunité post-vaccination complète. Résultat → `data/vaccines/vaccination_lag14.csv`.

**C) Vérification de cohérence (Merge Readiness Check)**
À la fin, le script vérifie l'intersection des codes département entre GeoJSON, hospitalisations et population INSEE. Alerte sur les codes non communs.

### 3.4 Gestion des erreurs

Chaque source non critique est enveloppée dans un bloc `try/except`. En cas d'échec réseau ou de format inattendu :
- Un warning est affiché (`⚠️`)
- Le pipeline continue sans cette source
- Sources non critiques pour Phase 2 : FINESS, CovidTracker variants, SI-VIC

---

## 4. Phase 2 — Prétraitement et feature engineering

**Script :** `phase2_preprocessing.py`
**Input :** tous les fichiers de `data/raw/`, `data/vaccines/`, `data/labs/`
**Output principal :** `data/processed/master_dept_daily.csv` (113 174 lignes × 47 colonnes)

### 4.1 Constantes de configuration

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `ROLLING_WINDOW` | 7 | Fenêtre de moyenne mobile (jours) |
| `LAG_VACC_DAYS` | 14 | Décalage vaccinal (immunité) |
| `RATE_BASE` | 100 000 | Base de normalisation |
| `DOM_DEPS` | {971, 972, 973, 974, 976} | Codes départements d'outre-mer |
| `SPF_START` | 2020-03-18 | Début des données SPF (hospitalisations) |

### 4.2 Pipeline de construction étape par étape

#### STEP 2 — Base hospitalière

À partir de `hospitalisations.csv` (SPF, sexe=0) :

| Feature créée | Formule | Description |
|---------------|---------|-------------|
| `new_hosp` | `diff(hosp).clip(lower=0)` | Nouveaux hospitalisés par jour (flux positif) |
| `new_rea` | `diff(rea).clip(lower=0)` | Nouvelles admissions réanimation |
| `new_dc` | `diff(dc).clip(lower=0)` | Nouveaux décès hospitaliers |
| `new_rad` | `diff(rad).clip(lower=0)` | Nouvelles sorties (retours à domicile) |
| `new_hosp_7j` | `rolling_mean(new_hosp, 7)` | Moyenne 7j des nouvelles admissions |
| `new_rea_7j` | `rolling_mean(new_rea, 7)` | Moyenne 7j réanimation |
| `new_dc_7j` | `rolling_mean(new_dc, 7)` | Moyenne 7j décès |
| `hosp_7j` | `rolling_mean(hosp, 7)` | Stock hospitalisations lissé |
| `hosp_rate` | `hosp / population × 100K` | Taux d'hospitalisation pour 100 000 |
| `rea_rate` | `rea / population × 100K` | Taux réanimation pour 100 000 |
| `dc_rate` | `dc / population × 100K` | Taux de décès pour 100 000 |
| `new_hosp_rate` | `new_hosp / population × 100K` | Taux de nouvelles admissions |
| `new_dc_rate` | `new_dc / population × 100K` | Taux de nouveaux décès |

> **Note technique :** Le `clip(lower=0)` sur les diff() est intentionnel. Les valeurs négatives dans les stocks représentent des corrections rétroactives des sources officielles, pas des flux négatifs réels.

#### STEP 3 — Unification SI-DEP (taux de positivité)

Deux formats distincts selon la période :

| Format | Période | Granularité | Colonne date | Variables |
|--------|---------|-------------|--------------|-----------|
| Ancien | Mai 2020 → Mai 2022 | Hebdomadaire | `semaine_glissante` (10 premiers caractères) | `P` (positifs), `T` (testés) → `tx_pos = P/T × 100` |
| Nouveau | Mai 2022 → Mars 2023 | Quotidien | `jour` | `Tp` → renommé `tx_pos`, `Ti` = incidence/100K |

Après concaténation et dédoublonnage sur `(dep, jour)` :
```python
tx_pos_7j = rolling_mean(tx_pos, 7)  # moyenne glissante 7j par département
```

#### STEP 4 — Vaccination

Colonnes conservées après nettoyage :

| Colonne | Description |
|---------|-------------|
| `couv_dose1` | % population ayant reçu au moins 1 dose |
| `couv_complet` | % population avec schéma vaccinal complet |
| `n_cum_dose1` | Nombre cumulatif de premières doses |
| `n_cum_complet` | Nombre cumulatif de schémas complets |
| `couv_complet_lag14` | `couv_complet` décalée de 14 jours (immunité effective) |

#### STEP 5 — Indicateurs épidémiques (R0, Ti, TO)

Détection dynamique des colonnes (les noms varient selon la source miroir) :

| Variable cible | Alias reconnus | Description |
|----------------|----------------|-------------|
| `R0` | `r`, `r0`, `r_effectif`, `taux_reproduction` | Taux de reproduction effectif |
| `Ti` | `ti`, `ti100k`, `taux_incidence`, `tx_incid` | Taux d'incidence pour 100 000 |
| `TO` | `to`, `tx_occupation`, `sae_rate` | Taux d'occupation réanimation (%) |

Après mapping : `R0_7j = rolling_mean(R0, 7)` par département.

#### STEP 6 — Comblement 2020 (OpenCovid19-FR)

Les données SPF commencent le 2020-03-18. Les lignes d'OpenCovid19-FR antérieures à cette date (`jour < 2020-03-18`) sont extraites et prépendées au dataset principal.

Mapping des colonnes OpenCovid19 → master :

| OpenCovid19 | master |
|-------------|--------|
| `hospitalises` | `hosp` |
| `reanimation` | `rea` |
| `deces` | `dc` |
| `gueris` | `rad` |
| `cas_confirmes` | `cas_confirmes` |

#### STEP 7 — Fusion principale (Master Merge)

Ordre des jointures (toutes `LEFT JOIN` sur `[dep, jour]`) :

```
df_hosp  (base)
  ← df_sidep     [tx_pos, tx_pos_7j]
  ← df_vacc      [couv_dose1, couv_complet, n_cum_*, couv_complet_lag14]
  ← df_indic     [R0, R0_7j, Ti, TO]
  ← df_oc_gap    (prépendé pour les dates < 2020-03-18)
```

**Filtre des codes parasites :** les codes département non présents dans la table INSEE (ex. "00", "20" — l'ancien code de la Corse avant la division en 2A/2B) sont supprimés. La table INSEE sert de référentiel autoritaire.

#### STEP 8 — Features épidémiologiques dérivées

| Feature | Formule | Interprétation |
|---------|---------|----------------|
| `total_hosp_cases` | `dc + rad` | Cas résolus (décédés + guéris) |
| `cfr` | `dc / (dc + rad) × 100` | Létalité hospitalière (Case Fatality Rate) |
| `icu_pressure` | `rea / hosp × 100` | Part des hospitalisés en réanimation |
| `tx_pos_adj_vacc` | `tx_pos × (1 - couv_complet_lag14 / 100)` | Positivité ajustée par couverture vaccinale effective |
| `week` | `jour.dt.isocalendar().week` | Semaine ISO |
| `year` | `jour.dt.year` | Année |
| `month` | `jour.dt.month` | Mois |
| `day_of_week` | `jour.dt.dayofweek` | Jour de la semaine (0=lundi) |
| `is_weekend` | `day_of_week ∈ {5, 6}` | Marqueur week-end (biais de déclaration) |
| `is_dom` | `dep ∈ {971..976}` | Marqueur DOM |

#### STEP 9 — Détection des vagues épidémiques

7 vagues définies par des plages de dates calendaires fixes :

| Vague | Début | Fin | Variant dominant |
|-------|-------|-----|-----------------|
| 1 | 2020-03-01 | 2020-06-30 | Souche originale |
| 2 | 2020-09-01 | 2021-02-28 | Souche + Alpha émergent |
| 3 | 2021-03-01 | 2021-06-30 | Alpha (B.1.1.7) |
| 4 | 2021-07-01 | 2021-11-30 | Delta (B.1.617.2) |
| 5 | 2021-12-01 | 2022-03-31 | Omicron BA.1 |
| 6 | 2022-04-01 | 2022-07-31 | Omicron BA.2 / BA.5 |
| 7 | 2022-10-01 | 2023-02-28 | Omicron BQ.1 / XBB |

**Implémentation vectorisée :** construction d'un dictionnaire `date → vague`, puis merge sur `jour`. Évite une boucle Python `O(n×7)` sur 113K lignes. Les dates hors vague reçoivent `wave = 0`.

#### STEP 10 — Matrice de features pour le clustering

Agrégation sur les périodes de vague (wave > 0 uniquement) par département :

| Feature agrégée | Statistiques calculées |
|-----------------|----------------------|
| `hosp_rate` | mean, max |
| `new_hosp_rate` | mean, max |
| `rea_rate` | mean, max |
| `icu_pressure` | mean, max |
| `cfr` | mean, max |
| `tx_pos` | mean, max |
| `tx_pos_7j` | mean |
| `couv_complet` | max |
| `couv_complet_lag14` | mean |

> R0 **exclu** de la matrice de clustering : taux de remplissage de 9,5%, valeur quasi-constante après remplissage par la médiane → ajoute du bruit sans pouvoir discriminant.

Les valeurs manquantes résiduelles sont remplacées par la médiane de la colonne. Résultat : `dept_clusters_features.csv` (101 depts × 15 features).

### 4.3 Dataset final `master_dept_daily.csv`

| Dimension | Valeur |
|-----------|--------|
| Lignes | 113 174 |
| Colonnes | 47 |
| Départements | 101 |
| Période | 2020-01-24 → 2023-03-31 |
| Complétude globale | ~75% (colonnes manquantes selon la période) |

Complétude par métrique clé :
- `hosp`, `rea`, `dc` : ~100% (SPF complet)
- `tx_pos` : ~70% (SI-DEP commence mai 2020)
- `couv_complet` : ~65% (vaccination à partir de janvier 2021)
- `R0`, `Ti`, `TO` : ~30–65% (source indicateurs partielle)

---

## 5. Phase 3 — Clustering épidémiologique

**Script :** `phase3_clustering.py`
**Input :** `data/processed/dept_clusters_features.csv` (101 depts × 15 features)
**Output :** `data/clustering/` (10 fichiers)

### 5.1 Séparation DOM / Métropole

Avant tout clustering, les 5 DOM (971–976) sont mis de côté. Raison : leurs profils épidémiques et démographiques sont structurellement différents (éloignement géographique, démographie, taux de vaccination distincts). Les métadonnées DBSCAN les marquent comme `cluster = -2` (exclus de l'analyse de densité).

À la fin du pipeline, les DOM sont réassignés au cluster K-Means métropolitain le plus proche par leur centroïde (méthode `predict` sur le modèle ajusté sur la métropole).

### 5.2 STEP 2 — Mise à l'échelle (RobustScaler)

```python
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)
```

**Pourquoi RobustScaler plutôt que StandardScaler ?**
- Utilise la médiane et l'IQR (interquartile range) au lieu de la moyenne et l'écart-type
- Résistant aux départements atypiques (Paris, Hauts-de-Seine pour la densité ; DOM pour les profils extrêmes)
- Évite que des outliers fassent diverger la normalisation de l'ensemble

### 5.3 STEP 3 — Analyse en Composantes Principales (PCA)

**Objectif :** réduire la dimensionnalité avant le clustering pour supprimer le bruit et visualiser les groupements.

#### Variance expliquée

| Seuil | Nombre de composantes requises |
|-------|-------------------------------|
| 80% | calculé à l'exécution |
| 90% | **7 composantes** |
| 95% | **8 composantes** |

#### Projections créées

| Projection | Composantes | Variance captée | Usage |
|------------|-------------|-----------------|-------|
| 2D | PC1, PC2 | variable | Visualisation carte + scatter |
| 3D | PC1, PC2, PC3 | variable | Visualisation interactive 3D |

#### Loadings — Contributions des features à PC1

PC1 représente l'axe de **sévérité hospitalière globale** :

| Feature | Loading PC1 | Loading PC2 |
|---------|------------|------------|
| `rea_rate_max` | **+0.4211** | −0.0276 |
| `hosp_rate_mean` | **+0.3697** | +0.1378 |
| `hosp_rate_max` | **+0.3481** | +0.2225 |
| `rea_rate_mean` | **+0.3326** | −0.1305 |
| `new_hosp_rate_max` | **+0.2688** | +0.4323 |

PC2 capture plutôt la **dynamique des nouvelles admissions** (pic vs stock moyen).

### 5.4 STEP 4 — Sélection du k optimal (K-Means)

Trois critères évalués pour k = 2 à 10 :

| Critère | Formule | Optimal quand |
|---------|---------|---------------|
| Méthode du coude (elbow) | Plus grande chute d'inertie entre k et k+1 | k = argmax(Δinertie) |
| Score de silhouette | `(b - a) / max(a, b)` | **Maximiser** (range : −1 à +1) |
| Calinski-Harabász (CH) | Ratio variance inter/intra | **Maximiser** |

Le k final est déterminé par **vote majoritaire** parmi les 3 critères. En cas d'ex-aequo : priorité au score de silhouette (plus interprétable géographiquement).

**Résultat obtenu : k = 2**

### 5.5 STEP 5 — K-Means final

```python
KMeans(n_clusters=2, random_state=42, n_init=50, max_iter=1000)
```

- `n_init=50` : 50 initialisations aléatoires → converge vers l'optimum global
- `max_iter=1000` : garantit la convergence sur des données dimensionnelles

#### Scores de qualité

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Silhouette | **0.2406** | Séparation modérée (structure réelle dans les données) |
| Calinski-Harabász | **26.7** | Rapport variance inter/intra |
| Davies-Bouldin | **1.6041** | Distance intra/inter-cluster (plus bas = mieux) |
| Inertie | — | Compacité intra-cluster |

#### Profils des clusters

| Cluster | Taille | `hosp_rate_mean` | `tx_pos_mean` | `couv_complet_max` | `cfr_mean` | Caractérisation |
|---------|--------|-----------------|---------------|-------------------|------------|-----------------|
| **0** | 25 depts | **37.1 /100K** | 9.5% | 75.2% | 18.4% | Forte sévérité hospitalière, densité urbaine élevée |
| **1** | 71 depts | **22.5 /100K** | 11.3% | 79.4% | 17.8% | Impact modéré, diffusion plus large, meilleure vaccination |

> Cluster 0 = départements à haute pression hospitalière (généralement Île-de-France, PACA, grandes métropoles)
> Cluster 1 = reste de la métropole

### 5.6 STEP 6 — DBSCAN (clustering par densité)

DBSCAN opère dans l'espace PCA réduit à n_90 composantes (90% de variance) pour éviter la **malédiction de la dimensionnalité**.

**Paramètres :**
- `eps` : rayon de voisinage (testé : 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0)
- `min_samples` : minimum de voisins pour former un cluster (testé : 2, 3, 4, 5)

**Critère de sélection :** meilleure silhouette, avec la contrainte que ≥ 2 clusters soient formés et que les outliers restent < 30% du total.

**Résultat optimal :**

| Paramètre | Valeur |
|-----------|--------|
| `eps` | **2.0** |
| `min_samples` | **3** |
| Clusters trouvés | **2** |
| Points bruit (outliers) | **16** départements |
| Silhouette | **0.1268** |

Les 16 départements marqués comme bruit (`cluster = -1`) sont ceux dont le profil épidémique est trop atypique pour appartenir à une densité locale.

**Intérêt de DBSCAN vs K-Means :**
- Détecte les outliers vrais (départements à profil unique)
- Ne force pas tous les points dans un cluster
- Permet de trouver des formes non-sphériques dans l'espace de features

### 5.7 STEP 9 — Stabilité inter-vagues

Pour chaque vague épidémique (1 à 7), un K-Means indépendant est ajusté sur les features agrégées pour cette vague seulement. On mesure ensuite si un département reste dans le même cluster relatif à travers les vagues.

Un département est dit **wave-stable** si son label de cluster ne change pas d'une vague à l'autre (après alignement des labels). Ce taux de stabilité indique dans quelle mesure les profils épidémiques sont constants ou évoluent au fil des variants.

---

## 6. Phase 4 — Modèles de Machine Learning

**Script :** `phase4_Ml-models.py`
**Input :** `master_dept_daily.csv` + `global_cluster_labels.csv`
**Output :** `data/models/` (6 fichiers)

### 6.1 Définition du problème

**Type de problème :** régression supervisée
**Variable cible (`target`) :** `hosp_rate` — taux d'hospitalisation pour 100 000 habitants

> Ce taux est la variable de synthèse la plus opérationnelle : il intègre la sévérité clinique, la démographie du département et est directement utilisé pour dimensionner les capacités hospitalières.

### 6.2 STEP 2 — Feature engineering : les variables retardées (lags)

Les lags capturent la **causalité temporelle** entre indicateurs épidémiques et hospitalisations :

| Feature | Lag (jours) | Justification épidémiologique |
|---------|-------------|-------------------------------|
| `hosp_rate_lag7` | 7 | Autorégressif : le stock de la semaine passée est le meilleur prédicteur du stock actuel |
| `hosp_rate_lag14` | 14 | Autorégressif long terme |
| `new_hosp_rate_lag7` | 7 | Les nouvelles admissions d'il y a 7j composent le stock actuel |
| `new_hosp_rate_lag1` | 1 | Admissions d'hier |
| `tx_pos_7j_lag7` | 7 | La positivité ~7j avant annonce les hospitalisations (délai symptômes → test → hospit) |
| `couv_complet_lag14` | 0 | Déjà décalé de 14j en Phase 2 — pas de lag supplémentaire |
| `rea_rate` | 0 | Concurrent — co-évolution avec `hosp_rate` |
| `TO` | 0 | Taux d'occupation réa — indicateur de pression système |
| `icu_pressure` | 0 | Proportion des hospitalisés en réa — sévérité instantanée |

> **R0 et Ti exclus :** taux de remplissage insuffisant (9.5% et ~65%) et les données s'arrêtent en 2022, ce qui supprimerait 70% des lignes du jeu d'entraînement après `dropna()`.

**Variables contextuelles (sans lag) :**
- `wave` : numéro de vague (contexte épidémique)
- `kmeans_cluster` : cluster du département (profil structurel)
- `is_dom` : marqueur DOM (différences structurelles)
- `month`, `year`, `day_of_week` : saisonnalité

**Total : 15 features** après dédoublonnage

### 6.3 STEP 3 — Découpage temporel (Chronological Split)

```python
TEST_FRAC = 0.2  # 80% train / 20% test
split_idx = int(len(df_model) * 0.8)
```

Les données sont découpées **chronologiquement** (pas de shuffle aléatoire) pour respecter l'intégrité temporelle :

| Ensemble | Période | Taille |
|----------|---------|--------|
| **Train** | 2021-01-10 → 2022-05-02 | 48 172 lignes |
| **Test** | 2022-05-02 → 2022-08-29 | 12 043 lignes |

> Le dataset commence effectivement en 2021-01-10 (et non 2020) car après `dropna()` sur les features retardées, les lignes manquantes de couv_complet (vaccination commence en janvier 2021) sont supprimées.

### 6.4 STEP 4 — Validation croisée temporelle (TimeSeriesSplit)

```python
TimeSeriesSplit(n_splits=5)
```

Contrairement à un `KFold` classique, `TimeSeriesSplit` garantit que les données de validation sont **toujours postérieures** aux données d'entraînement dans chaque fold. Cela évite le **data leakage temporel** (utiliser des données futures pour prédire le passé).

Structure des folds :
```
Fold 1:  Train [0..n1]          Val [n1..n2]
Fold 2:  Train [0..n2]          Val [n2..n3]
Fold 3:  Train [0..n3]          Val [n3..n4]
Fold 4:  Train [0..n4]          Val [n4..n5]
Fold 5:  Train [0..n5]          Val [n5..n6]
```

### 6.5 STEP 5 — Modèles entraînés

Tous les modèles sont encapsulés dans un `Pipeline` scikit-learn :
```python
Pipeline([("scaler", RobustScaler()), ("model", <estimateur>)])
```

Le `RobustScaler` dans chaque pipeline garantit que la normalisation est ajustée uniquement sur les données d'entraînement (pas de data leakage).

#### Modèles individuels

| Modèle | Hyperparamètres | Type |
|--------|----------------|------|
| **Baseline (Median)** | strategy="median" | DummyRegressor |
| **Ridge** | alpha=1.0 | Régression L2 |
| **Lasso** | alpha=0.1, max_iter=5000 | Régression L1 (sélection de features) |
| **ElasticNet** | alpha=0.1, l1_ratio=0.5 | Combinaison L1+L2 |
| **Random Forest** | n_estimators=300, max_depth=15, min_samples_leaf=3 | Ensemble bagging |
| **ExtraTrees** | n_estimators=300, max_depth=15, min_samples_leaf=3 | Ensemble bagging extrême |
| **HistGradientBoosting** | max_iter=400, max_depth=6, learning_rate=0.05, l2_reg=0.1 | Boosting (inspiré LightGBM) |
| **SVR** | kernel="rbf", C=10.0, epsilon=0.5, gamma="scale" | Support Vector Regression |

### 6.6 STEP 6 — Résultats et comparaison

#### Modèles individuels + ensembles (triés par R² test)

| Modèle | Train R² | Test R² | MAE | RMSE | MAPE | Overfit gap |
|--------|----------|---------|-----|------|------|------------|
| **HistGradientBoosting** | 0.994 | **0.958** | 1.690 | 2.497 | 8.73% | 0.036 |
| WeightedVoting | 0.992 | 0.938 | 2.128 | 3.037 | 10.64% | 0.055 |
| ManualStack | 0.991 | 0.920 | 2.395 | 3.433 | 12.51% | 0.071 |
| ExtraTrees | 0.987 | 0.917 | 2.496 | 3.497 | 12.65% | 0.070 |
| Random Forest | 0.992 | 0.917 | 2.451 | 3.498 | 11.81% | 0.074 |
| Lasso | 0.959 | 0.911 | 2.682 | 3.636 | 15.17% | 0.048 |
| SVR | 0.977 | 0.876 | 2.839 | 4.286 | 18.23% | 0.101 |
| ElasticNet | 0.946 | 0.873 | 3.356 | 4.333 | 20.49% | 0.073 |
| Ridge | 0.964 | 0.790 | 4.179 | 5.580 | 21.25% | 0.174 |
| Baseline (Médiane) | −0.043 | −0.036 | 9.565 | 12.385 | 60.93% | — |

**Interprétation :**
- L'**overfit gap** (R² train − R² test) est faible pour HistGradientBoosting (0.036), signe d'une bonne généralisation
- Ridge a un overfit gap élevé (0.174) : la normalisation L2 seule ne suffit pas à contraindre suffisamment le modèle sur des features temporellement autocorrélées
- Lasso (0.048) généralise mieux que Ridge grâce à la sélection implicite de features (certains coefficients mis à zéro)

#### Méthodes d'ensemble

**A) Weighted Voting :** moyenne pondérée des prédictions des 3 meilleurs modèles individuels, avec des poids proportionnels à leur R² test.

```python
weights = [max(0, r2_model_i) for i in top_3]
weights /= sum(weights)
y_pred = sum(w * pipe.predict(X_test) for w, pipe in zip(weights, top_3_pipes))
```

**B) Manual Stacking avec TimeSeriesSplit :**
1. Chaque modèle de base génère des prédictions **out-of-fold** (OOF) sur les données d'entraînement via TimeSeriesSplit
2. Un méta-modèle Ridge est ajusté sur ces prédictions OOF
3. Sur le test : les 3 modèles de base prédisent → le Ridge méta combine

> L'implémentation manuelle est nécessaire car `StackingRegressor` scikit-learn est incompatible avec `TimeSeriesSplit` (les partitions CV ne couvrent pas toutes les observations).

### 6.7 STEP 7 — Hyperparameter Tuning (RandomizedSearchCV)

Le meilleur modèle individuel (HistGradientBoosting) est soumis à une recherche aléatoire :

```python
RandomizedSearchCV(n_iter=40, cv=TimeSeriesSplit(5), scoring="r2", n_jobs=-1)
```

**Espace de recherche pour HistGradientBoosting :**

| Hyperparamètre | Distribution | Plage |
|----------------|-------------|-------|
| `max_iter` | randint | 200 – 800 |
| `max_depth` | randint | 3 – 10 |
| `learning_rate` | uniform | 0.01 – 0.16 |
| `l2_regularization` | uniform | 0.0 – 1.0 |
| `min_samples_leaf` | randint | 10 – 60 |

**40 itérations × 5 folds = 200 fits** au total.

Si le modèle tuné dépasse le modèle par défaut en R² test → version tunée retenue. Sinon → version par défaut conservée.

**Résultat final après tuning :**

| Métrique | Valeur |
|----------|--------|
| Test R² | **0.9744** |
| Test MAE | **1.140** hospitalisation/100K |
| Test RMSE | **1.946** hospitalisation/100K |
| Test MAPE | **5.80%** |

### 6.8 STEP 8 — Importance des features

Pour les modèles arborescents (Random Forest, ExtraTrees, HistGradientBoosting) : `feature_importances_` (impureté de Gini normalisée)
Pour les modèles linéaires (Ridge, Lasso, ElasticNet) : `abs(coef_)` normalisés

Les features autorégressives (`hosp_rate_lag7`, `hosp_rate_lag14`) dominent généralement l'importance, ce qui confirme que la prédiction du taux d'hospitalisation est **principalement autoregressive** sur un horizon court.

### 6.9 STEP 9 — Analyse des résidus

Les prédictions et résidus sont analysés selon deux dimensions de croisement :

**Par vague épidémique :**
```
Wave 1 : données pré-vaccination → modèle moins précis (features vaccin = NaN)
Wave 5–7 : données avec vaccination complète → meilleures performances
```

**Par cluster K-Means :**
Les départements du Cluster 0 (forte sévérité) peuvent présenter des MAE plus élevés lors des pics, car ce sont des situations extrêmes sous-représentées dans l'ensemble d'entraînement.

**Top 10 pires départements (MAE moyen)** : identifie les départements pour lesquels le modèle généralise moins bien, utile pour cibler des analyses supplémentaires.

### 6.10 Sérialisation du modèle (`best_model.pkl`)

Le fichier pickle contient un dictionnaire complet :

```python
{
    "model":       final_pipe,   # Pipeline(RobustScaler + HistGradientBoosting)
    "features":    ALL_FEATURES, # Liste des 15 features (noms et ordre)
    "target":      "hosp_rate",
    "best_name":   "HistGradientBoosting",
    "test_scores": {"mae": 1.140, "rmse": 1.946, "r2": 0.9744, "mape": 5.80},
    "train_date":  "2022-05-02",
    "test_date":   "2022-08-29",
}
```

**Usage en production :**
```python
import pickle, pandas as pd

with open("data/models/best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipe     = bundle["model"]
features = bundle["features"]

# df_new doit contenir exactement les colonnes dans features, dans le même ordre
X_new    = df_new[features].to_numpy(dtype=float)
y_pred   = pipe.predict(X_new)  # hosp_rate prédit /100K
```

---

## 7. Phase 5 — Tableau de bord interactif

**Script :** `phase5_dashboard.py`
**Output :** `outputs/dashboard_covid.html` (~1.2 MB, autonome)

### 7.1 Architecture technique

Le tableau de bord est un **fichier HTML autonome** — aucun serveur nécessaire, ouvrable directement dans un navigateur. Les données sont sérialisées en JSON et intégrées dans le HTML à la génération. Le rendu des graphiques utilise **Plotly.js 2.27** chargé depuis CDN.

### 7.2 Données embarquées

| Données | Représentation | Taille embarquée |
|---------|---------------|-----------------|
| Série nationale quotidienne | `{dates: [...], hosp: [...], ...}` (~1 149 dates × 10 métriques) | ~84 KB |
| Série départementale hebdomadaire | `{dep: {metric: [167 valeurs]}}` (101 depts × 6 métriques × 167 semaines) | ~536 KB |
| GeoJSON simplifié | Polygones sans attributs inutiles | ~547 KB |
| Périodes de vagues | 7 objets | < 1 KB |
| Labels clusters | dict dep → cluster | < 1 KB |

### 7.3 Sections du tableau de bord

**Barre de contrôle (sticky)**
- Curseur de date (slider HTML range, 167 positions hebdomadaires)
- Affichage de la date sélectionnée (format long français)
- Sélecteur d'indicateur (6 métriques pour la carte et le classement)
- Badge de vague active (surlignage bleu si la date est dans une vague)

**Cartes KPI (5 indicateurs)**

| KPI | Source | Type d'agrégation |
|-----|--------|------------------|
| Hospitalisés | `nat_data.hosp` | Somme nationale |
| Réanimation | `nat_data.rea` | Somme nationale |
| Décès cumulés | `nat_data.dc` | Somme nationale |
| Positivité 7j | `nat_data.tx_pos_7j` | Moyenne nationale |
| Couv. vaccinale | `nat_data.couv_complet` | Moyenne nationale |

Chaque KPI affiche une flèche de tendance (▲/▼) avec le pourcentage d'évolution vs la semaine précédente, colorée en rouge/vert selon la direction favorable.

**Graphique évolution nationale**
- 3 séries sur axe Y gauche : Hospitalisés (bleu, aire), Réanimation (rouge, aire), Nouveaux hosp. 7j (orange pointillé)
- 1 série sur axe Y droit : Positivité 7j (violet tiret)
- Fonds colorés par vague (7 couleurs distinctes)
- Ligne verticale mobile (curseur de date)

**Carte choroplèthe France**
- Choropleth Plotly.js avec GeoJSON custom (sans Mapbox, sans token)
- `featureidkey: 'properties.dep'` pour l'association dept ↔ valeur
- Palette de couleurs selon l'indicateur (Reds pour hosp, Greens pour vaccination…)
- Clic sur un département → affichage du détail

**Top 15 départements (barres horizontales)**
- Trié par valeur décroissante de l'indicateur sélectionné
- Couleur des barres selon le cluster K-Means du département
- Clic sur une barre → affichage du détail

**Scatter Positivité vs Hospitalisation**
- Axe X : `tx_pos_7j` (%) ; Axe Y : `hosp_rate` (/100K)
- Un point par département, coloré par cluster
- Hover : nom complet du département + valeurs

**Panel de détail département (affiché au clic)**
- Série temporelle complète (167 semaines)
- Axe Y gauche : hosp_rate, new_hosp_rate, icu_pressure
- Axe Y droit : tx_pos_7j (%), couv_complet (%)
- Fonds de vague colorés

### 7.4 Logique de mise à jour JavaScript

Toutes les visualisations sont mises à jour via des fonctions Plotly efficaces :

```javascript
// Évite un full redraw — met à jour uniquement les données
Plotly.restyle('ch-map', { locations: [locs], z: [vals] }, 0);

// Déplace uniquement la ligne curseur
Plotly.relayout('ch-nat', {
    [`shapes[${WAVES.length}].x0`]: WDATES[curIdx],
    [`shapes[${WAVES.length}].x1`]: WDATES[curIdx],
});
```

L'objet `DEPT` en mémoire JavaScript a la structure `{dep: {metric: [valeurs_par_semaine]}}`, ce qui permet d'extraire en O(101) les valeurs pour n'importe quelle date et n'importe quel indicateur.

---

## 8. Conventions et décisions de conception

### 8.1 Codes département

| Règle | Description |
|-------|-------------|
| Format standard | Chaîne à 2 caractères avec zéro de tête : `"01"` à `"95"` |
| Corse | `"2A"` et `"2B"` — conservés tels quels (pas de conversion numérique) |
| DOM | `"971"` à `"976"` — 3 caractères (pas de troncature) |
| Fonction de normalisation | `normalize_dep()` — point d'entrée unique pour tout code externe |
| Codes à supprimer | `"00"` (invalide), `"20"` (ancien code Corse avant 1976) |

### 8.2 Colonne de date

- Toujours renommée `"jour"` après chargement de n'importe quelle source
- Parsée avec `pd.to_datetime(..., errors="coerce")` (NaT en cas d'erreur)
- Format de stockage dans les CSV : ISO 8601 (`YYYY-MM-DD`)

### 8.3 Filtrage des âges (SI-DEP)

Les datasets SI-DEP contiennent des lignes stratifiées par tranche d'âge. La logique de filtrage prioritaire est :
1. `cl_age90 == "0"` → toutes tranches d'âge confondues
2. `cage10 == "00"` → equivalent dans l'ancien format
3. Si aucune colonne d'âge → toutes les lignes conservées (warning)

### 8.4 Gestion des valeurs négatives dans les diff()

Les stocks hospitaliers (hosp, rea, dc) sont des valeurs cumulées. Leur différentiation quotidienne peut produire des valeurs négatives lors de corrections rétroactives des sources officielles. Ces valeurs négatives sont **clippées à 0** (`clip(lower=0)`) car elles ne représentent pas un flux réel.

### 8.5 Décalage vaccinal de 14 jours

La couverture vaccinale (`couv_complet`) est décalée de 14 jours avant fusion. Ce délai correspond au temps moyen pour développer une protection immunitaire significative après la deuxième dose. La variable `couv_complet_lag14` représente donc la **couverture effective** au moment t.

### 8.6 Exclusion de R0 du clustering et du ML

R0 n'est pas utilisé dans la matrice de clustering ni comme feature ML :
- **Taux de remplissage insuffisant :** ~9.5% du dataset
- **Données arrêtées en 2022 :** inclure R0 supprimerait ~70% des lignes après `dropna()`
- **Comportement quasi-constant après remplissage médian :** aucun pouvoir discriminant

### 8.7 Pipeline sklearn avec RobustScaler intégré

Chaque modèle Phase 4 est encapsulé dans `Pipeline([("scaler", RobustScaler()), ("model", ...)])`. Avantages :
- Le scaler est ajusté uniquement sur `X_train` (pas de data leakage)
- Compatible avec `RandomizedSearchCV` (le scaler est refitté à chaque fold)
- Le `best_model.pkl` contient le pipeline complet, utilisable directement sur des données brutes non normalisées

### 8.8 Répertoire de travail

Tous les scripts utilisent des **chemins relatifs** depuis `src/`. Ne pas exécuter depuis le répertoire parent, sinon les chemins `data/raw/...` seront introuvables.

---

## 9. Dépendances

### Packages Python requis

```
pandas>=1.5         # DataFrame, resample, groupby
numpy>=1.23         # Calculs vectorisés
geopandas>=0.12     # Lecture/écriture GeoJSON, reprojection CRS
requests>=2.28      # Téléchargement des sources réseau
scikit-learn>=1.2   # Clustering, ML, prétraitement, métriques
scipy>=1.9          # Distributions statistiques pour RandomizedSearchCV
```

> Pas de `requirements.txt` dans le dépôt. Installation suggérée :
> ```bash
> pip install pandas numpy geopandas requests scikit-learn scipy
> ```

### Environnement suggéré

- Python 3.10+ (syntax `dict[str, Type]` dans les type hints)
- Miniconda ARM64 fourni : `src/Miniconda3-latest-MacOSX-arm64.sh`

### Ressources réseau requises (Phase 1 uniquement)

- `data.gouv.fr` — SPF, VAC-SI, SI-DEP, INSEE
- `public.opendatasoft.com` — Kalisio GeoJSON, centres vaccination
- `data.ampmetropole.fr` — Indicateurs épidémiques
- `raw.githubusercontent.com` — GeoJSON france, OpenCovid19-FR
- `datavaccin-covid.ameli.fr` — Ameli vaccination
- `data.smartidf.services` — DREES variants, SI-VIC
- `atlasante.fr` / data.gouv.fr — FINESS établissements

Les phases 2, 3, 4 et 5 fonctionnent **entièrement hors ligne** à partir des fichiers générés par Phase 1.

---

*Documentation générée à partir du code source — versions des scripts : Phase 1 v10, Phase 2 v1, Phase 3 v1, Phase 4 v2, Phase 5 v2.*
