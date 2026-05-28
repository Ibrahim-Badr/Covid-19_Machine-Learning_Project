# Données affichées dans le tableau de bord — Origine et calcul

Ce document répond à une question simple : **pour chaque indicateur affiché dans le dashboard, d'où vient la valeur — est-elle calculée (ML, clustering, feature engineering) ou simplement copiée depuis un CSV source ?**

---

## Légende des origines

| Symbole | Signification |
|---------|--------------|
| 📥 **CSV brut** | Valeur lue directement depuis un fichier source (SPF, VAC-SI…) sans transformation |
| ⚙️ **Calculé Phase 2** | Valeur calculée par le script de prétraitement (`phase2_preprocessing.py`) |
| 🔵 **Clustering Phase 3** | Résultat du clustering K-Means ou DBSCAN (`phase3_clustering.py`) |
| 🤖 **ML Phase 4** | Prédiction ou indicateur produit par un modèle ML (`phase4_Ml-models.py`) |

---

## 1. Cartes KPI — 5 valeurs en haut du dashboard

Ces 5 KPI affichent une **agrégation nationale** (somme ou moyenne sur tous les départements) à la date sélectionnée par le curseur. Toutes proviennent de `master_dept_daily.csv` (Phase 2).

| KPI | Colonne source | Agrégation nationale | Origine |
|-----|---------------|---------------------|---------|
| **Hospitalisés** | `hosp` | Somme sur 101 depts | 📥 CSV brut SPF — stock journalier des personnes hospitalisées |
| **Réanimation** | `rea` | Somme sur 101 depts | 📥 CSV brut SPF — stock journalier des personnes en réanimation |
| **Décès cumulés** | `dc` | Somme sur 101 depts | 📥 CSV brut SPF — cumul des décès hospitaliers |
| **Positivité 7j** | `tx_pos_7j` | Moyenne sur 101 depts | ⚙️ Calculé Phase 2 : `rolling_mean(tx_pos, 7)` par département |
| **Couv. vaccinale** | `couv_complet` | Moyenne sur 101 depts | 📥 CSV brut VAC-SI — % population avec schéma vaccinal complet |

> **Flèche de tendance** : calculée en JavaScript dans le navigateur — variation % entre la date sélectionnée et 7 jours avant (données de `nat_data` embeddées dans le HTML).

---

## 2. Graphique évolution nationale (ligne du temps)

Toutes les séries viennent de `master_dept_daily.csv`, agrégées par date.

| Série | Colonne source | Agrégation | Origine |
|-------|---------------|------------|---------|
| **Hospitalisés** (courbe bleue) | `hosp` | Somme nationale par jour | 📥 CSV brut SPF |
| **Réanimation** (courbe rouge) | `rea` | Somme nationale par jour | 📥 CSV brut SPF |
| **Nouveaux hosp. 7j** (tirets orange) | `new_hosp_7j` | Somme nationale par jour | ⚙️ Calculé Phase 2 : `rolling_mean(new_hosp, 7)` |
| **Positivité 7j %** (tirets violet, axe droit) | `tx_pos_7j` | Moyenne nationale par jour | ⚙️ Calculé Phase 2 : `rolling_mean(tx_pos, 7)` |
| **Fonds colorés** (bandes de vagues) | `wave_periods.csv` | — | ⚙️ Calculé Phase 2 : plages de dates fixes définies manuellement |
| **Ligne verticale curseur** | Slider HTML | — | Calculé dans le navigateur |

---

## 3. Carte choroplèthe France

La couleur de chaque département est déterminée par la valeur de l'**indicateur sélectionné dans le dropdown** à la date du curseur. Les valeurs viennent de `master_dept_daily.csv` (Phase 2), rééchantillonnées à la semaine.

### 6 indicateurs disponibles

| Indicateur | Colonne dans master | Calcul ou source | Origine |
|------------|--------------------|-----------------|-|
| **Hosp. /100K** | `hosp_rate` | `hosp / population × 100 000` | ⚙️ Calculé Phase 2 |
| **Nouv. hosp. /100K** | `new_hosp_rate` | `new_hosp / population × 100 000` | ⚙️ Calculé Phase 2 |
| **Positivité 7j (%)** | `tx_pos_7j` | `rolling_mean(P/T×100, 7)` par dept | ⚙️ Calculé Phase 2 |
| **Couv. vaccinale (%)** | `couv_complet` | % population vaccinée complète | 📥 CSV brut VAC-SI |
| **Pression réa (%)** | `icu_pressure` | `rea / hosp × 100` | ⚙️ Calculé Phase 2 |
| **Létalité hosp. (%)** | `cfr` | `dc / (dc + rad) × 100` | ⚙️ Calculé Phase 2 |

> **Aucun de ces 6 indicateurs n'est produit par un modèle ML.** Ils viennent tous de la Phase 2 (soit depuis les CSV bruts, soit calculés par feature engineering).

---

## 4. Classement Top 15 départements (barres horizontales)

Même chose que la carte : l'**indicateur sélectionné** détermine la valeur des barres. La **couleur des barres** est la seule information venant du clustering.

| Élément visuel | Données | Calcul ou source | Origine |
|----------------|---------|-----------------|---------|
| **Hauteur de la barre** | Valeur de l'indicateur sélectionné | Voir tableau section 3 | ⚙️ Phase 2 ou 📥 CSV brut |
| **Ordre** (haut = plus élevé) | Tri décroissant par valeur | Calculé dans le navigateur | — |
| **Couleur de la barre** | `kmeans_cluster` (0 ou 1) | Label K-Means | 🔵 Clustering Phase 3 |
| **Label** ("01 — Ain") | `dep` + nom du GeoJSON | — | 📥 GeoJSON géographique |

### D'où vient le label de couleur `kmeans_cluster` ?

C'est le **seul endroit dans le dashboard** où les résultats du clustering Phase 3 sont directement visibles.

Le fichier `global_cluster_labels.csv` (Phase 3) contient pour chaque département :

```
dep, kmeans_cluster, dbscan_cluster, is_dom, pc1, pc2, pc3
01,  1,              1,              0,       ...
75,  0,              0,              0,       ...
```

- `kmeans_cluster = 0` → barre rouge (25 depts, forte sévérité hospitalière)
- `kmeans_cluster = 1` → barre bleue (71 depts, sévérité modérée)

Ce label est **calculé une seule fois** par K-Means sur les données agrégées par vague (Phase 3), puis utilisé comme décoration visuelle dans le dashboard. Il **ne change pas** quand on déplace le curseur de date.

---

## 5. Scatter Positivité vs Hospitalisation

| Axe / Élément | Données | Origine |
|---------------|---------|---------|
| **Axe X** — Positivité 7j (%) | `tx_pos_7j` à la date du curseur | ⚙️ Calculé Phase 2 |
| **Axe Y** — Hosp. /100K | `hosp_rate` à la date du curseur | ⚙️ Calculé Phase 2 |
| **Couleur du point** | `kmeans_cluster` | 🔵 Clustering Phase 3 |
| **Taille du point** | Fixe (9 px) | — |
| **Tooltip** | Nom dept + valeurs X et Y | Phase 2 + GeoJSON |

---

## 6. Courbe de détail département (affichée au clic)

Quand on clique sur un département dans la carte ou les barres, une courbe temporelle apparaît avec 5 séries :

| Série | Colonne | Origine |
|-------|---------|---------|
| **Hosp. /100K** (bleu) | `hosp_rate` | ⚙️ Calculé Phase 2 |
| **Nouv. hosp. /100K** (orange tirets) | `new_hosp_rate` | ⚙️ Calculé Phase 2 |
| **Pression réa %** (rouge tirets) | `icu_pressure` | ⚙️ Calculé Phase 2 |
| **Positivité 7j %** (violet, axe droit) | `tx_pos_7j` | ⚙️ Calculé Phase 2 |
| **Couv. vaccinale %** (vert tirets, axe droit) | `couv_complet` | 📥 CSV brut VAC-SI |

---

## 7. Badge de vague épidémique

| Élément | Données | Origine |
|---------|---------|---------|
| Label affiché ("Vague 5 — hiver 2021/22…") | `wave_periods.csv` | ⚙️ Calculé Phase 2 (dates définies manuellement) |
| Activation | Comparaison date curseur vs plages de vague | Calculé dans le navigateur |

---

## Récapitulatif global

```
╔══════════════════════════════════════════════════════════════════╗
║  DANS LE DASHBOARD — QUI VIENT D'OÙ ?                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📥 CSV BRUT (SPF/VAC-SI/SI-DEP)         ← utilisé directement ║
║     hosp, rea, dc, rad                                          ║
║     couv_complet, couv_dose1                                    ║
║     tx_pos (P/T × 100, calculé en Phase 1 depuis P et T)        ║
║                                                                  ║
║  ⚙️  CALCULÉ EN PHASE 2                   ← feature engineering  ║
║     new_hosp = diff(hosp).clip(0)                               ║
║     new_hosp_7j = rolling_mean(new_hosp, 7)                     ║
║     hosp_rate = hosp / population × 100K                        ║
║     new_hosp_rate = new_hosp / population × 100K                ║
║     tx_pos_7j = rolling_mean(tx_pos, 7)                         ║
║     cfr = dc / (dc + rad) × 100                                 ║
║     icu_pressure = rea / hosp × 100                             ║
║     couv_complet_lag14 = shift(couv_complet, 14j)               ║
║     wave = 0..7 (plages de dates fixes)                         ║
║                                                                  ║
║  🔵 CLUSTERING PHASE 3 (K-Means)         ← couleur des barres   ║
║     kmeans_cluster = 0 (25 depts, forte sévérité)               ║
║     kmeans_cluster = 1 (71 depts, sévérité modérée)             ║
║     Calculé une seule fois, statique dans le dashboard          ║
║                                                                  ║
║  🤖 ML PHASE 4 (HistGradientBoosting)    ← PAS dans le dashboard║
║     Le modèle ML (best_model.pkl) n'est PAS utilisé             ║
║     directement dans le dashboard.                              ║
║     Il sert à prédire hosp_rate sur de nouvelles données.       ║
║     Ses résultats (predictions_test.csv) ne sont pas            ║
║     chargés dans phase5_dashboard.py.                           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Ce que le modèle ML ne fait PAS dans le dashboard

**Le modèle ML (Phase 4) n'est pas du tout appelé dans le tableau de bord.** Le dashboard affiche des données historiques réelles, pas des prédictions.

Le modèle ML sert à un usage différent : prédire `hosp_rate` **sur de nouvelles données** (données futures non observées). Voici comment l'utiliser en dehors du dashboard :

```python
import pickle, pandas as pd

# Charger le modèle entraîné
with open("data/models/best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipe     = bundle["model"]     # Pipeline(RobustScaler + HistGradientBoosting)
features = bundle["features"]  # Liste des 15 features dans le bon ordre

# Préparer de nouvelles données avec les mêmes features
# Exemple : prédire hosp_rate pour le 01/09/2022 pour le dept 75
df_new = pd.DataFrame([{
    "hosp_rate_lag7":       45.2,   # hosp_rate 7 jours avant
    "hosp_rate_lag14":      38.1,   # hosp_rate 14 jours avant
    "new_hosp_rate_lag7":   1.8,
    "new_hosp_rate_lag1":   2.1,
    "tx_pos_7j_lag7":       18.4,
    "couv_complet_lag14":   78.2,
    "rea_rate":             3.2,
    "TO":                   0.0,    # non disponible → 0
    "icu_pressure":         11.5,
    "wave":                 6,
    "kmeans_cluster":       0,
    "is_dom":               0,
    "month":                9,
    "year":                 2022,
    "day_of_week":          3,
}])

X_new = df_new[features].to_numpy(dtype=float)
y_pred = pipe.predict(X_new)
print(f"Hosp. /100K prédit : {y_pred[0]:.2f}")
```

---

## Tableau de correspondance complet

| Variable dans le dashboard | Type | Formule ou source | Phase |
|---------------------------|------|------------------|-------|
| `hosp` | 📥 Brut | SPF hospitalisations.csv (sexe=0) | Phase 1 |
| `rea` | 📥 Brut | SPF hospitalisations.csv | Phase 1 |
| `dc` | 📥 Brut | SPF hospitalisations.csv | Phase 1 |
| `rad` | 📥 Brut | SPF hospitalisations.csv | Phase 1 |
| `couv_complet` | 📥 Brut | VAC-SI vaccination_dept.csv | Phase 1 |
| `tx_pos` | ⚙️ Calculé P1 | `P / T × 100` (P=positifs, T=testés, SI-DEP) | Phase 1 |
| `new_hosp` | ⚙️ Calculé P2 | `diff(hosp).clip(lower=0)` par dept | Phase 2 |
| `new_rea` | ⚙️ Calculé P2 | `diff(rea).clip(lower=0)` par dept | Phase 2 |
| `new_dc` | ⚙️ Calculé P2 | `diff(dc).clip(lower=0)` par dept | Phase 2 |
| `new_hosp_7j` | ⚙️ Calculé P2 | `rolling_mean(new_hosp, 7)` par dept | Phase 2 |
| `hosp_rate` | ⚙️ Calculé P2 | `hosp / population × 100 000` | Phase 2 |
| `rea_rate` | ⚙️ Calculé P2 | `rea / population × 100 000` | Phase 2 |
| `new_hosp_rate` | ⚙️ Calculé P2 | `new_hosp / population × 100 000` | Phase 2 |
| `tx_pos_7j` | ⚙️ Calculé P2 | `rolling_mean(tx_pos, 7)` par dept | Phase 2 |
| `cfr` | ⚙️ Calculé P2 | `dc / (dc + rad) × 100` | Phase 2 |
| `icu_pressure` | ⚙️ Calculé P2 | `rea / hosp × 100` | Phase 2 |
| `couv_complet_lag14` | ⚙️ Calculé P2 | `groupby(dep).shift(14)` sur `couv_complet` | Phase 2 |
| `wave` (1–7 ou 0) | ⚙️ Calculé P2 | Plages de dates fixes fusionnées | Phase 2 |
| `is_dom` | ⚙️ Calculé P2 | `dep ∈ {971, 972, 973, 974, 976}` | Phase 2 |
| `kmeans_cluster` | 🔵 Clustering P3 | K-Means k=2 sur 15 features agrégées par vague | Phase 3 |
| `pc1`, `pc2`, `pc3` | 🔵 Clustering P3 | Coordonnées PCA 2D/3D | Phase 3 |
| `hosp_rate` prédit | 🤖 ML P4 | HistGradientBoosting (best_model.pkl) | Phase 4 |

> Note : `hosp_rate` prédit (Phase 4) **n'est pas affiché** dans le dashboard. Il est sauvegardé dans `predictions_test.csv` et `best_model.pkl` mais non chargé par `phase5_dashboard.py`.
