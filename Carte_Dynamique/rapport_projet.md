# Rapport de projet — Pipeline COVID-19 France : Carte Dynamique
  
**Date :** mai 2026  
**Contexte :** Projet de Machine Learning appliqué à l'épidémiologie

---

## I. Présentation générale du projet

### Objectif

Le projet **Carte Dynamique** est un pipeline de data science de bout en bout conçu pour analyser la propagation de l'épidémie de COVID-19 sur l'ensemble des **101 départements français** (France métropolitaine et DOM), sur la période allant de **janvier 2020 à mars 2023**, soit 1 149 jours calendaires.

L'objectif principal était de construire un système capable de :
1. Collecter automatiquement les données épidémiques auprès de sources officielles françaises
2. Produire une série temporelle quotidienne nettoyée et enrichie par département
3. Identifier des groupes de départements au comportement épidémique similaire par clustering
4. Prédire le taux d'hospitalisation à partir d'indicateurs épidémiques décalés
5. Restituer l'ensemble dans un tableau de bord interactif exploitable sans serveur

### Architecture séquentielle

Le projet est structuré en cinq phases séquentielles, chaque phase consommant les sorties de la précédente :

```
Phase 1          Phase 2          Phase 3           Phase 4           Phase 5
─────────        ─────────        ─────────         ─────────         ─────────
17 sources   →   master_dept  →   PCA +         →   Régression    →   Dashboard
réseau           _daily.csv       K-Means           hosp_rate         HTML
                                  DBSCAN            prédictive        interactif
```

---

## II. Phase 1 — Acquisition des données

### Sources collectées

La première phase du pipeline réalise l'acquisition automatisée de **17 sources de données** issues d'organismes officiels français : SPF (Santé Publique France), VAC-SI, SI-DEP, INSEE, DREES, Ameli, ainsi que des dépôts GitHub publics (OpenCovid19-FR, GeoJSON France).

Les données couvrent :
- **Hospitalisations** : stock de patients hospitalisés, en réanimation, décédés et sortis (SPF, sexe=0)
- **Tests virologiques** : taux de positivité hebdomadaire puis quotidien (SI-DEP)
- **Vaccination** : couverture par département (VAC-SI), déclinée par âge et par marque
- **Indicateurs épidémiques** : taux de reproduction R0, taux d'incidence Ti, taux d'occupation réanimation TO
- **Géographie** : polygones GeoJSON des départements et régions

### Traitement post-acquisition

Plusieurs transformations clés sont appliquées dès cette phase :
- **Normalisation des codes département** : fonction `normalize_dep()` garantissant le format standard à 2 caractères, avec gestion des cas particuliers (Corse 2A/2B, DOM 971–976)
- **Décalage vaccinal de 14 jours** : la variable `couv_complet_lag14` modélise le délai d'acquisition de l'immunité post-vaccination
- **Calcul de taux pour 100 000 habitants** : application aux indicateurs hospitaliers à partir de la table de population INSEE 2020
- **Résolution dynamique d'URL** : la fonction `fetch_latest_datagouv_url()` interroge l'API data.gouv.fr pour obtenir toujours la ressource la plus récente, évitant les URLs obsolètes

La robustesse est assurée par des blocs `try/except` autour de chaque source non critique, permettant au pipeline de continuer même en cas d'indisponibilité réseau.

---

## III. Phase 2 — Prétraitement et Feature Engineering

### Construction du dataset maître

La phase 2 fusionne l'ensemble des sources en un unique dataset quotidien par département : `master_dept_daily.csv` contenant **113 174 lignes × 47 colonnes**.

Les principales étapes sont :
1. **Base hospitalière** (SPF) : calcul des flux journaliers (`new_hosp`, `new_rea`, `new_dc`) par différentiation des stocks, avec clipping à zéro pour corriger les corrections rétroactives officielles
2. **Unification SI-DEP** : concaténation de deux formats (hebdomadaire 2020–2022 et quotidien 2022–2023) avec dédoublonnage sur `(dep, jour)`
3. **Moyennes glissantes à 7 jours** : lissage des indicateurs bruités (hosp, positivité, R0)
4. **Fusion principale** : 4 jointures LEFT sur `[dep, jour]`, avec comblement de la période pré-SPF (avant mars 2020) par les données OpenCovid19-FR

### Features dérivées

Des indicateurs épidémiologiques composites sont calculés :

| Feature | Formule | Interprétation |
|---------|---------|----------------|
| `cfr` | `dc / (dc + rad) × 100` | Létalité hospitalière |
| `icu_pressure` | `rea / hosp × 100` | Pression sur la réanimation |
| `tx_pos_adj_vacc` | `tx_pos × (1 − couv_complet_lag14/100)` | Positivité corrigée par la vaccination |

### Détection des vagues épidémiques

7 vagues sont définies par des plages de dates calendaires fixes, couvrant les variants successifs (souche originale, Alpha, Delta, Omicron BA.1/BA.2/BA.5/BQ.1). L'assignation est réalisée de façon vectorisée (dictionnaire `date → vague` + merge) pour éviter une boucle Python sur 113K lignes.

### Matrice de features pour le clustering

Une agrégation sur les périodes de vague produit `dept_clusters_features.csv` : **101 départements × 15 features** (moyennes et maxima de `hosp_rate`, `rea_rate`, `cfr`, `tx_pos`, `couv_complet`). R0 est exclu (seulement 9,5% de taux de remplissage).

---

## IV. Phase 3 — Clustering épidémiologique

### Méthodologie

La phase 3 identifie des groupes de départements aux profils épidémiques similaires sur la durée totale de l'épidémie.

**Prétraitement** : un `RobustScaler` (médiane + IQR) est appliqué avant tout, afin de ne pas laisser les départements atypiques (Paris, DOM) fausser la normalisation.

**Analyse en Composantes Principales** : la PCA révèle que 7 composantes suffisent à capturer 90% de la variance. PC1 représente l'axe de sévérité hospitalière globale (chargements dominants : `rea_rate_max` = +0,42, `hosp_rate_mean` = +0,37) ; PC2 capture la dynamique des nouvelles admissions.

### Résultats K-Means

La sélection du nombre optimal de clusters k est déterminée par vote majoritaire entre trois critères : méthode du coude, score de silhouette et indice de Calinski-Harabász. Le résultat est **k = 2**, avec les profils suivants :

| Cluster | Taille | Hosp. moyenne (/100K) | Caractérisation |
|---------|--------|-----------------------|-----------------|
| **0** | 25 depts | 37,1 | Forte sévérité, densité urbaine élevée (Île-de-France, PACA, grandes métropoles) |
| **1** | 71 depts | 22,5 | Impact modéré, meilleure couverture vaccinale (reste de la métropole) |

*Score de silhouette : 0,24 — séparation modérée, reflétant la réalité épidémique française.*

### DBSCAN complémentaire

Un DBSCAN dans l'espace PCA (eps=2,0, min_samples=3) identifie **16 départements outliers** aux profils trop atypiques pour appartenir à une densité locale. Les 5 DOM sont traités séparément avant d'être réassignés au cluster K-Means métropolitain le plus proche.

---

## V. Phase 4 — Modèles de Machine Learning

### Problème de régression

La variable cible est `hosp_rate` (taux d'hospitalisation pour 100 000 habitants). Ce taux est la variable de synthèse la plus opérationnelle : il intègre la sévérité clinique et la démographie, et est directement utilisé pour dimensionner les capacités hospitalières.

### Feature engineering : variables retardées (lags)

15 features sont construites en intégrant des décalages temporels justifiés épidémiologiquement :
- Lag 7 et 14 jours sur `hosp_rate` (autorégressif)
- Lag 7 jours sur `new_hosp_rate` et `tx_pos_7j` (délai test → hospitalisation)
- Variables contextuelles : vague, cluster K-Means, indicateur DOM, saisonnalité

Le dataset après `dropna()` couvre la période **janvier 2021 – août 2022** (60 215 lignes), la vaccination commençant en janvier 2021 conditionnant le début effectif.

### Protocole d'évaluation

Le découpage est strictement **chronologique** (80/20) pour respecter l'intégrité temporelle, et la validation croisée utilise `TimeSeriesSplit(5 folds)` pour éviter tout data leakage temporel.

### Comparaison des modèles

8 modèles sont entraînés, tous encapsulés dans un `Pipeline(RobustScaler + estimateur)` :

| Modèle | R² test | MAE | Overfit |
|--------|---------|-----|---------|
| **HistGradientBoosting** | **0,958** | 1,69 | 0,036 |
| WeightedVoting (ensemble) | 0,938 | 2,13 | 0,055 |
| Random Forest | 0,917 | 2,45 | 0,074 |
| Lasso | 0,911 | 2,68 | 0,048 |
| Ridge | 0,790 | 4,18 | 0,174 |
| Baseline (médiane) | −0,036 | 9,57 | — |

### Résultat final après tuning

Le meilleur modèle (HistGradientBoosting) est soumis à un `RandomizedSearchCV` (40 itérations × 5 folds) sur 5 hyperparamètres. Le modèle tuné atteint :

> **R² = 0,974 — MAE = 1,14 hosp./100K — RMSE = 1,95 — MAPE = 5,80 %**

Ces performances confirment que la prédiction du taux d'hospitalisation est principalement autoregressive sur un horizon court, les features `hosp_rate_lag7` et `hosp_rate_lag14` dominant l'importance des variables.

Le modèle final est sérialisé dans `data/models/best_model.pkl` comme un dictionnaire contenant le pipeline complet, la liste des features et les métriques de performance.

---

## VI. Phase 5 — Tableau de bord interactif

Le tableau de bord est généré sous la forme d'un **fichier HTML autonome** (~1,2 Mo) ne nécessitant aucun serveur : toutes les données sont sérialisées en JSON et embarquées dans le fichier, le rendu graphique étant assuré par Plotly.js.

L'interface comprend :
- Un **curseur de date** (167 positions hebdomadaires) pilotant l'ensemble des visualisations simultanément
- **5 cartes KPI** avec flèches de tendance (hospitalisés, réanimation, décès, positivité, vaccination)
- Un **graphique d'évolution nationale** avec fonds colorés par vague épidémique
- Une **carte choroplèthe** de France (sans Mapbox, sans token API) avec sélecteur d'indicateur
- Un **classement Top 15 départements** coloré par cluster K-Means
- Un **scatter plot** positivité vs hospitalisation par département
- Un **panel de détail** par département avec série temporelle complète et fonds de vague

Les mises à jour JavaScript utilisent `Plotly.restyle` et `Plotly.relayout` (sans full redraw) pour garantir une fluidité optimale lors du déplacement du curseur.

---

## VII. Conclusions

Ce projet constitue un pipeline de data science complet allant de la collecte automatisée de données brutes jusqu'à la visualisation interactive, en passant par le feature engineering, le clustering non supervisé et la modélisation prédictive.

Les principaux résultats obtenus sont :
- Un dataset maître de 113 174 observations couvrant 101 départements sur 3 ans
- Une segmentation épidémiologique en 2 clusters interprétables géographiquement
- Un modèle de régression atteignant R² = 0,974 avec une erreur absolue moyenne de 1,14 hospitalisation pour 100 000 habitants
- Un tableau de bord interactif autonome permettant d'explorer l'évolution épidémique par date et par indicateur

Ce travail illustre l'application de techniques de Machine Learning (clustering, régression supervisée, ensemble methods, hyperparameter tuning) à une problématique de santé publique réelle, avec une attention particulière portée à la rigueur méthodologique (chronological split, TimeSeriesSplit CV, prévention du data leakage).

---

*Ce rapport synthétise le travail réalisé dans le cadre du projet Carte Dynamique COVID-19 France.*
