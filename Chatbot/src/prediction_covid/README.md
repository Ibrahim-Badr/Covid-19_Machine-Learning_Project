# Analyse et Évaluation des Modèles de Prédiction Covid-19

Ce document détaille la démarche d'optimisation et d'évaluation de nos modèles de Machine Learning visant à prédire la présence du Covid-19. 

## 1. Approche Initiale et Paramètres Basiques
Dans un premier temps, les différents algorithmes (XGBoost, Random Forest, Régression Logistique, etc.) ont été entraînés avec leurs hyperparamètres par défaut. Face au déséquilibre important des classes dans notre jeu de données, les résultats initiaux n'étaient pas satisfaisants, notamment en ce qui concerne le score F1.

## 2. Optimisation via GridSearchCV
Pour tenter d'améliorer les performances, une recherche exhaustive des meilleurs hyperparamètres a été menée à l'aide de `GridSearchCV` et `RandomizedSearchCV`. La recherche a été effectuée sur un sous-échantillon représentatif de 30 000 patients (pour des raisons de temps de calcul), avec une validation croisée à 3 folds en interne.

Afin de pallier le déséquilibre des classes, nous avons également intégré des paramètres de pénalisation :
- `scale_pos_weight=1.57` pour XGBoost.
- `class_weight='balanced'` pour les autres modèles.

Malgré ces ajustements approfondis, les scores globaux n'ont pas connu d'amélioration drastique, soulignant la complexité inhérente aux données.

## 3. Résultats et Évaluation Avancée

### 3a. Évaluation sur Split Unique (80% train / 20% test)

Voici un récapitulatif des scores obtenus par nos meilleurs modèles après optimisation, sur un découpage unique 80/20 :

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.5164 | 0.4363 | **0.8301** | 0.5720 | 0.6572 | 0.5570 |
| **LightGBM** | 0.5164 | 0.4362 | 0.8289 | 0.5716 | 0.6574 | 0.5562 |
| **Random Forest** | 0.5024 | 0.4298 | **0.8529** | 0.5716 | 0.6574 | 0.5570 |
| **Reg. Logistique**| 0.5170 | 0.4361 | 0.8219 | 0.5698 | 0.6505 | 0.5499 |

Pour mieux comprendre le comportement de nos modèles, nous nous sommes appuyés sur l'analyse "Under the Curve" (AUC-ROC et PR-AUC) :
- **AUC-ROC (Area Under the ROC Curve)** : Nos modèles tournent autour de 0.65. L'AUC-ROC est supérieur à l'aléatoire (0.5), mais il peut être légèrement faussé par la forte présence de la classe majoritaire (les cas négatifs). Cela indique une difficulté persistante à séparer parfaitement les deux classes.
- **PR-AUC (Precision-Recall Area Under Curve)** : Avec un score d'environ 0.55, cette métrique est beaucoup plus sévère et réaliste face au déséquilibre, car elle se concentre uniquement sur la classe minoritaire. Elle confirme que notre modèle a du mal à maintenir une bonne Précision.

En observant les métriques détaillées dans le tableau, on constate cependant une tendance forte : **le Recall est très élevé (entre 0.82 et 0.85)**, au détriment de la Précision (autour de 0.43).

### 3b. Validation Croisée Stratifiée (3-Fold) — Dataset Complet

Pour s'assurer que les performances ci-dessus ne sont pas liées au hasard du découpage 80/20, on a également évalué les modèles via une **validation croisée à 3 folds** sur l'intégralité des 260 000 patients. À chaque tour, un fold différent (~87 000 patients) sert de test et les 2 autres (~174 000) servent d'entrainement — un peu comme faire le split 3 fois d'affilée en changeant la partie test à chaque fois.

Résultats obtenus sur l'ensemble des 260 919 patients, 3 folds stratifiés :

| Modèle | F1-Score (CV) | std | Precision (CV) | Recall (CV) | AUC-ROC (CV) | PR-AUC (CV) |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.5711** | ±0.0007 | 0.4362 | **0.8267** | **0.6569** | **0.5563** |
| **LightGBM** | 0.5707 | ±0.0007 | **0.4364** | 0.8245 | 0.6565 | 0.5557 |
| **Reg. Logistique** | 0.5701 | ±0.0010 | 0.4363 | 0.8222 | 0.6536 | 0.5516 |
| **Random Forest** | 0.5690 | ±0.0007 | 0.4351 | 0.8223 | 0.6527 | 0.5491 |

**Ce que ça confirme :** L'écart-type est extrêmement faible (±0.0007 à ±0.0010) pour tous les modèles. Ça veut dire que peu importe comment on découpe les données, les scores sont quasi-identiques. Nos modèles sont robustes, ce n'est pas du hasard lié à un bon split.

XGBoost reste le meilleur modèle de façon cohérente entre le split unique et la cross-validation.

## 4. Conclusion et Justification Métier (Le Coût de l'Erreur)

Concrètement, notre modèle n'est pas très précis : il a tendance à prédire beaucoup de personnes saines comme étant porteuses du Covid-19 (taux de faux positifs élevé). 

**Néanmoins, dans notre contexte de pandémie, ce comportement est un choix assumé.**
Cette réflexion s'appuie sur ce qu'on appelle l'évaluation du coût de l'erreur ("Cost of Error") :
- **Le coût d'un Faux Négatif est dramatique :** Classer comme "saine" une personne infectée signifie qu'elle rentre chez elle, risque de voir son état s'aggraver sans soins, et surtout propage le virus au sein de la population.
- **Le coût d'un Faux Positif est acceptable :** Convoquer une personne saine par précaution pour qu'elle passe un test PCR engendre du stress et mobilise du temps médical, mais ne met aucune vie en danger.

### Limites et Perspectives (Threshold Tuning)
En théorie, pour améliorer la Précision, il serait possible d'utiliser le **Threshold Tuning** (augmenter le seuil de décision par défaut de 0.5 pour que le modèle soit plus strict avant de déclarer un patient malade). 
Cependant, vu la gravité de la maladie à l'époque, **nous avons volontairement choisi de ne pas faire baisser le Recall**. 

Notre modèle agit comme un **premier filet de sécurité très large**. La seule limite à cette approche (maximiser le Recall au détriment de la précision) est le risque logistique : si la précision est trop faible, on risque d'engorger le système de santé avec des patients sains (pénurie de tests, personnel surchargé). Notre compromis actuel (Recall à ~83% et Précision à ~43%) tente de trouver le juste milieu entre une sécurité sanitaire maximale et la viabilité de nos hôpitaux.
