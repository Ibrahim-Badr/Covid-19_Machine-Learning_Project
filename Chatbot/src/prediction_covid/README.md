# Analyse et Évaluation des Modèles de Prédiction Covid-19
 
Ce module regroupe l'ensemble de la démarche de Machine Learning visant à prédire la présence du Covid-19 à partir de données cliniques mexicaines, incluant la préparation des données, la modélisation prédictive, l'analyse par clustering et une interface chatbot interactive.
 
---
 
## Fichiers du dossier
 
| Fichier | Description |
|---|---|
| `prediction_covid.ipynb` | Notebook principal : entraînement, optimisation et évaluation des modèles ML |
| `questionnaire_covid.ipynb` | Interface de questionnaire interactif pour tester le modèle sur un patient |
| `README.md` | Ce document |
 
---
 
## Dataset
 
- **Source** : Données officielles mexicaines de suivi Covid-19
- **Taille** : 260 919 patients
- **Features** : symptômes cliniques, comorbidités (diabète, hypertension, obésité, etc.), âge, sexe
- **Déséquilibre de classes** : forte majorité de cas négatifs → nécessite des stratégies d'équilibrage
---
 
## 1. Approche Initiale et Paramètres Basiques
 
Dans un premier temps, les différents algorithmes (XGBoost, Random Forest, Régression Logistique, LightGBM) ont été entraînés avec leurs hyperparamètres par défaut. Face au déséquilibre important des classes, les résultats initiaux n'étaient pas satisfaisants, notamment en ce qui concerne le score F1.
 
---
 
## 2. Optimisation via GridSearchCV
 
Pour améliorer les performances, une recherche exhaustive des meilleurs hyperparamètres a été menée via `GridSearchCV` et `RandomizedSearchCV` sur un sous-échantillon de 30 000 patients (pour des raisons de temps de calcul), avec une validation croisée à 3 folds.
 
Afin de pallier le déséquilibre des classes :
- `scale_pos_weight=1.57` pour XGBoost
- `class_weight='balanced'` pour les autres modèles
Malgré ces ajustements, les scores n'ont pas connu d'amélioration drastique, soulignant la complexité inhérente aux données.
 
---
 
## 3. Analyse par Clustering (Non Supervisé)
 
En complément de la classification supervisée, une analyse par clustering a été réalisée pour explorer la structure naturelle des données sans utiliser les labels de diagnostic.
 
### Méthode Elbow (K-Means)
La méthode Elbow (inertie en fonction du nombre de clusters k) a été utilisée pour identifier le nombre optimal de clusters. L'inertie mesure la compacité des clusters : plus elle est faible, plus les points sont proches de leur centroïde.
 
### K-Means
- Regroupement des patients en clusters homogènes selon leurs caractéristiques cliniques
- **Score Silhouette** utilisé pour mesurer la qualité de la séparation entre clusters (entre -1 et 1, plus c'est proche de 1, mieux c'est)
### DBSCAN
- Algorithme basé sur la densité, capable d'identifier des clusters de forme arbitraire et de détecter des outliers (points aberrants classés comme "bruit")
- Particulièrement utile pour identifier des profils atypiques de patients
### Apport du Clustering
Cette analyse non supervisée permet de :
- Confirmer l'existence de groupes distincts de patients (profils à risque vs faible risque)
- Identifier des sous-groupes avec des combinaisons de symptômes spécifiques
- Valider que la structure des données supporte l'approche de classification supervisée
---
 
## 4. Résultats et Évaluation Avancée
 
### 4a. Évaluation sur Split Unique (80% train / 20% test)
 
| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.5164 | 0.4363 | **0.8301** | 0.5720 | 0.6572 | 0.5570 |
| **LightGBM** | 0.5164 | 0.4362 | 0.8289 | 0.5716 | 0.6574 | 0.5562 |
| **Random Forest** | 0.5024 | 0.4298 | **0.8529** | 0.5716 | 0.6574 | 0.5570 |
| **Reg. Logistique** | 0.5170 | 0.4361 | 0.8219 | 0.5698 | 0.6505 | 0.5499 |
 
**Lecture des métriques AUC :**
- **AUC-ROC (~0.65)** : supérieur à l'aléatoire (0.5), mais légèrement faussé par la forte présence de la classe majoritaire. Indique une difficulté persistante à séparer parfaitement les deux classes.
- **PR-AUC (~0.55)** : métrique plus sévère et réaliste face au déséquilibre, car elle se concentre sur la classe minoritaire (cas positifs). Confirme la difficulté à maintenir une bonne Précision.
Tendance forte : **le Recall est très élevé (0.82–0.85)**, au détriment de la Précision (~0.43).
 
### 4b. Validation Croisée Stratifiée (3-Fold) — Dataset Complet
 
Pour s'assurer que les résultats ne sont pas liés au hasard d'un découpage unique, une **validation croisée à 3 folds stratifiés** a été réalisée sur l'intégralité des 260 919 patients. À chaque tour, un fold (~87 000 patients) sert de test et les 2 autres (~174 000) d'entraînement.
 
| Modèle | F1-Score (CV) | std | Precision (CV) | Recall (CV) | AUC-ROC (CV) | PR-AUC (CV) |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.5711** | ±0.0007 | 0.4362 | **0.8267** | **0.6569** | **0.5563** |
| **LightGBM** | 0.5707 | ±0.0007 | **0.4364** | 0.8245 | 0.6565 | 0.5557 |
| **Reg. Logistique** | 0.5701 | ±0.0010 | 0.4363 | 0.8222 | 0.6536 | 0.5516 |
| **Random Forest** | 0.5690 | ±0.0007 | 0.4351 | 0.8223 | 0.6527 | 0.5491 |
 
**Ce que ça confirme :** L'écart-type extrêmement faible (±0.0007 à ±0.0010) prouve que les scores sont stables quel que soit le découpage des données. Les modèles sont robustes — ce n'est pas du hasard lié à un bon split.
 
XGBoost reste le meilleur modèle de façon cohérente.
 
---
 
## 5. Conclusion et Justification Métier (Le Coût de l'Erreur)
 
Notre modèle n'est pas très précis : il a tendance à prédire beaucoup de personnes saines comme porteuses du Covid-19 (faux positifs élevés). **Dans notre contexte de pandémie, ce comportement est un choix assumé**, fondé sur l'analyse du coût de l'erreur :
 
- **Faux Négatif (coût dramatique)** : classer comme "saine" une personne infectée → elle rentre chez elle, risque d'aggraver son état et propage le virus.
- **Faux Positif (coût acceptable)** : convoquer par précaution une personne saine pour un test PCR → du stress et du temps médical mobilisé, mais aucune vie en danger.
Notre compromis actuel — **Recall ~83%, Précision ~43%** — tente de trouver le juste milieu entre sécurité sanitaire maximale et viabilité du système hospitalier.
 
---
 
## 6. Limites et Perspectives
 
### Threshold Tuning
En théorie, augmenter le seuil de décision (par défaut 0.5) permettrait d'améliorer la Précision. Cependant, vu la gravité de la maladie, **nous avons volontairement choisi de ne pas faire baisser le Recall**. Notre modèle agit comme un **premier filet de sécurité très large**.
 
La seule limite à cette approche : si la Précision est trop faible, on risque d'engorger le système de santé avec des patients sains (pénurie de tests, personnel surchargé).
 
### Perspectives d'amélioration
- **SMOTE / Oversampling** : générer des exemples synthétiques de la classe minoritaire pour mieux équilibrer les données
- **Ensemble Methods** : combiner plusieurs modèles pour améliorer la robustesse
- **Features supplémentaires** : intégrer des données biologiques (taux d'oxygène, température) pour affiner les prédictions
- **Déploiement** : le modèle final XGBoost est sauvegardé dans `../data/layer_gold_data_model/modele_final/` et peut être chargé via le chatbot Streamlit
---
 
## 7. Interface Chatbot (Streamlit)
 
Un chatbot interactif permet de tester le modèle en simulant un questionnaire médical. L'application collecte les symptômes et comorbidités du patient, puis retourne une prédiction avec le niveau de risque Covid-19.
 
- Application : `../../app_streamlit.py`
- Questionnaire notebook : `questionnaire_covid.ipynb`
- Modèle chargé : `../data/layer_gold_data_model/modele_final/meilleur_modele.joblib`
