# Chatbot COVID-19

Application Streamlit de prédiction pédagogique du risque COVID-19 à partir d'un questionnaire patient. Le projet contient les notebooks de préparation des données, d'entraînement des modèles et une interface web qui charge le meilleur modèle sauvegardé.

## Structure du projet

```text
Chatbot/
  app_streamlit.py
  requirements.txt
  data/
    layer_bronze_data_brut/          donnees sources
    layer_silver_data_transform/     donnees nettoyees
    layer_gold_data_model/           donnees modele, graphiques et artefacts
      modele_final/
        meilleur_modele.joblib
        metadata.json
  src/
    clusturing_covid/                notebooks de lecture et transformation
    prediction_covid/                notebooks de prediction et evaluation
```

## Installation

Depuis le dossier `Chatbot` :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sur macOS ou Linux, utilisez `source .venv/bin/activate` pour activer l'environnement virtuel.

## Lancer l'application

```bash
streamlit run app_streamlit.py
```

L'application doit être lancée depuis la racine du projet `Chatbot`, car elle charge le modèle depuis `data/layer_gold_data_model/modele_final/`.

## Fonctionnement

Le formulaire Streamlit collecte les informations générales, symptômes et comorbidités d'un patient. Ces réponses sont transformées en variables numériques, réordonnées selon la liste `features` présente dans `metadata.json`, puis envoyées au modèle sauvegardé `meilleur_modele.joblib`.

Le résultat affiché est une probabilité estimée d'infection et une classe prédite selon un seuil de 0.5. Cette application est un outil pédagogique de Machine Learning et ne remplace pas un avis médical.

## Données et modèle

Les données suivent une logique bronze, silver et gold :

- `layer_bronze_data_brut` : données sources.
- `layer_silver_data_transform` : données nettoyées.
- `layer_gold_data_model` : jeux prêts pour la modélisation, graphiques d'évaluation et modèle final.

Le README détaillé de l'évaluation des modèles se trouve dans `src/prediction_covid/README.md`.
