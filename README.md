# Covid-19 Machine-Learning Project

This repository contains two related projects:

- `Carte_Dynamique`: a COVID-19 data pipeline and visualization project for French departments.
- `Chatbot`: a Streamlit app that uses a trained model to provide a pedagogical COVID-19 prediction.

## Repository Layout

```text
Covid-19_Machine-Learning_Project/
  Carte_Dynamique/
  Chatbot/
```

## Carte_Dynamique

Python data-science pipeline that fetches, preprocesses, clusters, and models COVID-19 epidemiological data.

Main entry points are in `Carte_Dynamique/src/`:

- `phase1_data_acquisition 1.py`
- `phase2_preprocessing.py`
- `phase3_clustering.py`
- `phase4_Ml-models.py`
- `phase5_dashboard.py`
- `phase5_map.py`

Run it from the `Carte_Dynamique/src` folder so the relative data paths resolve correctly.

## Chatbot

Streamlit application for a questionnaire-based COVID-19 prediction demo.

Main entry point:

- `Chatbot/app_streamlit.py`

Run it from the `Chatbot` folder:

```bash
streamlit run app_streamlit.py
```

## Shared Notes

- Both projects use Python and separate `requirements.txt` files in their own folders.
- Large generated files, notebook caches, virtual environments, and local secrets are ignored at the repository root through `.gitignore`.
