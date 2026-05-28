import streamlit as st
import pandas as pd
import joblib
import json
import os

# Configuration de la page
st.set_page_config(page_title="Prédiction COVID-19", page_icon="🦠", layout="centered")

st.title("🤖 Chatbot de Diagnostic COVID-19")
st.write("Répondez aux questions ci-dessous en sélectionnant les options. L'IA calculera ensuite votre probabilité d'infection.")

# ── Chargement du modèle en cache pour éviter de le recharger à chaque clic ──
@st.cache_resource
def load_model():
    MODEL_DIR = 'data/layer_gold_data_model/modele_final'
    model_path = f'{MODEL_DIR}/meilleur_modele.joblib'
    meta_path = f'{MODEL_DIR}/metadata.json'
    
    if not os.path.exists(model_path):
        return None, None
        
    modele = joblib.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return modele, metadata

modele, metadata = load_model()

if modele is None:
    st.error("Le modèle n'a pas été trouvé. Assurez-vous de lancer l'application depuis la racine du projet.")
    st.stop()

st.success(f"Modèle prêt : {metadata['nom_modele']} (F1-Score: {metadata['metriques']['f1_score']:.2f})")

# ── Formulaire Interactif ──
with st.form("covid_form"):
    st.subheader("1. Informations Générales")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sexe_input = st.selectbox("Quel est votre sexe ?", ["Femme", "Homme"])
        sexe = 0 if sexe_input == "Femme" else 1
        
    with col2:
        age_input = st.number_input("Quel est votre âge ?", min_value=0, max_value=120, value=30)
        age_enfant = 1 if age_input <= 17 else 0
        age_jeune = 1 if 18 <= age_input <= 39 else 0
        age_adulte = 1 if 40 <= age_input <= 59 else 0
        age_senior = 1 if age_input >= 60 else 0
        
    with col3:
        hosp_input = st.selectbox("Avez-vous été hospitalisé(e) ?", ["Non", "Oui"])
        type_patient = 1 if hosp_input == "Oui" else 0

    st.subheader("2. Symptômes et Antécédents")
    col4, col5 = st.columns(2)
    
    with col4:
        pneumonie = 1 if st.selectbox("Avez-vous une pneumonie ?", ["Non", "Oui"]) == "Oui" else 0
        diabete = 1 if st.selectbox("Êtes-vous diabétique ?", ["Non", "Oui"]) == "Oui" else 0
        bpco = 1 if st.selectbox("Avez-vous une BPCO ?", ["Non", "Oui"]) == "Oui" else 0
        asthme = 1 if st.selectbox("Avez-vous de l'asthme ?", ["Non", "Oui"]) == "Oui" else 0
        immuno = 1 if st.selectbox("Êtes-vous immunodéprimé(e) ?", ["Non", "Oui"]) == "Oui" else 0
        
    with col5:
        hypertension = 1 if st.selectbox("Faites-vous de l'hypertension ?", ["Non", "Oui"]) == "Oui" else 0
        cardio = 1 if st.selectbox("Maladie cardiovasculaire ?", ["Non", "Oui"]) == "Oui" else 0
        obesite = 1 if st.selectbox("Êtes-vous en situation d'obésité ?", ["Non", "Oui"]) == "Oui" else 0
        renale = 1 if st.selectbox("Insuffisance rénale chronique ?", ["Non", "Oui"]) == "Oui" else 0
        tabac = 1 if st.selectbox("Êtes-vous fumeur/fumeuse ?", ["Non", "Oui"]) == "Oui" else 0
        
    autre_comor = 1 if st.selectbox("Avez-vous une autre comorbidité non listée ?", ["Non", "Oui"]) == "Oui" else 0

    # Le bouton pour valider
    submit_button = st.form_submit_button(label="Lancer l'analyse 🚀")

# ── Actions après le clic sur Valider ──
if submit_button:
    # Construction du DataFrame avec les noms de colonnes exacts
    patient_data = pd.DataFrame([{
        'Sexe': sexe,
        'Type_de_patient': type_patient,
        'Pneumonie': pneumonie,
        'Diabète': diabete,
        'Bronchopneumopathie_chronique_obstructive': bpco,
        'Asthme': asthme,
        'Immunosuppression': immuno,
        'Hypertension': hypertension,
        'Autre_comorbidité': autre_comor,
        'Maladie_cardiovasculaire': cardio,
        'Obésité': obesite,
        'Insuffisance_rénale_chronique': renale,
        'Tabagisme': tabac,
        'Tranche_Age_tranche_age_enfant': age_enfant,
        'Tranche_Age_tranche_age_jeune': age_jeune,
        'Tranche_Age_tranche_age_adulte': age_adulte,
        'Tranche_Age_tranche_age_senior': age_senior
    }])
    
    # Réordonner comme le modèle l'attend
    patient_data = patient_data[metadata['features']]
    
    # Prédiction
    proba = modele.predict_proba(patient_data)[0][1]
    
    # Affichage des résultats
    st.markdown("---")
    st.subheader("🔍 Résultat de l'analyse IA")
    
    if proba >= 0.5:
        st.error(f"**Diagnostic Prédictif : POSITIF (COVID+)**")
    else:
        st.success(f"**Diagnostic Prédictif : NÉGATIF (COVID-)**")
        
    st.write(f"Probabilité d'infection estimée par le modèle : **{proba*100:.1f}%**")
    
    # Barre de progression visuelle native Streamlit
    st.progress(float(proba))
    
    st.caption("⚠️ Avertissement : Ceci est un outil pédagogique basé sur le Machine Learning. Ne remplace pas un avis médical.")
