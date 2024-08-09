import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Configuration de la page
st.set_page_config(layout="wide")

# Styles CSS
def add_css_styles():
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 24px;
            font-weight: bold;
            color: #004080;
            text-align: center;
            margin-bottom: 25px;
        }
        .banner {
            text-align: center;
            margin-bottom: 40px;
        }
        .dataframe-container {
            margin-bottom: 40px;
        }
        .recommendation-container {
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid #004080;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .recommendation-header {
            font-weight: bold;
            font-size: 18px;
            color: #004080;
            margin-bottom: 10px;
        }
        .recommendation-content {
            margin-bottom: 10px;
            font-size: 16px;
            line-height: 1.5;
        }
        .progress-bar-container {
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Configuration du modèle d'IA
def configure_model():
    genai.configure(api_key=st.secrets["api_key"])
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 1024,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return model

# Chargement du plan d'action
def load_action_plan(uploaded_file):
    try:
        action_plan_df = pd.read_excel(uploaded_file, header=12)
        action_plan_df.columns = [col.strip() for col in action_plan_df.columns]
        expected_columns = ["Numéro d'exigence", "Exigence IFS Food 8", "Notation", "Explication (par l’auditeur/l’évaluateur)"]
        action_plan_df = action_plan_df[expected_columns]
        return action_plan_df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return None

# Préparation du prompt pour la non-conformité
def prepare_prompt_for_non_conformity(non_conformity):
    prompt = f"""
Je suis un expert en IFS Food 8. Voici une non-conformité détectée lors d'un audit :

- **Exigence** : {non_conformity['Numéro d\'exigence']}
- **Non-conformité** : {non_conformity['Exigence IFS Food 8']}
- **Explication** : {non_conformity['Explication (par l’auditeur/l’évaluateur)']}

Fournissez une recommandation structurée comprenant :
- **Correction immédiate** : Action immédiate à entreprendre pour corriger la non-conformité.
- **Preuves requises** : Documents ou preuves à fournir pour montrer que la correction a été effectuée.
- **Actions Correctives** : Mesures à long terme pour prévenir la réapparition de la non-conformité.

Référez-vous au Guide IFS Food 8 pour des preuves et des recommandations appropriées.
"""
    return prompt

# Obtenir la recommandation de l'IA
def get_ai_recommendation_for_non_conformity(prompt, model):
    try:
        convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
        response = convo.send_message(prompt)
        return response.text
    except ResourceExhausted as e:
        st.error(f"Les ressources de l'API sont épuisées : {str(e)}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite: {str(e)}")
        return None

# Analyser la recommandation
def parse_recommendation(text):
    rec = {
        "Correction proposée": "",
        "Preuves potentielles": "",
        "Actions correctives": ""
    }
    if "Correction immédiate" in text:
        rec["Correction proposée"] = text.split("Correction immédiate")[1].split("Preuves requises")[0].strip()
    if "Preuves requises" in text:
        rec["Preuves potentielles"] = text.split("Preuves requises")[1].split("Actions Correctives")[0].strip()
    if "Actions Correctives" in text:
        rec["Actions correctives"] = text.split("Actions Correctives")[1].strip()
    return rec

# Afficher la recommandation
def display_recommendation(recommendation, index, requirement_number):
    st.markdown(f'<div class="recommendation-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="recommendation-header">Non-conformité {index + 1} : Exigence {requirement_number}</div>', unsafe_allow_html=True)

    for key, value in recommendation.items():
        if value:
            st.markdown(f'<div class="recommendation-content"><b>{key} :</b> {value}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Recommandation manquante pour {key}")

    st.markdown('</div>', unsafe_allow_html=True)

# Affichage du plan d'action
def display_action_plan(action_plan_df):
    st.markdown('<div class="dataframe-container">' + action_plan_df.to_html(classes='dataframe', index=False) + '</div>', unsafe_allow_html=True)

# Fonction principale
def main():
    add_css_styles()

    st.markdown('<div class="banner"><img src="https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/visipilot%20banner.PNG" alt="Banner" width="80%"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Assistant VisiPilot pour Plan d\'Actions IFS</div>', unsafe_allow_html=True)
    st.write("Téléchargez votre plan d'action et obtenez des recommandations pour les corrections et les actions correctives.")

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:

            # Initialisation de l'état
            if 'current_index' not in st.session_state:
                st.session_state.current_index = 0
            if 'recommendations' not in st.session_state:
                st.session_state.recommendations = []

            # Affichage du plan d'action
            display_action_plan(action_plan_df)

            # Configuration du modèle d'IA
            model = configure_model()

            # Barre de progression
            progress_bar = st.empty()
            progress_bar.progress(0)

            # Boucle principale pour chaque non-conformité
            while st.session_state.current_index < len(action_plan_df):
                current_non_conformity = action_plan_df.iloc[st.session_state.current_index]
                requirement_number = current_non_conformity["Numéro d'exigence"]

                st.subheader(f"Non-conformité {st.session_state.current_index + 1} : Exigence {requirement_number}")

                # Affichage de la non-conformité actuelle
                st.write(f"**Explication :** {current_non_conformity['Explication (par l’auditeur/l’évaluateur)']}")

                # Obtenir la recommandation
                if st.button("Obtenir Recommandation"):
                    prompt = prepare_prompt_for_non_conformity(current_non_conformity)
                    raw_recommendation = get_ai_recommendation_for_non_conformity(prompt, model)
                    if raw_recommendation:
                        recommendation = parse_recommendation(raw_recommendation)
                        st.session_state.current_recommendation = recommendation
                        display_recommendation(recommendation, st.session_state.current_index, requirement_number)

                # Afficher ou mettre à jour la recommandation actuelle
                if 'current_recommendation' in st.session_state:
                    display_recommendation(st.session_state.current_recommendation, st.session_state.current_index, requirement_number)

                    # Bouton pour un nouvel essai
                    if st.button("Nouvel essai"):
                        prompt = prepare_prompt_for_non_conformity(current_non_conformity)
                        raw_recommendation = get_ai_recommendation_for_non_conformity(prompt, model)
                        if raw_recommendation:
                            recommendation = parse_recommendation(raw_recommendation)
                            st.session_state.current_recommendation = recommendation
                            display_recommendation(recommendation, st.session_state.current_index, requirement_number)

                    # Bouton pour passer à la non-conformité suivante
                    if st.button("Continuer"):
                        st.session_state.recommendations.append({
                            "Numéro d'exigence": current_non_conformity["Numéro d'exigence"],
                            "Correction proposée": st.session_state.current_recommendation["Correction proposée"],
                            "Preuves potentielles": st.session_state.current_recommendation["Preuves potentielles"],
                            "Actions correctives": st.session_state.current_recommendation["Actions correctives"]
                        })
                        st.session_state.current_index += 1
                        del st.session_state['current_recommendation']

                        # Mettre à jour la barre de progression
                        progress_bar.progress(st.session_state.current_index / len(action_plan_df))

            # Téléchargement des recommandations
            if st.session_state.current_index >= len(action_plan_df):
                st.write("### Toutes les non-conformités ont été traitées.")
                st.write("Vous pouvez maintenant télécharger toutes les recommandations.")
                df_recommendations = pd.DataFrame(st.session_state.recommendations)
                st.write(df_recommendations.to_html(classes='dataframe', index=False), unsafe_allow_html=True)
                st.download_button(
                    label="Télécharger les recommandations",
                    data=df_recommendations.to_csv(index=False),
                    file_name="recommandations_ifs_food.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()





