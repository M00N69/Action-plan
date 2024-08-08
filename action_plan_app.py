import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import time

# Fonction pour ajouter des styles CSS
def add_css_styles():
    st.markdown(
        """
        <style>
        .table-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        table {
            border-collapse: collapse;
            width: 80%;
            max-width: 1200px;
            border: 1px solid #ddd;
            background-color: #29292F; /* Fond sombre */
        }

        th, td {
            border: 1px solid #ddd;
            text-align: left;
            padding: 8px;
            color: #fff; /* Texte blanc */
        }

        tr:nth-child(even) {
            background-color: #333; /* Ligne paire plus foncée */
        }

        th {
            background-color: #333; /* En-têtes plus foncés */
            font-weight: bold;
        }

        a {
            color: #3080F8; /* Bleu clair pour les liens */
            text-decoration: none; /* Supprimer le soulignement par défaut */
        }

        a:hover {
            text-decoration: underline; /* Soulignement au survol */
        }

        .analyze-button {
            padding: 4px 8px;
            color: #fff;
            background-color: #3080F8;
            border: none;
            cursor: pointer;
        }

        .analyze-button:hover {
            background-color: #1A5BB1;
        }

        .dataframe td {
            white-space: pre-wrap; /* Retour à la ligne automatique */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Configure the GenAI model
def configure_model(document_text):
    genai.configure(api_key=st.secrets["api_key"])
    generation_config = {
        "temperature": 2,
        "top_p": 0.4,
        "top_k": 32,
        "max_output_tokens": 8192,
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
        system_instruction=document_text,
        safety_settings=safety_settings
    )
    return model

@st.cache_data(ttl=86400)
def load_document_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Échec de téléchargement du document: {str(e)}")
        return None

def load_action_plan(uploaded_file):
    if uploaded_file is not None:
        action_plan_df = pd.read_excel(uploaded_file, header=12)
        columns_to_keep = ["Numéro d'exigence", "Exigence IFS Food 8", "Notation", "Explication (par l’auditeur/l’évaluateur)"]
        action_plan_df = action_plan_df[columns_to_keep]
        return action_plan_df
    return None

def prepare_prompts(action_plan_df):
    prompts = []
    for _, row in action_plan_df.iterrows():
        requirement_text = row["Exigence IFS Food 8"]
        non_conformity_text = row["Explication (par l’auditeur/l’évaluateur)"]
        prompt = f"""
        Je suis un expert en IFS Food 8, avec une connaissance approfondie des exigences et des industries alimentaires.
        J'ai un plan d'action IFS Food 8.
        Une non-conformité a été trouvée pour l'exigence suivante:
        {requirement_text}
        
        La description de la non-conformité est: {non_conformity_text}

        Veuillez fournir:
        1. Une correction proposée.
        2. Un plan d'action pour corriger la non-conformité, avec une échéance suggérée.
        3. Des preuves à l'appui de l'action proposée, en citant les sections du Guide IFS Food 8. 

        N'oubliez pas de vous référer au Guide IFS Food 8 pour des preuves et des recommandations.
        """
        prompts.append(prompt)
    return prompts

def get_ai_recommendations(prompts, model):
    recommendations = []
    for prompt in prompts:
        try:
            convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
            response = convo.send_message(prompt)
            corrective_actions = response.text
            recommendations.append(parse_recommendation(corrective_actions))
            time.sleep(2)
        except ResourceExhausted:
            st.error("Ressources épuisées pour l'API GenAI. Veuillez réessayer plus tard.")
            break
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite: {str(e)}")
            recommendations.append({
                "Correction proposée": "Erreur lors de la génération de la recommandation",
                "Plan d'action": "",
                "Preuves": ""
            })
    return recommendations

def parse_recommendation(text):
    parts = text.split('\n\n', 2)
    correction = parts[0] if len(parts) > 0 else "Pas de correction disponible"
    plan_action = parts[1] if len(parts) > 1 else "Pas de plan d'action disponible"
    preuves = parts[2] if len(parts) > 2 else "Pas de preuves disponibles"
    return {
        "Correction proposée": correction,
        "Plan d'action": plan_action,
        "Preuves": preuves
    }

def generate_table(recommendations):
    recommendations_df = pd.DataFrame(recommendations)
    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.dataframe(recommendations_df.style.set_properties(**{
        'white-space': 'pre-wrap',
        'text-align': 'left'
    }))
    st.markdown('</div>', unsafe_allow_html=True)
    csv = recommendations_df.to_csv(index=False)
    st.download_button(
        label="Télécharger les Recommandations",
        data=csv,
        file_name="recommendations.csv",
        mime="text/csv",
    )

def main():
    add_css_styles()
    
    st.image('https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/visipilot%20banner.PNG', use_column_width=True)
    st.title("Assistant VisiPilot pour Plan d'Actions IFS")
    st.write("Cet outil vous aide à gérer votre plan d'action IFS Food 8 avec l'aide de l'IA.")
    st.write("Téléchargez votre plan d'action et obtenez des recommandations pour les corrections et les actions correctives.")

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            st.markdown('<div class="table-container">', unsafe_allow_html=True)
            st.dataframe(action_plan_df.style.set_properties(**{
                'white-space': 'pre-wrap',
                'text-align': 'left'
            }))
            st.markdown('</div>', unsafe_allow_html=True)
            
            url = "https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/BRC9_GUIde%20_interpretation.txt"
            document_text = load_document_from_github(url)
            if document_text:
                model = configure_model(document_text)
                prompts = prepare_prompts(action_plan_df)
                recommendations = get_ai_recommendations(prompts, model)
                st.subheader("Recommandations de l'IA")
                generate_table(recommendations)

if __name__ == "__main__":
    main()





