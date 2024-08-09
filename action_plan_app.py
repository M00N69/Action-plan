import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Utiliser le mode wide pour l'application
st.set_page_config(layout="wide")

def add_css_styles():
    st.markdown(
        """
        <style>
        .recommendation-container {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .recommendation-title {
            font-weight: bold;
            font-size: 16px;
        }
        .recommendation-section {
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def configure_model():
    genai.configure(api_key=st.secrets["api_key"])
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config={
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "max_output_tokens": 512,
        }
    )

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
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, header=None)
            header_row_index = 12
            df.columns = df.iloc[header_row_index]
            df = df[header_row_index + 1:]
            expected_columns = ["Numéro d'exigence", "Exigence IFS Food 8", "Notation", "Explication (par l’auditeur/l’évaluateur)"]
            df = df[expected_columns]
            return df
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    return None

def prepare_prompt(action_plan_df):
    prompt = "Je suis un expert en IFS Food 8. Veuillez fournir une correction, des preuves potentielles et des actions correctives pour chaque non-conformité listée ci-dessous. Répondez en trois phrases distinctes :\n\n"
    
    for _, row in action_plan_df.iterrows():
        prompt += f"Exigence {row['Numéro d\'exigence']} :\n"  # Correction ici avec le bon usage des guillemets
    
    prompt += "\nRépondez uniquement avec les corrections, les preuves et les actions correctives, sans inclure les descriptions de non-conformité."
    return prompt

def get_ai_recommendations(prompt, model, num_items):
    recommendations = []
    try:
        convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
        response = convo.send_message(prompt)
        recommendations_text = response.text
        recommendations = parse_recommendations(recommendations_text, num_items)
    except ResourceExhausted:
        st.error("Ressources épuisées pour l'API GenAI. Veuillez réessayer plus tard.")
    except Exception as e:
        st.error(f"Erreur inattendue : {str(e)}")
    return recommendations

def parse_recommendations(text, num_items):
    recommendations = []
    sections = text.strip().split("\n\n")
    if len(sections) < num_items * 3:
        st.error("Le modèle n'a pas fourni suffisamment de recommandations.")
        return []

    for i in range(0, num_items * 3, 3):
        correction = sections[i].replace("Correction:", "").strip()
        proofs = sections[i+1].replace("Preuves:", "").strip()
        actions = sections[i+2].replace("Actions correctives:", "").strip()
        recommendations.append({
            "Correction proposée": correction,
            "Preuves potentielles": proofs,
            "Actions correctives": actions
        })
    return recommendations

def display_recommendations(recommendations, action_plan_df):
    for index, (i, row) in enumerate(action_plan_df.iterrows()):
        st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-title">Non-conformité {index + 1} : Exigence {row["Numéro d\'exigence"]}</div>', unsafe_allow_html=True)
        
        if index < len(recommendations):
            rec = recommendations[index]
            for key, value in rec.items():
                st.markdown(f'<div class="recommendation-section"><b>{key} :</b> {value}</div>', unsafe_allow_html=True)
        else:
            st.warning("Recommandation manquante pour cette non-conformité.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    add_css_styles()
    st.title("Assistant VisiPilot pour Plan d'Actions IFS")

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            st.markdown('<div class="dataframe-container">' + dataframe_to_html(action_plan_df) + '</div>', unsafe_allow_html=True)
            
            document_text = load_document_from_github("https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv")
            if document_text:
                model = configure_model()
                prompt = prepare_prompt(action_plan_df)
                recommendations = get_ai_recommendations(prompt, model, len(action_plan_df))
                display_recommendations(recommendations, action_plan_df)

if you open and start the code."
    main()





