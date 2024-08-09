import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import time

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

def configure_model(document_text):
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
        try:
            # Load the file without specifying header to inspect column names
            temp_df = pd.read_excel(uploaded_file, header=None)

            # Identify the correct header row (adjust the index based on actual header location)
            header_row_index = 12  # Adjust this index based on your file structure
            action_plan_df = pd.read_excel(uploaded_file, header=header_row_index)

            # Attempt to rename columns to match expected format
            action_plan_df.columns = [col.strip() for col in action_plan_df.columns]
            action_plan_df = action_plan_df.rename(columns={
                "Numéro d'exigence": "Numéro d'exigence",
                "Exigence IFS Food 8": "Exigence IFS Food 8",
                "Notation": "Notation",
                "Explication (par l’auditeur/l’évaluateur)": "Explication (par l’auditeur/l’évaluateur)"
            })

            # Selecting expected columns
            expected_columns = ["Numéro d'exigence", "Exigence IFS Food 8", "Notation", "Explication (par l’auditeur/l’évaluateur)"]
            action_plan_df = action_plan_df[expected_columns]

            return action_plan_df
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    return None

def prepare_prompt(action_plan_df):
    prompt = "Je suis un expert en IFS Food 8, avec une connaissance approfondie des exigences et des industries alimentaires. J'ai un plan d'action IFS Food 8 contenant des non-conformités détectées. Pour chaque non-conformité, veuillez fournir les informations suivantes :\n\n"
    prompt += "1. La correction proposée pour résoudre la non-conformité.\n"
    prompt += "2. Les preuves potentielles pour soutenir la correction proposée.\n"
    prompt += "3. Les actions correctives permettant d'éliminer les causes sous-jacentes de la non-conformité.\n\n"
    prompt += "Voici les non-conformités détectées :\n\n"
    
    for _, row in action_plan_df.iterrows():
        requirement_text = row["Exigence IFS Food 8"]
        non_conformity_text = row["Explication (par l’auditeur/l’évaluateur)"]
        prompt += f"**Exigence IFS Food 8**: {requirement_text}\n"
        prompt += f"**Description de la non-conformité**: {non_conformity_text}\n\n"
    
    prompt += "Répondez en utilisant le format suivant pour chaque non-conformité :\n\n"
    prompt += "**Correction proposée**: {correction_proposée}\n\n"
    prompt += "**Preuves potentielles**: {preuves_potentielles}\n\n"
    prompt += "**Actions correctives**: {actions_correctives}\n\n"
    prompt += "Référez-vous au Guide IFS Food 8 pour des preuves et des recommandations appropriées."
    return prompt

def get_ai_recommendations(prompt, model, action_plan_df):
    recommendations = []
    try:
        convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
        response = convo.send_message(prompt)
        recommendations_text = response.text
        recommendations = parse_recommendations(recommendations_text)
        
        # S'assurer que nous avons une recommandation pour chaque non-conformité
        while len(recommendations) < len(action_plan_df):
            missing_recs = len(action_plan_df) - len(recommendations)
            additional_prompt = f"Veuillez fournir des recommandations pour les {missing_recs} non-conformités restantes."
            additional_response = convo.send_message(additional_prompt)
            additional_recs = parse_recommendations(additional_response.text)
            recommendations.extend(additional_recs)
        
        # Tronquer si nous avons trop de recommandations
        recommendations = recommendations[:len(action_plan_df)]
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite: {str(e)}")
    
    return recommendations

def parse_recommendations(text):
    recommendations = []
    sections = text.split("Non-conformité")
    for section in sections[1:]:  # Ignorer la première section qui est généralement vide
        rec = {
            "Correction proposée": "",
            "Preuves potentielles": "",
            "Actions correctives": ""
        }
        lines = section.split("\n")
        current_key = None
        for line in lines:
            line = line.strip()
            if line.startswith("Correction proposée:"):
                current_key = "Correction proposée"
            elif line.startswith("Preuves potentielles:"):
                current_key = "Preuves potentielles"
            elif line.startswith("Actions correctives:"):
                current_key = "Actions correctives"
            elif current_key and line:
                rec[current_key] += line + " "
        recommendations.append(rec)
    return recommendations

def dataframe_to_html(df):
    return df.to_html(classes='dataframe table-container', escape=False, index=False)

def display_recommendations(recommendations, action_plan_df):
    for index, (i, row) in enumerate(action_plan_df.iterrows()):
        st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-title">Non-conformité {index + 1} : {row["Exigence IFS Food 8"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="recommendation-section"><b>Description de la non-conformité :</b> {row["Explication (par l’auditeur/l’évaluateur)"]}</div>', unsafe_allow_html=True)
        
        if index < len(recommendations):
            rec = recommendations[index]
            for key, value in rec.items():
                if value:
                    st.markdown(f'<div class="recommendation-section"><b>{key} :</b> {value}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="recommendation-section"><b>{key} :</b> Recommandation manquante</div>', unsafe_allow_html=True)
        else:
            st.markdown("<div class='recommendation-section'><b>Recommandation manquante pour cette non-conformité.</b></div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
            st.markdown('<div class="dataframe-container">' + dataframe_to_html(action_plan_df) + '</div>', unsafe_allow_html=True)
            
            url = "https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv"
            document_text = load_document_from_github(url)
            if document_text:
                model = configure_model(document_text)
                prompt = prepare_prompt(action_plan_df)
                recommendations = get_ai_recommendations(prompt, model, action_plan_df)
                st.subheader("Recommandations de l'IA")
                display_recommendations(recommendations, action_plan_df)

if __name__ == "__main__":
    main()




