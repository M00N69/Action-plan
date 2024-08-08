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

def get_ai_recommendations(prompt, model):
    recommendations = []
    try:
        convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
        response = convo.send_message(prompt)
        recommendations_text = response.text
        st.write(f"Recommendations Text: {recommendations_text}")  # Debugging line
        recommendations = parse_recommendations(recommendations_text)
    except ResourceExhausted:
        st.error("Ressources épuisées pour l'API GenAI. Veuillez réessayer plus tard.")
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite: {str(e)}")
        recommendations.append({
            "Correction proposée": "Erreur lors de la génération de la recommandation",
            "Preuves potentielles": "",
            "Actions correctives": ""
        })
    return recommendations

def parse_recommendations(text):
    recommendations = []
    sections = text.split("Non-conformité")
    for section in sections:
        if section.strip():
            correction = ""
            preuves = ""
            actions = ""
            correction_found = False
            preuves_found = False
            actions_found = False
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("Correction proposée:"):
                    correction = line.replace("Correction proposée:", "").strip()
                    correction_found = True
                elif line.startswith("Preuves potentielles:"):
                    preuves = line.replace("Preuves potentielles:", "").strip()
                    preuves_found = True
                elif line.startswith("Actions correctives:"):
                    actions = line.replace("Actions correctives:", "").strip()
                    actions_found = True
            
            if correction_found and preuves_found and actions_found:
                recommendations.append({
                    "Correction proposée": correction,
                    "Preuves potentielles": preuves,
                    "Actions correctives": actions
                })
            else:
                st.warning(f"Recommandation incomplète détectée : {section}")
    
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
            st.markdown(f'<div class="recommendation-section"><b>Correction proposée :</b> {rec["Correction proposée"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="recommendation-section"><b>Preuves potentielles :</b> {rec["Preuves potentielles"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="recommendation-section"><b>Actions correctives :</b> {rec["Actions correctives"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("Recommandation manquante pour cette non-conformité.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def generate_downloadable_text(recommendations, action_plan_df):
    output = "Plan d'action IFS Food 8 : Corrections et Actions Correctives\n\n"
    for index, (i, row) in enumerate(action_plan_df.iterrows()):
        output += f"Non-conformité {index + 1} : {row['Exigence IFS Food 8']}\n"
        output += f"Description de la non-conformité : {row['Explication (par l’auditeur/l’évaluateur)']}\n"
        if index < len(recommendations):
            rec = recommendations[index]
            output += f"Correction proposée : {rec['Correction proposée']}\n"
            output += f"Preuves potentielles : {rec['Preuves potentielles']}\n"
            output += f"Actions correctives : {rec['Actions correctives']}\n"
        else:
            output += "Recommandation manquante pour cette non-conformité.\n"
        output += "\n"
    return output

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
                recommendations = get_ai_recommendations(prompt, model)
                st.subheader("Recommandations de l'IA")
                display_recommendations(recommendations, action_plan_df)
                
                if st.button("Télécharger les Recommandations"):
                    text_output = generate_downloadable_text(recommendations, action_plan_df)
                    st.download_button(
                        label="Télécharger le fichier",
                        data=text_output,
                        file_name="recommandations_ifs.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()





