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
        .table-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            max-width: 100%;
            border: 1px solid #ddd;
            background-color: #29292F; /* Fond sombre */
        }

        th, td {
            border: 1px solid #ddd;
            text-align: left;
            padding: 8px;
            color: #fff; /* Texte blanc */
            white-space: pre-wrap; /* Retour à la ligne automatique */
        }

        tr:nth-child(even) {
            background-color: #333; /* Ligne paire plus foncée */
        }

        th {
            background-color: #333; /* En-têtes plus foncés */
            font-weight: bold;
        }

        .dataframe-container {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
        try:
            # Load the file without specifying header to inspect column names
            temp_df = pd.read_excel(uploaded_file, header=None)
            
            # Identify the correct header row (adjust the index based on actual header location)
            header_row_index = 12  # Adjust this index based on your file structure
            action_plan_df = pd.read_excel(uploaded_file, header=header_row_index)
            
            # Correct column names
            expected_columns = ["Numéro d'exigence", "Exigence IFS Food 8", "Notation", "Explication (par l’auditeur/l’évaluateur)"]
            if all(col in action_plan_df.columns for col in expected_columns):
                action_plan_df = action_plan_df[expected_columns]
            else:
                st.error(f"Les colonnes attendues ne sont pas présentes dans le fichier. Colonnes trouvées: {action_plan_df.columns.tolist()}")
                return None

            return action_plan_df
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    return None

def prepare_prompt(action_plan_df, guide_text):
    prompt = "Je suis un expert en IFS Food 8, avec une connaissance approfondie des exigences et des industries alimentaires. J'ai un plan d'action IFS Food 8.\n"
    for _, row in action_plan_df.iterrows():
        requirement_text = row["Exigence IFS Food 8"]
        non_conformity_text = row["Explication (par l’auditeur/l’évaluateur)"]
        prompt += f"""
        Une non-conformité a été trouvée pour l'exigence suivante:
        {requirement_text}
        
        La description de la non-conformité est: {non_conformity_text}

        Veuillez fournir:
        1. Une correction proposée.
        2. Un plan d'action pour corriger la non-conformité, avec une échéance suggérée.
        3. Des preuves à l'appui de l'action proposée, en citant les sections du Guide IFS Food 8.
        """
    prompt += f"\nN'oubliez pas de vous référer au Guide IFS Food 8 pour des preuves et des recommandations. Voici le texte du guide:\n{guide_text}"
    return prompt

def get_ai_recommendations(prompt, model):
    recommendations = []
    try:
        convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
        response = convo.send_message(prompt)
        recommendations_text = response.text
        recommendations = parse_recommendations(recommendations_text)
    except ResourceExhausted:
        st.error("Ressources épuisées pour l'API GenAI. Veuillez réessayer plus tard.")
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite: {str(e)}")
        recommendations.append({
            "Exigence IFS Food 8": "",
            "Non-conformité #": "Erreur lors de la génération de la recommandation",
            "Correction proposée": "",
            "Preuve possible": "",
            "Action corrective proposée": ""
        })
    return recommendations

def parse_recommendations(text):
    recommendations = []
    parts = text.split("\n\n")
    for i in range(0, len(parts), 5):
        exigence = parts[i] if i < len(parts) else "Pas d'exigence disponible"
        non_conformite = parts[i+1] if i+1 < len(parts) else "Pas de non-conformité disponible"
        correction = parts[i+2] if i+2 < len(parts) else "Pas de correction disponible"
        preuve = parts[i+3] if i+3 < len(parts) else "Pas de preuve disponible"
        action_corrective = parts[i+4] if i+4 < len(parts) else "Pas d'action corrective disponible"
        recommendations.append({
            "Exigence IFS Food 8": exigence,
            "Non-conformité #": non_conformite,
            "Correction proposée": correction,
            "Preuve possible": preuve,
            "Action corrective proposée": action_corrective
        })
    return recommendations

def dataframe_to_html(df):
    return df.to_html(classes='dataframe table-container', escape=False, index=False)

def main():
    add_css_styles()
    
    st.image('https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/visipilot%20banner.PNG', use_column_width=True)
    st.title("Assistant VisiPilot pour Plan d'Actions IFS")
    
    # Use session state to manage the flow between pages
    if "page" not in st.session_state:
        st.session_state.page = 1

    if st.session_state.page == 1:
        st.write("Cet outil vous aide à gérer votre plan d'action IFS Food 8 avec l'aide de l'IA.")
        uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
        if uploaded_file:
            action_plan_df = load_action_plan(uploaded_file)
            if action_plan_df is not None:
                st.markdown('<div class="dataframe-container">' + dataframe_to_html(action_plan_df) + '</div>', unsafe_allow_html=True)
                if st.button("Obtenir des recommandations de l'IA"):
                    st.session_state.action_plan_df = action_plan_df
                    st.session_state.page = 2
    elif st.session_state.page == 2:
        action_plan_df = st.session_state.action_plan_df
        
        # Load the guide document for reference
        guide_url = "https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv"
        guide_text = load_document_from_github(guide_url)
        if guide_text:
            model = configure_model(guide_text)
            prompt = prepare_prompt(action_plan_df, guide_text)
            recommendations = get_ai_recommendations(prompt, model)
            st.subheader("Recommandations de l'IA")
            
            recommendations_df = pd.DataFrame(recommendations)
            st.markdown('<div class="dataframe-container">' + dataframe_to_html(recommendations_df) + '</div>', unsafe_allow_html=True)
            
            csv = recommendations_df.to_csv(index=False)
            st.download_button(
                label="Télécharger les Recommandations",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv",
            )
        if st.button("Retour"):
            st.session_state.page = 1

if __name__ == "__main__":
    main()




