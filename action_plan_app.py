import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

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
    """Load a document from GitHub."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Échec de téléchargement du document: {str(e)}")
        return None

def load_action_plan(uploaded_file):
    """Load the user-uploaded action plan."""
    if uploaded_file is not None:
        action_plan_df = pd.read_excel(uploaded_file, header=11)  # header=11 to skip the first 11 rows
        return action_plan_df
    return None

def get_ai_recommendations(action_plan_df, model):
    """Generate AI recommendations using GenAI."""
    recommendations = []

    required_columns = ["Exigence IFS Food 8", "Explication (par l’auditeur/l’évaluateur)", "Numéro d'exigence", "Notation"]

    # Vérifier la présence des colonnes nécessaires
    for col in required_columns:
        if col not in action_plan_df.columns:
            st.error(f"La colonne requise '{col}' est manquante dans le fichier téléchargé.")
            st.write("Colonnes disponibles:", list(action_plan_df.columns))
            return []

    for _, row in action_plan_df.iterrows():
        try:
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
            convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
            response = convo.send_message(prompt)
            corrective_actions = response.text
            recommendations.append({
                "requirementNo": row["Numéro d'exigence"],
                "requirementText": requirement_text,
                "requirementScore": row["Notation"],
                "requirementExplanation": non_conformity_text,
                "correctiveActionDescription": corrective_actions,
            })
        except KeyError as e:
            st.error(f"Colonne manquante dans le DataFrame: {e}")
            st.write("Colonnes disponibles:", list(action_plan_df.columns))
            break
        except ResourceExhausted:
            st.error("Ressources épuisées pour l'API GenAI. Veuillez réessayer plus tard.")
            break
        except genai.InvalidArgument as e:
            st.error(f"Erreur lors de la génération de contenu: {str(e)}")
            break
    return recommendations

def generate_table(recommendations):
    """Generate a Streamlit table with recommendations."""
    recommendations_df = pd.DataFrame(recommendations)
    st.dataframe(recommendations_df)
    csv = recommendations_df.to_csv(index=False)
    st.download_button(
        label="Télécharger les Recommandations",
        data=csv,
        file_name="recommendations.csv",
        mime="text/csv",
    )

def main():
    """Main function for the Streamlit app."""
    st.image('https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/visipilot%20banner.PNG', use_column_width=True)
    st.title("Assistant VisiPilot pour Plan d'Actions IFS")
    st.write("Cet outil vous aide à gérer votre plan d'action IFS Food 8 avec l'aide de l'IA.")
    st.write("Téléchargez votre plan d'action et obtenez des recommandations pour les corrections et les actions correctives.")

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            st.dataframe(action_plan_df)
            url = "https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/BRC9_GUIde%20_interpretation.txt"
            document_text = load_document_from_github(url)
            if document_text:
                model = configure_model(document_text)
                recommendations = get_ai_recommendations(action_plan_df, model)
                st.subheader("Recommandations de l'IA")
                generate_table(recommendations)

if __name__ == "__main__":
    main()

