import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai

# Load VISIPACT data and IFS checklist from GitHub
visipact_df = pd.read_csv("https://raw.githubusercontent.com/M00N69/Action-plan/main/output%20vsipact.csv")
ifs_checklist_df = pd.read_csv("https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv")

# Function to load the user-uploaded action plan
def load_action_plan(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            # Skip the first 11 rows and use the 12th row as the header
            action_plan_df = pd.read_excel(uploaded_file, header=11) 
        else:
            st.error("Type de fichier incorrect. Veuillez télécharger un fichier Excel.")
            action_plan_df = None
        return action_plan_df

# Function to configure GenAI model
def configure_model(api_key, document_text):
    genai.configure(api_key=api_key)
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
        model_name="gemini-pro",  # Choose the Gemini model you want to use
        generation_config=generation_config,
        system_instruction=document_text,
        safety_settings=safety_settings
    )
    return model

@st.cache(allow_output_mutation=True, ttl=86400)
def load_document_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # S'assurer que les mauvaises réponses sont gérées
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Échec de téléchargement du document: {str(e)}")
        return None

# Function to generate AI recommendations using GenAI
def get_ai_recommendations(non_conformity_text, requirement_number, visipact_df, ifs_checklist_df, model):
    # 1. Create a prompt for GenAI
    prompt = f"""
    Je suis un expert en IFS Food 8, avec une connaissance approfondie des exigences et des industries alimentaires.
    Une non-conformité a été trouvée pour l'exigence suivante:
    {ifs_checklist_df.loc[ifs_checklist_df["NUM_REQ"] == requirement_number, "IFS Requirements"].values[0]}
    
    La description de la non-conformité est: {non_conformity_text}

    Veuillez fournir:
    1. Une correction proposée basée sur les données historiques de VISIPACT.
    2. Un plan d'action pour corriger la non-conformité, avec une échéance suggérée.
    3. Des preuves à l'appui de l'action proposée, en citant les sections du Guide IFS Food 8. 

    Voici quelques données historiques de VISIPACT:
    {visipact_df[["NomUnite", "CONSTATSDAUDIT", "ACTIONFOURNISSEUR"]].to_string()}
    
    N'oubliez pas de vous référer au Guide IFS Food 8 pour des preuves et des recommandations.
    """

    # 2. Start a chat with GenAI
    convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])

    # 3. Get the GenAI response
    response = convo.send_message(prompt) 

    # 4. Extract recommendations from GenAI's response
    # (You may need to adjust this based on how GenAI formats its response)
    corrective_actions = response.text
    evidence = " "  
    suggested_deadlines = " "  

    recommendations = {
        "corrective_actions": corrective_actions,
        "evidence": evidence,
        "suggested_deadlines": suggested_deadlines,
        "requirement_number": requirement_number,
        "non_conformity_text": non_conformity_text
    }

    return recommendations


# Function to generate a Streamlit table with recommendations
def generate_table(recommendations):
    table_data = {
        "Numéro d'Exigence": [recommendations["requirement_number"]],
        "Non-Conformité": [recommendations["non_conformity_text"]],
        "Action Corrective": [recommendations["corrective_actions"]],
        "Preuve": [recommendations["evidence"]],
        "Echéance Suggérée": [recommendations["suggested_deadlines"]]
    }
    st.dataframe(pd.DataFrame(table_data))

    # Allow user to download the table
    csv = pd.DataFrame(table_data).to_csv(index=False)
    st.download_button(
        label="Télécharger les Recommandations",
        data=csv,
        file_name="recommendations.csv",
        mime="text/csv",
    )

def main():
    # Streamlit App
    st.title("Assistant VisiPilot pour Plan d'Actions IFS")

    st.write("Cet outil vous aide à gérer votre plan d'action IFS Food 8 avec l'aide de l'IA.")
    st.write("Téléchargez votre plan d'action et obtenez des recommandations pour les corrections et les actions correctives.")

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    if uploaded_file is not None:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            st.dataframe(action_plan_df)
            
            # Create a DataFrame for recommendations
            recommendations_df = pd.DataFrame(columns=["requirementNo", "requirementText", "requirementScore", "requirementExplanation", "correctionDescription", "correctionResponsibility", "correctionDueDate", "correctionStatus", "correctionEvidence", "correctiveActionDescription", "correctiveActionResponsibility", "correctiveActionDueDate", "correctiveActionStatus", "releaseResponsibility", "releaseDate"])

            # Get recommendations for each non-conformity
            for index, row in action_plan_df.iterrows():
                non_conformity_text = row["Explication (par l’auditeur/l’évaluateur)"]  # Correct column name
                requirement_number = row["Numéro d'exigence"]  # Correct column name
                
                # Load the document from GitHub
                url = "https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/BRC9_GUIde%20_interpretation.txt"
                document_text = load_document_from_github(url)

                if document_text:
                    api_key = st.secrets["api_key"]
                    model = configure_model(api_key, document_text)

                    recommendations = get_ai_recommendations(non_conformity_text, requirement_number, visipact_df, ifs_checklist_df, model)
                    recommendations_df = pd.concat([recommendations_df, pd.DataFrame([recommendations])], ignore_index=True)
                    
            # Display the recommendations
            st.subheader("Recommandations de l'IA")
            generate_table(recommendations_df)
                
    non_conformity_text = st.text_input("Entrez la description de la non-conformité")
    requirement_number = st.selectbox("Sélectionnez le numéro d'exigence", list(ifs_checklist_df["NUM_REQ"].values))

    if st.button("Obtenir des Recommandations de l'IA"):
        recommendations = get_ai_recommendations(non_conformity_text, requirement_number, visipact_df, ifs_checklist_df, model)
        st.subheader("Recommandations de l'IA")
        generate_table(recommendations)

if __name__ == "__main__":
    main()
