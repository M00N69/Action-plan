import streamlit as st
import pandas as pd
import google.generativeai as genai
from string import Template

st.set_page_config(layout="wide")

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
            background-color: #f0f8ff;
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
        .warning {
            color: red;
            font-weight: bold;
        }
        .success {
            color: green;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

def handle_ai_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite: {str(e)}")
            return None
    return wrapper

@st.cache_data(ttl=86400)
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

@handle_ai_errors
def generate_ai_recommendation(non_conformity, model):
    prompt_template = Template("""
    Je suis un expert en IFS Food 8. Voici une non-conformité détectée lors d'un audit :
    
    - **Exigence** : $exigence
    - **Non-conformité** : $non_conformity
    
    Fournissez une recommandation structurée et détaillée comprenant :
    - **Correction immédiate** (action spécifique et claire)
    - **Preuves requises** (documents précis nécessaires)
    - **Actions Correctives** (mesures à long terme)
    """)

    prompt = prompt_template.substitute(
        exigence=non_conformity.get("Numéro d'exigence", "Exigence non spécifiée"),
        non_conformity=non_conformity.get("Exigence IFS Food 8", "Non-conformité non spécifiée")
    )

    convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
    response = convo.send_message(prompt)
    return response.text if response else None

def parse_recommendation(text):
    rec = {
        "Correction proposée": "",
        "Preuves potentielles": "",
        "Actions correctives": ""
    }
    try:
        if "Correction immédiate" in text:
            rec["Correction proposée"] = text.split("Correction immédiate")[1].split("Preuves requises")[0].strip()
        if "Preuves requises" in text:
            rec["Preuves potentielles"] = text.split("Preuves requises")[1].split("Actions Correctives")[0].strip()
        if "Actions Correctives" in text:
            rec["Actions correctives"] = text.split("Actions Correctives")[1].split("Remarques")[0].strip()
        else:
            rec["Actions correctives"] = "Aucune action corrective spécifique n'a été proposée dans cette recommandation."
    except IndexError:
        st.error("La recommandation fournie par l'IA est mal structurée ou incomplète. Veuillez essayer à nouveau.")
    return rec

def format_recommendation_text(recommendation):
    formatted_text = ""
    for key, value in recommendation.items():
        formatted_text += f"**{key}** : {value}\n\n"
    return formatted_text

def generate_summary_table(action_plan_df, model):
    summary_data = []

    for index, non_conformity in action_plan_df.iterrows():
        raw_recommendation = generate_ai_recommendation(non_conformity, model)
        if raw_recommendation:
            recommendation = parse_recommendation(raw_recommendation)
            summary_data.append({
                "Numéro de non-conformité": index + 1,
                "Numéro d'exigence": non_conformity["Numéro d'exigence"],
                "Correction proposée": recommendation["Correction proposée"],
                "Preuves potentielles": recommendation["Preuves potentielles"],
                "Actions correctives": recommendation["Actions correctives"],
            })

    return pd.DataFrame(summary_data)

def main():
    add_css_styles()

    st.markdown('<div class="banner"><img src="https://raw.githubusercontent.com/M00N69/Gemini-Knowledge/main/visipilot%20banner.PNG" alt="Banner" width="80%"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Assistant VisiPilot pour Plan d\'Actions IFS</div>', unsafe_allow_html=True)
    st.write("Téléchargez votre plan d'action et obtenez des recommandations pour les corrections et les actions correctives.")

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"], key="file_uploader")
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            st.markdown('<div class="dataframe-container">' + action_plan_df.to_html(classes='dataframe', index=False) + '</div>', unsafe_allow_html=True)
            model = configure_model()

            # Génération du tableau récapitulatif pour toutes les non-conformités
            summary_df = generate_summary_table(action_plan_df, model)
            
            st.subheader("Résumé des Recommandations")
            st.write(summary_df.to_html(classes='dataframe', index=False, escape=False), unsafe_allow_html=True)

            st.download_button(
                label="Télécharger les recommandations",
                data=summary_df.to_csv(index=False),
                file_name="recommandations_ifs_food.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()





