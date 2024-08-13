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

def get_guide_context(exigence_num, guide_df):
    """
    Recherche les informations pertinentes dans le guide IFSv8 pour une exigence spécifique.
    """
    context = guide_df[guide_df['NUM_REQ'].str.contains(exigence_num, na=False)]
    if not context.empty:
        guidelines = context['Good practice'].iloc[0]  # Assurez-vous que la colonne existe
        return guidelines
    else:
        return "Aucune information spécifique trouvée dans le guide pour cette exigence."

def generate_advanced_prompt(non_conformity, guide_df):
    """
    Génère un prompt détaillé en utilisant le guide IFSv8 comme contexte.
    """
    exigence_num = non_conformity.get("Numéro d'exigence", "Exigence non spécifiée")
    non_conformity_desc = non_conformity.get("Exigence IFS Food 8", "Non-conformité non spécifiée")
    
    # Récupérer le contexte du guide IFSv8
    guide_context = get_guide_context(exigence_num, guide_df)

    # Créer un prompt détaillé
    prompt = f"""
    Je suis un expert en IFS Food 8. Voici une non-conformité détectée lors d'un audit :

    - **Exigence** : {exigence_num}
    - **Non-conformité** : {non_conformity_desc}

    Contexte supplémentaire tiré du guide IFSv8 :
    {guide_context}

    Fournissez une recommandation structurée et détaillée comprenant :
    - **Correction immédiate** (action spécifique et claire)
    - **Preuves requises** (documents précis nécessaires)
    - **Actions Correctives** (mesures à long terme)
    """
    
    return prompt

@handle_ai_errors
def generate_ai_recommendation(non_conformity, model, guide_df):
    prompt = generate_advanced_prompt(non_conformity, guide_df)
    
    convo = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
    response = convo.send_message(prompt)
    
    if response is None or response.text.strip() == "":
        st.warning("L'IA n'a pas pu générer de réponse. Utilisation d'une valeur par défaut.")
        return {
            "Correction proposée": "Aucune recommandation générée.",
            "Preuves potentielles": "Aucune preuve recommandée.",
            "Actions correctives": "Aucune action corrective suggérée."
        }
    
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

def display_recommendation(recommendation, index, requirement_number):
    if recommendation is None:
        st.error("La recommandation n'a pas pu être générée ou est vide. Veuillez réessayer.")
        return

    if not isinstance(recommendation, dict):
        st.error("Erreur interne : La recommandation n'est pas dans le bon format.")
        return

    st.markdown(f'<div class="recommendation-container">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="recommendation-header">Recommandation pour la Non-conformité {index + 1} : Exigence {requirement_number}</h2>', unsafe_allow_html=True)

    for key, value in recommendation.items():
        if value:
            st.markdown(f'<div class="recommendation-content"><b>{key} :</b> {value}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="recommendation-content warning">Recommandation manquante pour {key}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def generate_summary_table(action_plan_df, model, guide_df):
    summary_data = []

    for index, non_conformity in action_plan_df.iterrows():
        raw_recommendation = generate_ai_recommendation(non_conformity, model, guide_df)
        if raw_recommendation:
            recommendation = parse_recommendation(raw_recommendation)
            if recommendation is None:
                recommendation = {
                    "Correction proposée": "Aucune recommandation générée.",
                    "Preuves potentielles": "Aucune preuve recommandée.",
                    "Actions correctives": "Aucune action corrective suggérée."
                }
            summary_data.append({
                "Numéro de non-conformité": index + 1,
                "Numéro d'exigence": non_conformity["Numéro d'exigence"],
                "Correction proposée": recommendation.get("Correction proposée", "Non disponible"),
                "Preuves potentielles": recommendation.get("Preuves potentielles", "Non disponible"),
                "Actions correctives": recommendation.get("Actions correctives", "Non disponible"),
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

            # Charger le guide IFSv8 depuis le CSV sur GitHub
            guide_df = pd.read_csv("https://raw.githubusercontent.com/M00N69/Action-plan/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv")

            if guide_df is not None:
                # Génération du tableau récapitulatif pour toutes les non-conformités
                summary_df = generate_summary_table(action_plan_df, model, guide_df)
                
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







