import streamlit as st
import pandas as pd
import requests
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

st.set_page_config(layout="wide")

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

def prepare_prompt_for_non_conformity(non_conformity):
    prompt = f"""
Je suis un expert en IFS Food 8. Voici une non-conformité détectée lors d'un audit :

- **Exigence** : {non_conformity['Numéro d'exigence']}
- **Non-conformité** : {non_conformity['Exigence IFS Food 8']}

Fournissez une recommandation structurée comprenant :
- **Correction immédiate**
- **Preuves requises**
- **Actions Correctives**
"""
    return prompt

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

def main():
    st.title("Assistant VisiPilot pour Plan d'Actions IFS")
    st.write("Téléchargez votre plan d'action et obtenez des recommandations pour les corrections et les actions correctives.")
    
    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            st.dataframe(action_plan_df)
            model = configure_model()

            if 'current_index' not in st.session_state:
                st.session_state.current_index = 0
            if 'recommendations' not in st.session_state:
                st.session_state.recommendations = []

            current_non_conformity = action_plan_df.iloc[st.session_state.current_index]
            st.subheader(f"Non-conformité {st.session_state.current_index + 1} : Exigence {current_non_conformity['Numéro d'exigence']}")

            if st.button("Obtenir Recommandation"):
                prompt = prepare_prompt_for_non_conformity(current_non_conformity)
                raw_recommendation = get_ai_recommendation_for_non_conformity(prompt, model)
                if raw_recommendation:
                    recommendation = parse_recommendation(raw_recommendation)
                    st.session_state.current_recommendation = recommendation
                    st.write(recommendation)

            if 'current_recommendation' in st.session_state:
                st.write("### Recommandation:")
                st.write(st.session_state.current_recommendation)
                
                if st.button("Nouvel essai"):
                    prompt = prepare_prompt_for_non_conformity(current_non_conformity)
                    raw_recommendation = get_ai_recommendation_for_non_conformity(prompt, model)
                    if raw_recommendation:
                        recommendation = parse_recommendation(raw_recommendation)
                        st.session_state.current_recommendation = recommendation
                        st.write(recommendation)
                
                if st.button("Continuer"):
                    st.session_state.recommendations.append({
                        "Numéro d'exigence": current_non_conformity["Numéro d'exigence"],
                        "Correction proposée": st.session_state.current_recommendation["Correction proposée"],
                        "Preuves potentielles": st.session_state.current_recommendation["Preuves potentielles"],
                        "Actions correctives": st.session_state.current_recommendation["Actions correctives"]
                    })
                    st.session_state.current_index += 1
                    del st.session_state['current_recommendation']
                    
                    if st.session_state.current_index >= len(action_plan_df):
                        st.write("### Toutes les non-conformités ont été traitées.")
                        st.write("Vous pouvez maintenant télécharger toutes les recommandations.")
                        df_recommendations = pd.DataFrame(st.session_state.recommendations)
                        st.write(df_recommendations)
                        st.download_button(
                            label="Télécharger les recommandations",
                            data=df_recommendations.to_csv(index=False),
                            file_name="recommandations_ifs_food.csv",
                            mime="text/csv",
                        )
                    else:
                        st.experimental_rerun()

if __name__ == "__main__":
    main()






