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

def display_recommendation(recommendation, index, requirement_number):
    if recommendation is None:
        st.error("La recommandation n'a pas pu être générée. Veuillez réessayer.")
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

            if 'current_index' not in st.session_state:
                st.session_state.current_index = 0
                st.session_state.recommendations = []

            st.markdown(f"**Non-conformité {st.session_state.current_index + 1} / {len(action_plan_df)}**")
            progress_bar = st.progress(st.session_state.current_index / len(action_plan_df))

            current_non_conformity = action_plan_df.iloc[st.session_state.current_index]
            requirement_number = current_non_conformity["Numéro d'exigence"]
            st.subheader(f"Non-conformité {st.session_state.current_index + 1} : Exigence {requirement_number}")

            if 'current_recommendation' not in st.session_state:
                if st.button("Obtenir Recommandation", key="get_recommendation"):
                    raw_recommendation = generate_ai_recommendation(current_non_conformity, model)
                    if raw_recommendation:
                        recommendation = parse_recommendation(raw_recommendation)
                        st.session_state.current_recommendation = recommendation
                    else:
                        st.session_state.current_recommendation = None

            if 'current_recommendation' in st.session_state:
                display_recommendation(st.session_state.current_recommendation, st.session_state.current_index, requirement_number)
                
                if st.session_state.current_index > 0:
                    if st.button("Précédent", key="previous_recommendation"):
                        st.session_state.current_index -= 1
                        st.session_state.current_recommendation = None

                if st.button("Nouvel essai", key="retry_recommendation"):
                    del st.session_state['current_recommendation']
                    raw_recommendation = generate_ai_recommendation(current_non_conformity, model)
                    if raw_recommendation:
                        recommendation = parse_recommendation(raw_recommendation)
                        st.session_state.current_recommendation = recommendation
                    else:
                        st.session_state.current_recommendation = None
                    display_recommendation(recommendation, st.session_state.current_index, requirement_number)
                
                if st.button("Accepter et Continuer", key="continue_to_next"):
                    if st.session_state.current_recommendation and all(st.session_state.current_recommendation.values()):
                        st.session_state.recommendations.append({
                            "Numéro de non-conformité": st.session_state.current_index + 1,
                            "Numéro d'exigence": current_non_conformity["Numéro d'exigence"],
                            "Correction proposée": st.session_state.current_recommendation["Correction proposée"],
                            "Preuves potentielles": st.session_state.current_recommendation["Preuves potentielles"],
                            "Actions correctives": st.session_state.current_recommendation["Actions correctives"]
                        })
                        st.session_state.current_index += 1
                        del st.session_state['current_recommendation']
                        progress_bar.progress(st.session_state.current_index / len(action_plan_df))
                        
                        if st.session_state.current_index >= len(action_plan_df):
                            st.markdown('<div class="success">Toutes les non-conformités ont été traitées. Vous pouvez maintenant télécharger toutes les recommandations.</div>', unsafe_allow_html=True)
                            df_recommendations = pd.DataFrame(st.session_state.recommendations)
                            st.write(df_recommendations.to_html(classes='dataframe', index=False), unsafe_allow_html=True)
                            st.download_button(
                                label="Télécharger les recommandations",
                                data=df_recommendations.to_csv(index=False),
                                file_name="recommandations_ifs_food.csv",
                                mime="text/csv",
                            )
                        else:
                            st.session_state.current_recommendation = None
                            st.success("Recommandation acceptée. Passez à la non-conformité suivante.")
                    else:
                        st.warning("Certaines sections de la recommandation sont manquantes ou non générées. Veuillez essayer à nouveau.")

            st.subheader("Résumé des Recommandations")
            if st.session_state.recommendations:
                df_recommendations = pd.DataFrame(st.session_state.recommendations)
                st.write(df_recommendations.to_html(classes='dataframe', index=False), unsafe_allow_html=True)
                st.download_button(
                    label="Télécharger les recommandations",
                    data=df_recommendations.to_csv(index=False),
                    file_name="recommandations_ifs_food.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()





