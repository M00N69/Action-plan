import streamlit as st
import pandas as pd
from pocketgroq import GroqProvider

# Configuration de la page
st.set_page_config(layout="wide")

# Ajouter les styles CSS personnalisés
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
    .dataframe-container {
        margin-bottom: 20px;
    }
    .banner {
        background-image: url('https://github.com/M00N69/BUSCAR/blob/main/logo%2002%20copie.jpg?raw=true');
        background-size: cover;
        height: 200px;
        background-position: center;
        margin-bottom: 20px;
    }
    div.stButton > button {
        background-color: #004080;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        background-color: #0066cc;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Ajouter la bannière
st.markdown('<div class="banner"></div>', unsafe_allow_html=True)

# Initialiser PocketGroq
def get_groq_provider():
    if not st.session_state.api_key:
        st.error("Veuillez entrer votre clé API Groq.")
        return None
    return GroqProvider(api_key=st.session_state.api_key)

# Fonction pour charger le fichier Excel avec le plan d'action
def load_action_plan(uploaded_file):
    try:
        action_plan_df = pd.read_excel(uploaded_file, header=11)
        action_plan_df = action_plan_df[["requirementNo", "requirementText", "requirementExplanation"]]
        action_plan_df.columns = ["Numéro d'exigence", "Exigence IFS Food 8", "Explication (par l’auditeur/l’évaluateur)"]
        return action_plan_df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return None

# Fonction pour générer une recommandation avec Groq et CoT
def generate_ai_recommendation_groq(non_conformity, guide_row, use_cot):
    groq = get_groq_provider()
    if not groq:
        return "Erreur: clé API non fournie."

    general_context = (
        "En tant qu'expert en IFS Food 8, "
        "fournissez des recommandations pour la correction, le type de preuve, "
        "la cause probable, et les actions correctives."
    )
    
    prompt = f"""
    {general_context}
    Voici une non-conformité issue d'un audit IFS Food 8 :
    - Exigence : {non_conformity['Numéro d\'exigence']}
    - Description : {non_conformity['Exigence IFS Food 8']}
    - Constat détaillé : {non_conformity['Explication (par l’auditeur/l’évaluateur)']}
    
    Basé sur le guide IFSv8 pour cette exigence :
    - Bonnes pratiques : {guide_row['Good practice']}
    - Éléments à vérifier : {guide_row['Elements to check']}
    - Exemple de question à poser : {guide_row['Example questions']}
    
    {f'Appliquez une analyse détaillée avec une approche de type "Chain of Thought".' if use_cot else ''}
    """

    try:
        # Si CoT est activé, ajouter une requête de réflexion étape par étape
        return groq.generate(prompt, max_tokens=1000, temperature=0, use_cot=use_cot)
    except Exception as e:
        st.error(f"Erreur lors de la génération de la recommandation : {str(e)}")
        return None

# Fonction pour récupérer les informations du guide en fonction du numéro d'exigence
def get_guide_info(num_exigence, guide_df):
    guide_row = guide_df[guide_df['NUM_REQ'].str.contains(num_exigence, na=False)]
    if guide_row.empty:
        st.error(f"Aucune correspondance trouvée pour le numéro d'exigence : {num_exigence}")
        return None
    return guide_row.iloc[0]

# Fonction principale
def main():
    st.markdown(
        """
        <div class="main-header">Assistant VisiPilot pour Plan d'Actions IFS</div>
        """, 
        unsafe_allow_html=True
    )
    
    # Ajouter l'explication
    with st.expander("Comment utiliser cette application"):
        st.write("""
        **Étapes d'utilisation:**
        1. Téléchargez votre plan d'actions IFSv8.
        2. Sélectionnez un numéro d'exigence.
        3. Cochez 'Analyse détaillée (Chain of Thought)' si nécessaire.
        4. Générez des recommandations basées sur la non-conformité.
        """)
    
    if 'recommendation_expanders' not in st.session_state:
        st.session_state['recommendation_expanders'] = {}
    
    # Clé API Groq
    api_key = st.text_input("Entrez votre clé API Groq:", type="password")
    if api_key:
        st.session_state.api_key = api_key

    uploaded_file = st.file_uploader("Téléchargez votre plan d'action (fichier Excel)", type=["xlsx"])
    
    if uploaded_file:
        action_plan_df = load_action_plan(uploaded_file)
        if action_plan_df is not None:
            guide_df = pd.read_csv("https://raw.githubusercontent.com/M00N69/Action-planGroq/main/Guide%20Checklist_IFS%20Food%20V%208%20-%20CHECKLIST.csv")
            
            st.write("## Plan d'Action IFS")
            for index, row in action_plan_df.iterrows():
                cols = st.columns([1, 4, 4, 2])
                cols[0].write(row["Numéro d'exigence"])
                cols[1].write(row["Exigence IFS Food 8"])
                cols[2].write(row["Explication (par l’auditeur/l’évaluateur)"])
                
                # Ajout du checkbox pour Chain of Thought
                use_cot = st.checkbox("Analyse détaillée (Chain of Thought)", key=f"cot_{index}")
                
                # Bouton pour générer les recommandations
                cols[3].button(
                    "Générer Recommandation", 
                    key=f"generate_{index}",
                    on_click=generate_recommendation_and_expand,
                    args=(index, row, guide_df, use_cot)
                )
                
                if index in st.session_state['recommendation_expanders']:
                    expander = st.expander(f"Recommandation pour Numéro d'exigence: {row['Numéro d\'exigence']}", expanded=True)
                    expander.markdown(st.session_state['recommendation_expanders'][index]['text'])

def generate_recommendation_and_expand(index, row, guide_df, use_cot):
    guide_row = get_guide_info(row["Numéro d'exigence"], guide_df)
    
    if guide_row is not None:
        recommendation_text = generate_ai_recommendation_groq(row, guide_row, use_cot)
        
        if recommendation_text:
            st.success("Recommandation générée avec succès!")
            st.session_state['recommendation_expanders'][index] = {'text': recommendation_text}

if __name__ == "__main__":
    main()






