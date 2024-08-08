## UPLOAD DU FCHIER EXCEL ET MIS EEN FORME

Le code identifié les bonnes colonnes et le bon en-tête pour le fichier Excel en suivant plusieurs étapes. Voici une explication détaillée de ce processus :

Téléchargement du fichier Excel :

L'utilisateur télécharge un fichier Excel via st.file_uploader.
Chargement et Prévisualisation du Fichier :

Le fichier est chargé dans un DataFrame sans spécifier d'en-tête (header=None) pour prévisualiser les données et identifier où se trouve la ligne d'en-tête correcte.
Identification de la Ligne d'En-tête :

Une ligne d'en-tête spécifique est choisie (header_row_index = 12). Ce choix est basé sur la structure spécifique du fichier fourni.
Chargement du DataFrame avec la Ligne d'En-tête Correcte :

Une fois que la ligne d'en-tête correcte est identifiée, le fichier est chargé à nouveau en utilisant cette ligne d'en-tête.
Renommage des Colonnes et Sélection des Colonnes Requises :

Les colonnes du DataFrame sont renommées pour correspondre aux noms attendus.
Les colonnes attendues sont ensuite sélectionnées pour s'assurer que seules les colonnes pertinentes sont utilisées.
Détail du Code
Voici un extrait de code qui montre ces étapes :

python
Copier le code
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
Explication des Étapes
Chargement Initial du Fichier :

python
Copier le code
temp_df = pd.read_excel(uploaded_file, header=None)
Charge le fichier sans en-tête pour inspecter les colonnes et identifier où se trouve la ligne d'en-tête correcte.
Identification de la Ligne d'En-tête :

python
Copier le code
header_row_index = 12
action_plan_df = pd.read_excel(uploaded_file, header=header_row_index)
Charge à nouveau le fichier en spécifiant la ligne 13 (index 12) comme ligne d'en-tête.
Renommage des Colonnes :

python
Copier le code
action_plan_df.columns = [col.strip() for col in action_plan_df.columns]
action_plan_df = action_plan_df.rename(columns={
    "Numéro d'exigence": "Numéro d'exigence",
    "Exigence IFS Food 8": "Exigence IFS Food 8",
    "Notation": "Notation",
    "Explication (par l’auditeur/l’évaluateur)": "Explication (par l’auditeur/l’évaluateur)"
})
Supprime les espaces autour des noms de colonnes et renomme les colonnes pour qu'elles correspondent aux noms attendus.
Sélection des Colonnes Requises :

python
Copier le code
expected_columns = ["Numéro d'exigence", "Exigence IFS Food 8", "Notation", "Explication (par l’auditeur/l’évaluateur)"]
action_plan_df = action_plan_df[expected_columns]
Sélectionne uniquement les colonnes attendues pour s'assurer que le DataFrame contient les données pertinentes.
