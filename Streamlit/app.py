import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# --- Constantes (adaptées de ton script) ---
MODEL_NAME_FOR_APP = "camembert-base"
MAX_LEN_FOR_APP = 160
MODEL_SAVE_PATH_FOR_APP = "../notebooks/BERT_fr/save_model_fr/camembert_coherence_final_model.bin"
TOKENIZER_SAVE_PATH_FOR_APP = "../notebooks/BERT_fr/save_model_fr/camembert_coherence_tokenizer/"
DEVICE_FOR_APP = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Fonction de nettoyage de texte (identique à ton script) ---
def clean_text_for_prediction(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text) # Enlever les balises HTML
    text = re.sub(r'[^a-zA-Z0-9\s\.\?,!àâéèêëîïôûùüçÀÂÉÈÊËÎÏÔÛÙÜÇ\']', '', text) # Garder seulement les caractères pertinents
    text = text.lower() # Mettre en minuscule
    text = re.sub(r'\s+', ' ', text).strip() # Normaliser les espaces
    return text

# --- Fonction pour charger le modèle et le tokenizer (mise en cache) ---
 # Important pour ne charger le modèle qu'une fois
@st.cache_resource 
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVE_PATH_FOR_APP)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FOR_APP, num_labels=2)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH_FOR_APP, map_location=DEVICE_FOR_APP))
        model.to(DEVICE_FOR_APP)
        model.eval()
        st.sidebar.success("Modèle et tokenizer chargés avec succès!")
        return model, tokenizer
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement du modèle/tokenizer: {e}")
        st.sidebar.error(f"Vérifiez que les chemins existent :")
        st.sidebar.error(f"Modèle: {MODEL_SAVE_PATH_FOR_APP}")
        st.sidebar.error(f"Tokenizer: {TOKENIZER_SAVE_PATH_FOR_APP}")
        return None, None

# --- Fonction de prédiction (identique à ton script, mais utilise les variables de l'app) ---
def predict_coherence(question, answer, model, tokenizer, device, max_len, clean_fn):
    cleaned_question = clean_fn(question)
    cleaned_answer = clean_fn(answer)

    encoding = tokenizer.encode_plus(
        cleaned_question,
        cleaned_answer,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False, # Important pour Camembert/RoBERTa
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probs, dim=1).cpu().item()
        probability_score = probs[0][prediction_idx].cpu().item()

    return prediction_idx, probability_score

# --- Interface Streamlit ---
st.set_page_config(page_title="Cohérence Q-R", layout="wide")
st.title("Prédicteur de Cohérence Question-Réponse")
st.markdown("""
Bienvenue ! Cette application utilise un modèle Camembert fine-tuné pour prédire
si une réponse donnée est cohérente par rapport à une question posée.
""")

# Charger le modèle et le tokenizer
model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    st.header("Entrez votre question et réponse ici :")

    col1, col2 = st.columns(2)
    with col1:
        question_input = st.text_area("Question :", height=150, placeholder="Ex: Où sont les toilettes ?")
    with col2:
        answer_input = st.text_area("Réponse :", height=150, placeholder="Ex: Au fond du couloir à droite.")

    if st.button("🔍 Prédire la Cohérence", type="primary", use_container_width=True):
        if question_input and answer_input:
            with st.spinner("Prédiction en cours..."):
                pred_label, pred_prob = predict_coherence(
                    question_input,
                    answer_input,
                    model,
                    tokenizer,
                    DEVICE_FOR_APP,
                    MAX_LEN_FOR_APP,
                    clean_text_for_prediction
                )

            st.subheader("Résultat de la Prédiction :")
            coherence_status_text = "✅ Cohérent" if pred_label == 1 else "❌ Non Cohérent"
            confidence_percentage = pred_prob * 100

            if pred_label == 1:
                st.success(f"**Statut :** {coherence_status_text}")
            else:
                st.error(f"**Statut :** {coherence_status_text}")

            st.metric(label="Score de Confiance", value=f"{confidence_percentage:.2f}%")

        else:
            st.warning("Veuillez entrer une question ET une réponse pour la prédiction.")
else:
    st.error("Le modèle n'a pas pu être chargé. L'application ne peut pas fonctionner. Vérifiez les logs dans la console et les chemins des fichiers.")

# Informations sur le modèle dans la sidebar
st.sidebar.header(" Informations sur le Modèle")
st.sidebar.markdown(f"**Modèle de base :** `{MODEL_NAME_FOR_APP}`")
st.sidebar.markdown(f"**Longueur Max. Séquence :** `{MAX_LEN_FOR_APP}`")
st.sidebar.markdown(f"**Dispositif :** `{DEVICE_FOR_APP}`")
st.sidebar.markdown(f"**Chemin Modèle :** `{MODEL_SAVE_PATH_FOR_APP}`")
st.sidebar.markdown(f"**Chemin Tokenizer :** `{TOKENIZER_SAVE_PATH_FOR_APP}`")

st.sidebar.markdown("---")
st.sidebar.markdown("Application créée avec Streamlit.")