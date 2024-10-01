import streamlit as st
import PyPDF2
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import numpy as np

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Fonction pour comparer le texte PDF avec les résultats de recherche
def compare_with_search_results(pdf_text, serper_api_key, query):
    headers = {"X-API-KEY": serper_api_key}
    response = requests.get(f"https://google.serper.dev/search?q={query}", headers=headers)
    search_results = response.json().get('organic', [])
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # Stocker les résultats de similarité
    similarity_results = []
    
    for result in search_results:
        search_text = result['snippet']
        search_title = result['title']  # Extraction du titre du résultat
        search_link = result['link']
        
        inputs = tokenizer(pdf_text, search_text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            similarity = torch.softmax(outputs.logits, dim=1)
        
        similarity_percentage = (similarity[0][1].item()) * 100
        similarity_results.append((search_title, search_link, similarity_percentage))
        # similarity_results.append((result['title'], similarity_percentage))
    
    return similarity_results

# Interface Streamlit
st.title("Bienvenue sur JCrify.")

st.sidebar.header('''
                  Posez vos questions à JCassistant.
                  Votre chatbot pour toutes les questions relatives à JCrify.''')

st.sidebar.empty()

st.sidebar.image("C:\\Users\\IR S\\Downloads\\chatbot_image_billing.jpg", caption="JCassistant", width=200)

st.write('''
Le logiciel anti plagiat pour la validation des thèses soumises en format PDF.''')


# Uploader le fichier PDF
uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")

# Saisir la clé API et la requête de recherche
#serper_api_key = st.text_input("Entrez votre clé API Serper")
serper_api_key = 'b4ab1717c887456b515197eeb8e599fae40b9888'
query = st.text_input("Entrez Le titre du mémoir")

# Analyser le fichier PDF une fois téléchargé
if uploaded_file is not None and query :  #serper_api_key
    # Extraire le texte du PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Comparer avec les résultats de recherche
    
    st.write("Analyse en cours...")
    similarity_results = compare_with_search_results(pdf_text, serper_api_key, query)  #query
    
    # Afficher les résultats de similarité
    st.write("Résultats des similarités :")
    plagiat_values = []
    for title, link, similarity in similarity_results:
        plagiat_values.append(similarity)
        if similarity > 52.0:
           st.write(f"Le résultat de recherche à la page: {title}, venu du lien: {link}, donne une Similarité de : {similarity:.2f}%. Nous avons donc un plagiat.")
        else:
            st.write(f"Le résultat de recherche à la page: {title}, venu du lien: {link}, donne une Similarité de : {similarity:.2f}%. La  thèse est acceptée.")   
    st.write(f"Nous avons en moyenne : {np.mean(plagiat_values)}%")

    

