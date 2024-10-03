from flask import Flask, request, jsonify
import PyPDF2
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests

app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to compare PDF content with search results 
def compare_with_search_results(pdf_text, serper_api_key, query):
    headers = {"X-API-KEY": serper_api_key}
    response = requests.get(f"https://google.serper.dev/search?q={query}", headers=headers)
    search_results = response.json().get('organic', [])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    


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
        similarity_results.append({
            'title':search_title, 
            'link':search_link, 
            'similarity_percentage': similarity_percentage})
    
    
    return similarity_results

#to verify wheter API is working
@app.route('/', methods = ['GET'])
def voir_sur_web():
    return jsonify({'salutation': 'hello'})
# API route to handlcheck-plagiarisme plagiarism check
@app.route('/check-plagiarism', methods=['POST'])
def check_plagiarism():
    if 'pdf_file' not in request.files or 'serper_api_key' not in request.form:
        return jsonify({'error': 'Missing PDF file or API key'}), 400

    pdf_file = request.files['pdf_file']
    serper_api_key = request.form['serper_api_key']
    query = request.form.get('query', '')

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_file)
    
    # Compare with search results
    similarities = compare_with_search_results(pdf_text, serper_api_key, query)
    
    return jsonify(similarities)

if __name__ == '__main__':
    app.run(debug=True)
