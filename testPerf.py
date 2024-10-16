# test_comparison.py

import time
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Import your QuantumGrantSearcher and grants data
from qstate import QuantumGrantSearcher, grants

# Initialize the QuantumGrantSearcher
quantum_searcher = QuantumGrantSearcher(grants)

# Define the query
query = "quantum computing research in Boston"

# Define a baseline search function using TF-IDF and cosine similarity
def baseline_search(grants, query):
    start_time = time.time()
    # Prepare the corpus: combine title, description, and location
    corpus = [f"{grant.title} {grant.description} {grant.location}" for grant in grants]
    
    # Preprocess the text
    translator = str.maketrans('', '', string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.WordNetLemmatizer()
    
    def preprocess(text):
        tokens = nltk.word_tokenize(text.lower().translate(translator))
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    corpus = [preprocess(doc) for doc in corpus]
    preprocessed_query = preprocess(query)
    
    # Vectorize the corpus and the query
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([preprocessed_query])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get indices of grants sorted by similarity score
    sorted_indices = np.argsort(-similarities)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Baseline search runtime: {runtime:.4f} seconds")
    return sorted_indices, similarities

# Perform the baseline search
baseline_indices, baseline_similarities = baseline_search(grants, query)

# Perform the quantum search
def quantum_search(quantum_searcher, query):
    start_time = time.time()
    matching_indices, partial_match_scores = quantum_searcher.encode_query(query)
    if matching_indices:
        indices_to_search = matching_indices
    elif partial_match_scores:
        partial_match_scores.sort(key=lambda x: x[1], reverse=True)
        indices_to_search = [index for index, score in partial_match_scores]
    else:
        indices_to_search = []
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Quantum search runtime: {runtime:.4f} seconds")
    return indices_to_search

quantum_indices = quantum_search(quantum_searcher, query)

# Define relevant grants for the query
# For testing purposes, let's assume that the relevant grants are known
# In a real scenario, you might have a labeled dataset or ground truth
relevant_indices = [58]  # Assuming "Quantum Computing Grant" is the relevant one

# Prepare binary relevance vectors
num_items = len(grants)
mlb = MultiLabelBinarizer(classes=range(num_items))

# Baseline search results
baseline_relevance = mlb.fit_transform([baseline_indices])[0]

# Quantum Search results
quantum_relevance = mlb.fit_transform([quantum_indices])[0]

# Ground truth relevance
true_relevance = mlb.fit_transform([relevant_indices])[0]

# Calculate metrics for Baseline Search
baseline_precision = precision_score(true_relevance, baseline_relevance)
baseline_recall = recall_score(true_relevance, baseline_relevance)
baseline_f1 = f1_score(true_relevance, baseline_relevance)
baseline_accuracy = accuracy_score(true_relevance, baseline_relevance)

# Calculate metrics for Quantum Search
quantum_precision = precision_score(true_relevance, quantum_relevance)
quantum_recall = recall_score(true_relevance, quantum_relevance)
quantum_f1 = f1_score(true_relevance, quantum_relevance)
quantum_accuracy = accuracy_score(true_relevance, quantum_relevance)

# Print comparison
print("\nBaseline Search Metrics:")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall: {baseline_recall:.4f}")
print(f"F1-Score: {baseline_f1:.4f}")
print(f"Accuracy: {baseline_accuracy:.4f}")

print("\nQuantum Search Metrics:")
print(f"Precision: {quantum_precision:.4f}")
print(f"Recall: {quantum_recall:.4f}")
print(f"F1-Score: {quantum_f1:.4f}")
print(f"Accuracy: {quantum_accuracy:.4f}")

# Optionally, print top results from both searches
print("\nTop results from Baseline Search:")
for idx in baseline_indices[:5]:
    grant = grants[idx]
    print(f"Title: {grant.title}, Similarity Score: {baseline_similarities[idx]:.4f}")

print("\nTop results from Quantum Search:")
for idx in quantum_indices[:5]:
    grant = grants[idx]
    print(f"Title: {grant.title}")
