# test_comparison.py

import time
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Define relevant grants for the query (expanded to include more relevant grants)
relevant_indices = [58, 18, 10, 35]  # Indices of relevant grants

# Evaluation Metrics Implementation
def calculate_precision_at_k(retrieved_indices, relevant_indices, k):
    retrieved_k = retrieved_indices[:k]
    relevant_set = set(relevant_indices)
    retrieved_set = set(retrieved_k)
    num_relevant = len(retrieved_set & relevant_set)
    precision_at_k = num_relevant / k
    return precision_at_k

def calculate_average_precision(retrieved_indices, relevant_indices):
    relevant_set = set(relevant_indices)
    num_relevant = 0
    sum_precision = 0.0
    for i, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_set:
            num_relevant += 1
            precision = num_relevant / i
            sum_precision += precision
    if num_relevant == 0:
        return 0.0
    average_precision = sum_precision / min(len(relevant_set), num_relevant)
    return average_precision

# Implement NDCG metric
def calculate_dcg(relevance_scores):
    dcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        dcg += (2 ** rel - 1) / np.log2(i + 1)
    return dcg

def calculate_ndcg(retrieved_indices, relevance_dict, k):
    relevance_scores = [relevance_dict.get(idx, 0) for idx in retrieved_indices[:k]]
    dcg = calculate_dcg(relevance_scores)
    # Ideal DCG (IDCG)
    ideal_relevance_scores = sorted(relevance_dict.values(), reverse=True)[:k]
    idcg = calculate_dcg(ideal_relevance_scores)
    if idcg == 0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

# Set K for Precision@K
K = 5

# Define relevance scores (graded relevance)
relevance_dict = {
    58: 2,  # Quantum Computing Grant (Highly Relevant)
    18: 1,  # AI Research Grant (Somewhat Relevant)
    10: 1,  # Education Advancement Grant (Somewhat Relevant)
    35: 1,  # Educational Technology Grant (Somewhat Relevant)
}

# Baseline Metrics
baseline_precision_at_k = calculate_precision_at_k(baseline_indices, relevant_indices, K)
baseline_average_precision = calculate_average_precision(baseline_indices, relevant_indices)
baseline_ndcg = calculate_ndcg(baseline_indices, relevance_dict, K)

# Quantum Search Metrics
quantum_precision_at_k = calculate_precision_at_k(quantum_indices, relevant_indices, K)
quantum_average_precision = calculate_average_precision(quantum_indices, relevant_indices)
quantum_ndcg = calculate_ndcg(quantum_indices, relevance_dict, K)

# Print comparison
print("\nBaseline Search Metrics:")
print(f"Precision@{K}: {baseline_precision_at_k:.4f}")
print(f"Average Precision (AP): {baseline_average_precision:.4f}")
print(f"NDCG@{K}: {baseline_ndcg:.4f}")

print("\nQuantum Search Metrics:")
print(f"Precision@{K}: {quantum_precision_at_k:.4f}")
print(f"Average Precision (AP): {quantum_average_precision:.4f}")
print(f"NDCG@{K}: {quantum_ndcg:.4f}")

# Optionally, print top results from both searches
print("\nTop results from Baseline Search:")
for idx in baseline_indices[:K]:
    grant = grants[idx]
    similarity = baseline_similarities[idx]
    print(f"Title: {grant.title}, Similarity Score: {similarity:.4f}")

print("\nTop results from Quantum Search:")
for idx in quantum_indices[:K]:
    grant = grants[idx]
    print(f"Title: {grant.title}")
