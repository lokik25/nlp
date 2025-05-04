import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample dataset
docs = [
    "Natural Language Processing is amazing.",
    "Machine learning and deep learning are parts of AI.",
    "Deep learning helps in NLP tasks like translation.",
    "Word embeddings capture semantic meaning of words.",
    "TF-IDF helps in text representation."
]

# 1. BoW (Raw + Normalized)
print("\n--- Bag-of-Words ---")
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(docs).toarray()
cols = vectorizer.get_feature_names_out()

for label, matrix in [("Count Occurrence", X_bow), 
                      ("Normalized Count", X_bow / X_bow.sum(axis=1, keepdims=True))]:
    print(f"\n{label}:\n", pd.DataFrame(matrix, columns=cols))

# 2. TF-IDF
print("\n--- TF-IDF ---")
X_tfidf = TfidfVectorizer().fit_transform(docs).toarray()
print(pd.DataFrame(X_tfidf, columns=TfidfVectorizer().fit(docs).get_feature_names_out()))

# 3. Word2Vec Embeddings
print("\n--- Word2Vec Embeddings ---")
tokens = [word_tokenize(doc.lower()) for doc in docs]
model = Word2Vec(sentences=tokens, vector_size=10, window=5, min_count=1, workers=4)

for word in ["learning", "deep", "natural", "text"]:
    print(f"Vector for '{word}':\n{model.wv[word]}\n" if word in model.wv else f"'{word}' not in vocabulary.\n")
