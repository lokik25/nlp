from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    "I love natural language processing",
    "Machine learning is fascinating",
    "NLP and machine learning make AI powerful"
]

# 2.1 Bag-of-Words (Count)
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(corpus)
print("BoW Vocabulary:", vectorizer.get_feature_names_out())
print("Count Matrix:\n", X_counts.toarray())

# 2.2 Normalized Count (term frequency)
X_norm_counts = X_counts.astype(float)
for row in range(X_norm_counts.shape[0]):
    X_norm_counts[row] /= X_norm_counts[row].sum()

print("Normalized Count Matrix:\n", X_norm_counts.toarray())

# 2.3 TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
print("TF-IDF Matrix:\n", X_tfidf.toarray())

# 2.4 Word2Vec Embeddings
tokenized_corpus = [sentence.lower().split() for sentence in corpus]
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1, sg=1)
word_vec = w2v_model.wv['machine']
print("Word2Vec embedding (machine):", word_vec)
