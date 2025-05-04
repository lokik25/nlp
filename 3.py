import pandas as pd, re, nltk, pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Download required resources
for res in ["punkt", "stopwords", "wordnet"]:
    nltk.download(res)

# Sample data
df = pd.DataFrame({
    "text": [
        "Natural Language Processing is an exciting field of AI!",
        "Machine learning helps in speech recognition.",
        "Deep learning advances AI technologies significantly.",
        "TF-IDF is useful for text representation in NLP.",
        "Lemmatization reduces words to their base form."
    ],
    "label": ["AI", "ML", "DL", "NLP", "Preprocessing"]
})

# Text cleaning and preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"\W", " ", text.lower())
    tokens = word_tokenize(text)
    return " ".join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)

df["processed_text"] = df["text"].apply(preprocess)

# Encode labels
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# TF-IDF vectorization
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df["processed_text"])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Save data and models
df.to_csv("processed_text_data.csv", index=False)
tfidf_df.to_csv("tfidf_representation.csv", index=False)
with open("label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)
with open("tfidf_vectorizer.pkl", "wb") as f: pickle.dump(tfidf, f)

# Display results
print("\n--- Processed Data ---\n", df[["text", "processed_text", "label", "label_encoded"]])
print("\n--- TF-IDF Representation ---\n", tfidf_df.head())
