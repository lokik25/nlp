from sklearn.preprocessing import LabelEncoder
import pandas as pd
import string
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download("stopwords")
# Sample data
data = {
    "Text": [
        "I love Natural Language Processing!",
        "Machine Learning is amazing.",
        "AI is the future of technology."
    ],
    "Label": ["Positive", "Positive", "Neutral"]
}
df = pd.DataFrame(data)

# 3.1 Cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['Cleaned'] = df['Text'].apply(clean_text)
print(df[['Text', 'Cleaned']])
print("\n")

# 3.2 Lemmatization + Stopword removal
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

df['Processed'] = df['Cleaned'].apply(preprocess)
print(df[['Cleaned', 'Processed']])
print("\n")

# 3.3 Label Encoding
le = LabelEncoder()
df['Encoded_Label'] = le.fit_transform(df['Label'])
print("Label classes mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df[['Label', 'Encoded_Label']])
print("\n")

# 3.4 TF-IDF Representation
tfidf_vectorizer = TfidfVectorizer()
X_features = tfidf_vectorizer.fit_transform(df['Processed'])
print(pd.DataFrame(X_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out()))
print("\n")

# Save outputs
df.to_csv('processed_data.csv', index=False)
import joblib
joblib.dump(X_features, 'tfidf_features.pkl')

print(df)
