import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer, WhitespaceTokenizer, RegexpTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

# Download necessary resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

text = "Hello there! I'm testing various tokenization methods: whitespace, punctuation-based, treebank, tweet & MWE."

# Tokenizers
tokenizers = {
    "Whitespace": WhitespaceTokenizer(),
    "Punctuation-based": RegexpTokenizer(r'\w+'),
    "Treebank": TreebankWordTokenizer(),
    "Tweet": TweetTokenizer(),
    "MWE": MWETokenizer([('testing', 'various'), ('tokenization', 'methods')])
}

print("\n--- Tokenization ---")
tokens = {}
for name, tokenizer in tokenizers.items():
    if name == "MWE":
        tokens[name] = tokenizer.tokenize(word_tokenize(text))
    else:
        tokens[name] = tokenizer.tokenize(text)
    print(f"{name} Tokenization: {tokens[name]}")

# Use punctuation tokens for stemming and lemmatization
base_tokens = tokens["Punctuation-based"]

print("\n--- Stemming ---")
for stemmer_name, stemmer in {
    "Porter": PorterStemmer(),
    "Snowball": SnowballStemmer("english")
}.items():
    stems = [stemmer.stem(w) for w in base_tokens]
    print(f"{stemmer_name} Stemming: {stems}")

print("\n--- Lemmatization ---")
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in base_tokens]
print(f"Lemmatization: {lemmatized}")
