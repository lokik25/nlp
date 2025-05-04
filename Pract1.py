# 1.1 Imports
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, TweetTokenizer, MWETokenizer, regexp_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
text = "NLTK's tokenizers are very powerful! You can split text like 'I'm learning NLP.' :) #AI #ML"

# 1.2 Tokenization
# Whitespace
whitespace_tokens = text.split()

# Punctuation-based (Regexp)
punctuation_tokens = regexp_tokenize(text, pattern=r'\w+|[^\w\s]', gaps=False)

# Treebank
treebank_tokenizer = TreebankWordTokenizer()
treebank_tokens = treebank_tokenizer.tokenize(text)

# Tweet
tweet_tokenizer = TweetTokenizer()
tweet_tokens = tweet_tokenizer.tokenize(text)

# Multi-word expressions
mwe_tokenizer = MWETokenizer([('natural', 'language'), ('machine', 'learning')])
mwe_tokens = mwe_tokenizer.tokenize(word_tokenize("I love natural language processing and machine learning!"))

# Display
print("Whitespace:", whitespace_tokens)
print("Punctuation:", punctuation_tokens)
print("Treebank:", treebank_tokens)
print("Tweet:", tweet_tokens)
print("MWE:", mwe_tokens)

# 1.3 Stemming
porter = PorterStemmer()
snowball = SnowballStemmer(language='english')
tokens = word_tokenize(text)

porter_stems = [porter.stem(token) for token in tokens]
snowball_stems = [snowball.stem(token) for token in tokens]

print("Porter Stems:", porter_stems)
print("Snowball Stems:", snowball_stems)

# 1.4 Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatized:", lemmas)
