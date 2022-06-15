from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def preprocess(text):
    word_tokens = word_tokenize(text)
    filtered = [w.lower() for w in word_tokens if not w.lower() in stop_words and w.lower() not in punctuations]
    return filtered