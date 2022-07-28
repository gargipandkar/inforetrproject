from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
import re

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:’'"\,<>./?@#$%^&*_~'''
lemmatizer = WordNetLemmatizer()

def preprocess_word(word):
    word = word.lower()
    word = lemmatizer.lemmatize(word)
    word = contractions.fix(word)
    word =  re.sub('\W+','', word)
    return word

def preprocess_text(text): 
    word_tokens = word_tokenize(text)
    filtered = [preprocess_word(w) for w in word_tokens if w.lower() not in stop_words and w not in punctuations]
    return filtered