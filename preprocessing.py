from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

stop_words = set(stopwords.words('english'))
punctuations = '''!()-[]{};:’‘'"\,<>./?@#$%^&*_~'''
lemmatizer = WordNetLemmatizer()

def preprocess_word(word):
    word = word.lower()
    word = lemmatizer.lemmatize(word)
    word = contractions.fix(word)
    return word

def preprocess_text(text):
    '''Returns list of words'''
    text = text.replace('—', ' ')
    word_tokens = word_tokenize(text)
    filtered = [preprocess_word(w) for w in word_tokens if w.lower() not in stop_words and w not in punctuations]
    return filtered

if __name__=="__main__":
    import pandas as pd
    ORIGINAL_FILE = './data/documents.csv'
    PROCESSED_FILE = './data/processed.csv'
    
    doc_collection = pd.read_csv(ORIGINAL_FILE, index_col=0)
        
    doc_collection['data'] = doc_collection['data'].apply(lambda x: ' '.join(preprocess_text(x)))
    doc_collection.index.name = 'id'
    
    doc_collection.to_csv(PROCESSED_FILE, index=True)