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
    import pickle
    from collections import defaultdict
    import math
    
    ORIGINAL_FILE = './data/documents.csv'
    PROCESSED_FILE = './data/processed.csv'
    INVIDX_FILE = './data/invidx.pickle'
    IDF_FILE = './data/idf.csv'
    
    doc_collection = pd.read_csv(ORIGINAL_FILE, index_col=0)
    
    # Save processed text
    doc_collection['data'] = doc_collection['data'].apply(lambda x: ' '.join(preprocess_text(x)))
    doc_collection.index.name = 'id'
    doc_collection.to_csv(PROCESSED_FILE, index=True)
    
    # Save inverted index
    docs_tokenized = []
    inv_idx = defaultdict(dict)
    for id, row in doc_collection.iterrows():
        doc_data = row['data']
        doc_terms = doc_data.split()
        docs_tokenized.append(doc_terms)
        for term in doc_terms:
            inv_idx[term][id] = inv_idx[term].get(id, 0) + 1
            
    with open(INVIDX_FILE, 'wb') as f:
        pickle.dump(inv_idx, f)
        
    # Save IDF
    num_docs = len(doc_collection)
    df, idf = {}, {}
    for term in inv_idx:
        df[term] = len(inv_idx[term]) 
        idf[term] = round(math.log(num_docs / df[term]), 2)
    
    pd.DataFrame({"term": idf.keys(), "idf": idf.values()}).to_csv(IDF_FILE, index=False)
        