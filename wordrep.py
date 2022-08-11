import math
import os
import time

import pandas as pd
from collections import defaultdict, Counter
import numpy as np

from preprocessing import preprocess_text
from metrics import *

from gensim.models import FastText
from tqdm import tqdm

def get_tf(inv_idx, doc_id):
    frequencies = {}
    for term in inv_idx:
        if doc_id in inv_idx[term].keys():
            frequencies[term] = inv_idx[term][doc_id]
    return frequencies
    
def sent2vec(model, sentence):
    '''
    Args:
        model: a dictionary of term-vector
        sentence: tokenized query/document
    Returns:
        A sentence vector calculated as weighted average of word vectors
    '''
    # calculate normalized term frequency
    tf = Counter(sentence)
    norm = 1.0/sum(tf.values())
    for t in tf:
       tf[t] = tf[t]*norm
    # do weighted average of word vectors
    sent_vec = None
    for word in sentence:
        word_vec = model[word]
        word_freq = tf[word]
        if sent_vec is None:
            sent_vec = word_freq*word_vec
        else:
            sent_vec = sent_vec + word_freq*word_vec

    sent_len = len(sentence)
    sent_vec = sent_vec/sent_len
    return sent_vec
        
if __name__ == "__main__":
    doc_collection = pd.read_csv('./data/processed.csv')
    num_docs = len(doc_collection)

    # FastText:
    # FastText model is able to handle unknown words
    # Use FastText: https://radimrehurek.com/gensim/models/fasttext.html

    # train model on our corpus
    fname = "./data/fasttext.model"
    if os.path.exists(fname):
        model = FastText.load(fname)
    else:
        new_sentences = list(doc_collection['data'])
        model = FastText(vector_size=100, window=3, min_count=1, sentences=new_sentences, epochs=10)
        model.save(fname)
    
    # Tokenize documents
    docs_tokenized = []
    for id, row in tqdm(doc_collection.iterrows()):
        doc_data = row['data']
        doc_terms = doc_data.split()
        docs_tokenized.append(doc_terms)
    
    # Vectorize documents
    all_doc_vec = []
    doc_vec_map = {}
    for id, doc in enumerate(docs_tokenized):
        sent_vec = sent2vec(model.wv, doc)
        all_doc_vec.append(sent_vec)
        doc_vec_map[id] = sent_vec
            
    import pickle
    fname = './data/wr_docs.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(doc_vec_map, f)

    # Test example
    '''
    Query: Is there a safari in Singapore?
    Documents: ['night-safari', 'river-safari-singapore', 'singapore-zoo']
    '''
    q = "Is there a safari in Singapore?"
    query = preprocess_text(q)
    print(f"Processed query: {query}")
    
    query_vector = sent2vec(model.wv, query)
    
    doc_scores = {}
    start = time.time()
    for id in doc_collection.index:
        doc_vector = all_doc_vec[id]
        score = cosinesimilarity(query_vector, doc_vector)
        doc_scores[id] = score

    ranking = get_ranking(doc_scores=doc_scores, top_K=3)
    end = time.time()
    print(f"Time taken = {end-start}s")
    
    # Evaluate example
    eval_docs = ['night-safari', 'river-safari-singapore', 'singapore-zoo']
    retr = [doc_collection.iloc[item[0]]['filename'] for item in ranking]
    print("Retrieved:", retr)
    r = get_relevance_scores(retr, eval_docs)
    print("Relevance: ", r)
    print(f"Avg. precision = {average_precision(r)}, reciprocal rank = {reciprocal_rank(r)}")
    result = ranking[0][0]
    top_doc = doc_collection.iloc[result]
    print(f"\nDocument {top_doc['filename']}")
