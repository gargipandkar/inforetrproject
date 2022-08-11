import math
import time

import pandas as pd
import json
from collections import defaultdict, Counter

from preprocessing import preprocess_text
from metrics import *

class VSM:

    def __init__(self, inverted_idx, docs_tokenized, num_docs, idf=None):
        self.inv_idx = inverted_idx
        self.docs_tokenized = docs_tokenized
        self.num_docs = num_docs
        self.avg_doc_len = sum([len(d) for d in docs_tokenized])/num_docs
        if idf is None:
            self.df, self.idf = self.calculate_df_idf()
        else:
            self.idf = idf
            
        self.vocab = self.get_vocab()
        
    def get_vocab(self):
        all_terms = []
        for id, doc in enumerate(self.docs_tokenized):
            all_terms += doc
        vocab = list(set(all_terms))
        vocab = sorted(vocab)
        return vocab
        
    '''
    Calculate TF
    Given an inverted index and a filename/docID, return a dictionary
    with key as a word and value as the number of times it occurs in the doc. For instance:
    inputs: {word1 : {doc1: 1 , doc2: 4 ...}, ... } and doc1
    output: { word1 : 1, word2 : 2, word3 : 1, word4 : 1}
    '''
    def get_tf(self, doc_id):
        frequencies = {}
        for term in self.inv_idx:
            if doc_id in self.inv_idx[term].keys():
                frequencies[term] = self.inv_idx[term][doc_id]
        return frequencies

    def get_tf_query(self, query):
        frequencies = {}
        if isinstance(query, str):
            query = set(preprocess_text(query))
        frequencies = Counter(query)
        return frequencies
    
    '''  
    Calculate DF
    Given an inverted index, return a dictionary with the number of documents (value) 
    that a term (key) appears in. For instance:
    input: {word1 : {doc1: 1 , doc2: 4 ...}, ... }
    output: { word1 : 1, word2 : 3, word3 : 4, word4 : 2}
    '''
    def calculate_df(self):
        df = {}
        for term in self.inv_idx:
            df[term] = len(self.inv_idx[term]) 
        return df
    
    '''
    IDF
    input = { 'a': 1, 'b' : 2, 'c' : 3, 'd' : 2, 'e' : 1}
    output =  { 'a' : 1.1, 'b' : 0.41, 'c' : 0.0, 'd' : 0.41, 'e' : 1.1}
    '''
    def calculate_idf(self):
        idf = {}
        for term, freq in self.df.items():
            idf[term] = round(math.log((self.num_docs) / (freq)), 2)
        return idf

    '''
    DF and IDF
    '''
    def calculate_df_idf(self):
        df, idf = {}, {}
        for term in self.inv_idx:
            df[term] = len(self.inv_idx[term]) 
            idf[term] = round(math.log(self.num_docs / df[term]), 2)
        return df, idf
    
    def get_doc_vector(self, doc_id):
        terms = self.vocab
        tf = self.get_tf(doc_id)
        doc_vector = []
        for term in terms:
            term_tf = tf.get(term, 0)
            term_idf = self.idf.get(term, 0)
            doc_vector.append(term_tf * term_idf)
        return doc_vector
    
    def get_query_vector(self, query):
        terms = self.vocab
        tf = self.get_tf_query(query)
        query_vector = []
        for term in terms:
            term_tf = tf.get(term, 0)
            term_idf = self.idf.get(term, 0)
            query_vector.append(term_tf * term_idf)
        return query_vector

    def get_score(self, query_vector, doc_vector):
        score = cosinesimilarity(doc_vector, query_vector)
        return score

if __name__ == "__main__":
    doc_collection = pd.read_csv('./data/documents.csv')
    num_docs = len(doc_collection)
        
    docs_tokenized = []
    inv_idx = defaultdict(dict)
    
    for id, row in doc_collection.iterrows():
        doc_data = row['data']
        doc_terms = preprocess_text(doc_data)
        docs_tokenized.append(doc_terms)
        for term in doc_terms:
            inv_idx[term][id] = inv_idx[term].get(id, 0) + 1
    
    vsm = VSM(inverted_idx=inv_idx, docs_tokenized=docs_tokenized, num_docs=num_docs)
      
    # Vectorize documents
    doc_vec_map = {}
    for id, doc in enumerate(docs_tokenized):
        doc_vec = vsm.get_doc_vector(id)
        doc_vec_map[id] = doc_vec
            
    import pickle
    fname = './data/vsm_docs.pickle'
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
    
    query_vector = vsm.get_query_vector(query)
    
    doc_scores = {}
    start = time.time()
    for id in doc_collection.index:
        doc_vector = doc_vec_map[id]
        score = vsm.get_score(query_vector, doc_vector)
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
