import math
import time

import pandas as pd
from collections import defaultdict

from preprocessing import preprocess_text
from metrics import *
from qa import qa_pipeline

class BM25:

    def __init__(self, inverted_idx, docs_tokenized, num_docs):
        self.inv_idx = inverted_idx
        self.num_docs = num_docs
        self.docs_tokenized = docs_tokenized
        self.avg_doc_len = sum([len(d) for d in docs_tokenized])/num_docs
        self.df = self.calculate_df()
        self.idf = self.calculate_idf()
    
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
    BM25
    Score a document given a tokenized query, doc, average document length, inverted index, filename/docID and document frequency. 
    Input:
    query= [ word3 , word7 , word8 ] 
    doc= [ word1 , word3 , word8 ]
    average document length= 5
    inverted index= {word1 : {doc1: 1 , doc2: 4 ...}, ... }
    filename= doc1
    document frequency= { word1 : 1, word2 : 3, word3 : 4, word4 : 2}
    '''
    def get_score(self, query, doc_id, k1=1.5, b=0.75):
        score = 0.0
        tf = self.get_tf(doc_id)
        doc_len = len(self.docs_tokenized[doc_id])
        for term in query:
            if term not in tf.keys():
                continue

            c_t = self.idf[term]
            num = (k1+1)*tf[term]
            denom = tf[term] + k1*((1-b) + b*(doc_len/self.avg_doc_len))
            doc_wt = num/denom
            score += c_t * doc_wt

        return score
    
    def get_ranking(self, doc_scores, top_K = None):
        ranking = sorted(doc_scores.items(),
                    key=lambda item: item[1], reverse=True)
        if top_K == None:
            return ranking
        else:
            return ranking[:top_K]

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

    bm25 = BM25(inverted_idx = inv_idx, docs_tokenized=docs_tokenized, num_docs=num_docs)
    
    eval = pd.read_csv('./data/evaluation.csv')
    eval['Documents'] = eval['Documents'].apply(lambda x: x.strip("[]").replace("'","").split(", "))
    rs = []
    times = []
    for _, row in eval.iterrows():
        q = row['Query']
        query = preprocess_text(q)
        print(f"Processed query: {query}")
        maxscore = -1*float("inf")
        result = None

        doc_scores = {}
        i = 0
        start = time.time()
        for id in doc_collection.index:
            score = bm25.get_score(query, id)
            doc_scores[id] = score
            if score > maxscore:
                maxscore = score
                result = i
            i += 1

        ranking = bm25.get_ranking(doc_scores=doc_scores, top_K=3)
    #     try:
    #         stop_idx = [item[1] for item in ranking].index(0)
    #     except ValueError:
    #         stop_idx = len(ranking)
    #     stop_idx = min(stop_idx, 3)
    #     retr = [doc_collection.iloc[item[0]]['filename'] for item in ranking[:stop_idx]]
    #     end = time.time()
    #     times.append(end-start)
    #     print(f"Time taken = {end-start}s")
    #     print("Retrieved:", retr)
    #     r = get_relevance_scores(retr, row['Documents'])
    #     print("Relevance: ", r)
    #     print(f"Avg. precision = {average_precision(r)}, reciprocal rank = {reciprocal_rank(r)}")
    #     rs.append(r)
    #     top_doc = doc_collection.iloc[result]
    #     print(f"\nDocument {top_doc['filename']}")
        
    #     # get answer 
    #     qa = {'question': q,
    #           'context': top_doc['data']
    #           }
    #     start = time.time()
    #     ans = qa_pipeline(qa)
    #     end = time.time()
    #     print(f"Q - {q}\nA - {ans['answer']}\nTime taken = {end-start}")
        
    #     print()

    # print(mean_average_precision(rs))
    # print(mean_reciprocal_rank(rs))
    # print(np.mean(times))
