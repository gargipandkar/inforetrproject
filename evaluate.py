import os
import time

import pickle
import pandas as pd
from collections import defaultdict

from preprocessing import preprocess_text
from metrics import *
from bm25 import BM25
from queryexp import *
from qa import qa_pipeline

class Engine:
    def __init__(self, collection, name):
        self.collection = collection
        self.name = name
        self.num_docs = len(collection)
        self.docs_tokenized = self.tokenize(collection)
        
    def tokenize(self, collection):
        if collection is None:
            collection = self.collection
        docs_tokenized = []
        for id, row in self.collection.iterrows():
            doc_data = row['data']
            doc_terms = doc_data.split()
            docs_tokenized.append(doc_terms)
        return docs_tokenized
    
    def get_inv_idx(self):
        # read from file, else calculate
        inv_idx = defaultdict(dict)
        fname = './data/invidx.pickle'
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                inv_idx = pickle.load(f)
        else:
            for id, row in self.collection.iterrows():
                doc_data = row['data']
                doc_terms = doc_data.split()
                for term in doc_terms:
                    inv_idx[term][id] = inv_idx[term].get(id, 0) + 1
        return inv_idx
    
    def get_idf(self):
        idf, fname = None, './data/idf.csv'
        if os.path.exists(fname):
            idf = pd.read_csv(fname, index_col='term')
            idf = idf.to_dict()['idf']
        return idf
    
    def get_wv_model(self):
        pass
    
    def get_ranking(self, doc_scores, top_K = None):
        ranking = sorted(doc_scores.items(),
                    key=lambda item: item[1], reverse=True)
        if top_K == None:
            return ranking
        else:
            return ranking[:top_K]
        
    def retrieve(self, query):
        ranking = None
        if self.name=="bm25":
            ranking = self.run_bm25(query)
            
        return ranking

    def retrieve_with_qe(self, query):
        '''
        Run BM25 to get initial ranking 
        Assume top-M documents are relevant
        Do local query expansion
        Run IR method with expanded query
        '''            
        init_ranking = self.run_bm25(query, top_K=3)
        
        query = preprocess_text(query)
        print(f"Processed query: {query}")
        query_exp = queryexp(query=query, topkdocs=init_ranking, docs_tokenized=self.docs_tokenized)
        print(f"Expanded query: {query_exp}")
        
        # Run IR with expanded query
        retr = [self.collection.iloc[id]['filename'] for id, _ in init_ranking]
        print(f"Initial ranking: {retr}")
       
        ranking = None
        if self.name=="bm25":
            ranking = self.run_bm25(query_exp)
            
        return ranking
        
    def run_bm25(self, query, top_K=3):
        inv_idx = self.get_inv_idx()
        idf = self.get_idf()            
        bm25 = BM25(inverted_idx=inv_idx, docs_tokenized=self.docs_tokenized, num_docs=self.num_docs, idf=idf)
        
        if isinstance(query, str):
            query = preprocess_text(query)
            
        doc_scores = {}
        for id in range(self.num_docs):
            score = bm25.get_score(query, id)
            doc_scores[id] = score

        ranking = self.get_ranking(doc_scores=doc_scores, top_K=top_K)
        return ranking
    
    def run_vsm(self, query, top_K=3):
        if isinstance(query, str):
            query = preprocess_text(query)
        
        model = self.get_wv_model()
        all_doc_vec = None #TODO: write method to read file or vectorize
        query_vec = sent2vec(model.wv, query)
        
        doc_scores = {}
        for id in range(self.num_docs):
            doc_vec = all_doc_vec[id]
            score = cosinesimilarity(query_vec, doc_vec)
            doc_scores[id] = score

        ranking = self.get_ranking(doc_scores=doc_scores, top_K=top_K)
        return ranking
    

if __name__ == "__main__":
    method = input("Select IR method: ")
    
    doc_collection = pd.read_csv('./data/processed.csv')
    
    engine = Engine(collection=doc_collection, name=method)
        
    docs_tokenized = []
    for id, row in doc_collection.iterrows():
        doc_data = row['data']
        doc_terms = doc_data.split()
        docs_tokenized.append(doc_terms)

    eval = pd.read_csv('./data/evaluation.csv')[:1]
    eval['Documents'] = eval['Documents'].apply(lambda x: x.strip("[]").replace("'","").split(", "))
    rs = []
    times = []
    for _, row in eval.iterrows():
        q = row['Query']
      
        start = time.time()
        ranking = engine.retrieve(query=q)
        end = time.time()
        
        times.append(end-start)
        print(f"Time taken = {end-start}s")
        retr = [doc_collection.iloc[id]['filename'] for id, item in ranking]
        print("Retrieved:", retr)
        r = get_relevance_scores(retr, row['Documents'])
        print("Relevance: ", r)
        print(f"Avg. precision = {average_precision(r)}, reciprocal rank = {reciprocal_rank(r)}")
        rs.append(r)
        result, _ = ranking[0]
        top_doc = doc_collection.iloc[result]
        print(f"\nDocument {top_doc['filename']}")
        
        # get answer 
        qa = {'question': q,
              'context': top_doc['data']
              }
        start = time.time()
        ans = qa_pipeline(qa)
        end = time.time()
        print(f"Q - {q}\nA - {ans['answer']}\nTime taken = {end-start}")
        
        print()

    print(mean_average_precision(rs))
    print(mean_reciprocal_rank(rs))
    print(np.mean(times))