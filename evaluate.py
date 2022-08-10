import os
import time

import pickle
import pandas as pd
from collections import defaultdict

from gensim.models import FastText
from sentence_transformers import SentenceTransformer, util
import torch 

from preprocessing import preprocess_text
from metrics import *
from bm25 import BM25
from vsm import *
from queryexp import *
from qa import qa_pipeline

class Engine:
    def __init__(self, collection, method="bm25", qe=False):
        self.collection = collection
        self.name = method
        self.qe = qe
        self.num_docs = len(collection)
        self.docs_tokenized = self.tokenize(collection)
        if qe:
            self.name += "+qe"
        
        if method=="bm25" or qe==True:
            inv_idx = self.get_inv_idx()
            idf = self.get_idf()            
            self.bm25 = BM25(inverted_idx=inv_idx, 
                             docs_tokenized=self.docs_tokenized, 
                             num_docs=self.num_docs, 
                             idf=idf)
        if method=="vsm":
            self.model = self.get_wv_model()
            self.all_doc_vec = self.get_wv_docs()
        if method=="nn":
            self.query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
            self.doc_embeddings = self.get_nn_docs()
        if method=="ss":
            self.encoder = SentenceTransformer('./data/semanticsearch')
            self.doc_embeddings = self.get_ss_docs()
        if method=="nlm":
            self.model = self.get_wv_model()
            self.pred_queries = self.get_nlm_queries()
            
    def tokenize(self, collection):
        if collection is None:
            collection = self.collection
        docs_tokenized = []
        for id, row in self.collection.iterrows():
            doc_data = row['data']
            doc_terms = preprocess_text(doc_data)
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
        fname = "./data/fasttext.model"
        model = FastText.load(fname)
        return model
            
    def get_wv_docs(self):
        all_doc_vec, fname = None, './data/vsm_docs.pickle'
        with open(fname, 'rb') as f:
            all_doc_vec  = pickle.load(f)
        return all_doc_vec
    
    def get_nn_docs(self):
        doc_embeddings, fname = None, './data/nn_docs.pickle'
        with open(fname, 'rb') as f:
            doc_embeddings = pickle.load(f)
        return doc_embeddings
    
    def get_ss_docs(self):
        doc_embeddings, fname = None, './data/ss_docs.pickle'
        with open(fname, 'rb') as f:
            doc_embeddings = pickle.load(f)
        return doc_embeddings
    
    def get_nlm_queries(self):
        preds, fname = None, './data/nlm_queries.pickle'
        with open(fname, 'rb') as f:
            preds = pickle.load(f)
        return preds
    
    def get_ranking(self, doc_scores, top_K = None):
        ranking = sorted(doc_scores.items(),
                    key=lambda item: item[1], reverse=True)
        if top_K == None:
            return ranking
        else:
            return ranking[:top_K]
        
    def retrieve(self, query):
        if self.qe:
            query = self.get_expanded_query(query)
        ranking = None
        if "bm25" in self.name:
            ranking = self.run_bm25(query)
        elif "vsm" in self.name:
            ranking = self.run_vsm(query)
        elif "nn" in self.name:
            ranking = self.run_nn_encoder(query)
        elif "ss" in self.name:
            ranking = self.run_semantic_search(query)
        elif "nlm" in self.name:
            ranking = self.run_nlm(query)
        return ranking

    def get_expanded_query(self, query):
        '''
        Run BM25 to get initial ranking 
        Assume top-M documents are relevant
        Do local query expansion
        '''
        init_ranking = self.run_bm25(query, top_K=3)
        query = preprocess_text(query)
        query_exp = queryexp(query=query, topkdocs=init_ranking, docs_tokenized=self.docs_tokenized)
        return query_exp
        
    def run_bm25(self, query, top_K=3):
        if isinstance(query, str):
            query = preprocess_text(query)
             
        doc_scores = {}
        for id in range(self.num_docs):
            score = self.bm25.get_score(query, id)
            doc_scores[id] = score
        ranking = self.get_ranking(doc_scores=doc_scores, top_K=top_K)
        return ranking
    
    def run_vsm(self, query, top_K=3):
        if isinstance(query, str):
            query = preprocess_text(query)
        
        query_vec = sent2vec(self.model.wv, query)
        
        doc_scores = {}
        for id in range(self.num_docs):
            doc_vec = self.all_doc_vec[id]
            score = cosinesimilarity(query_vec, doc_vec)
            doc_scores[id] = score
        ranking = self.get_ranking(doc_scores=doc_scores, top_K=top_K)
        return ranking
    
    def run_nn_encoder(self, query):        
        query_embedding = self.query_encoder.encode(query)

        scores = util.dot_score(query_embedding, self.doc_embeddings)
        values, indices = torch.topk(scores, k=3, sorted=True)
        ranking =  list(zip(indices.flatten().tolist(), values.flatten().tolist()))
        return ranking
    
    def run_semantic_search(self, query):
        query_embedding = self.encoder.encode(query)

        scores = util.cos_sim(query_embedding, self.doc_embeddings)
        values, indices = torch.topk(scores, k=3, sorted=True)
        ranking =  list(zip(indices.flatten().tolist(), values.flatten().tolist()))
        return ranking
    
    def run_nlm(self, query):
        if isinstance(query, str):
            query = preprocess_text(query)
        
        query_vec = sent2vec(self.model.wv, query)
        
        doc_scores = {}
        for id in range(self.num_docs):
            pred_query = self.pred_queries[id]
            pred_query = preprocess_text(pred_query)
            pred_query_vec = sent2vec(self.model.wv, pred_query)
            score = cosinesimilarity(query_vec, pred_query_vec)
            doc_scores[id] = score
        ranking = self.get_ranking(doc_scores=doc_scores, top_K=3)
        return ranking
    
if __name__ == "__main__":
   
    doc_collection = pd.read_csv('./data/documents.csv')
    eval = pd.read_csv('./data/newevaluation.csv')
    eval['Documents'] = eval['Documents'].apply(lambda x: x.strip("[]").replace("'","").split(", "))
    
    # params = [("bm25", False), ("bm25", True), ("vsm", False), ("vsm", True), ("nn", False)]
    # params = [("bm25", False),  ("vsm", False), ("nn", False)]
    # params = [("bm25", True), ("vsm", True)]
    # params = [("nlm", False)]
    params = [("ss", False)]
    metrics = {}
    
    for param in params:
        method, qe = param
        engine = Engine(collection=doc_collection, method=method, qe=qe)

        eval[engine.name] = None
        eval[engine.name+'_score'] = None
        rs = []
        times = []
        for idx, row in eval.iterrows():
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
            print(f"Document: {top_doc['filename']}")
            
            # get answer
            qa = {'question': q,
                  'context': top_doc['data']
                  }
            start = time.time()
            ans = qa_pipeline(qa)
            end = time.time()
            print(f"Q - {q}\nA - {ans['answer']}\nTime taken = {end-start}\n")
            
            eval.at[idx, engine.name+'_doc'] = top_doc['filename']
            eval.at[idx, engine.name+'_ans'] = ans['answer']
            if r[0] == 0:
                eval.at[idx, engine.name+'_score'] = 0
        
        print(f"\n{engine.name}")
        print(mean_average_precision(rs))
        print(mean_reciprocal_rank(rs))
        print(np.mean(times))
        
        metrics_dict = {"mAP": mean_average_precision(rs),
                        "MRR": mean_reciprocal_rank(rs),
                        "avgTime": np.mean(times)
                        }
        metrics[engine.name] = metrics_dict
        
    # Save metrics and answers
    pd.DataFrame.from_dict(metrics).to_csv('./data/metrics.csv', index=True, header=True)
    eval.to_csv('./data/answers.csv', index=False)
