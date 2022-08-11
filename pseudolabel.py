import pandas as pd
from collections import defaultdict
import random
import json
import re

from preprocessing import preprocess_text
from bm25 import BM25

'''
Create pseudo-labelled dataset (query, positive sample, negative sample)

1. use page name or title as query
2. run BM25 to get positive and negative samples

'''

def clean_filename(filename):
    return re.sub(r'[0-9]+', '', filename)
    
if __name__=="__main__":
    doc_collection = pd.read_csv('./data/documents.csv')
    num_docs = len(doc_collection)
        
    ele_json_list = []
    query_set = set()
    
    # define M, N for top-M and last-N docs
    M = 3
    N = 10
        
    # set up BM25
    docs_tokenized = []
    inv_idx = defaultdict(dict)
    
    for id, row in doc_collection.iterrows():
        doc_data = row['data']
        doc_terms = preprocess_text(doc_data)
        docs_tokenized.append(doc_terms)
        for term in doc_terms:
            inv_idx[term][id] = inv_idx[term].get(id, 0) + 1

    bm25 = BM25(inverted_idx = inv_idx, docs_tokenized=docs_tokenized, num_docs=num_docs)
    
    count=0
    for doc_id, row in doc_collection.iterrows():
        ele_json = {}
        
        # create queries 
        queries = []
        # by default one query from title
        q_t = row["title"]
        queries.append(q_t)
        q_f = clean_filename(row["filename"])
        # if filename is different from title then another query from filename
        if preprocess_text(q_t)!=preprocess_text(q_f) and q_f not in query_set:
            query_set.add(q_f)
            queries.append(q_f)
        
        for q in queries:
            query = preprocess_text(q)
            pos_docs = []
            neg_docs = []
            
            # add document which query comes from as positive sample
            # pos_docs.append(doc_id)
            
            # run BM25 
            doc_scores = {}
            for d in doc_collection.index:
                score = bm25.get_score(query=query, doc_id=d)
                doc_scores[d] = score
            
            # Normalize BM25 scores
            total_scores = sum(doc_scores.values())
            for doc, score in doc_scores.items():
                if total_scores != 0.0:
                    doc_scores[doc] = round(score / total_scores, 5)

            ranking = bm25.get_ranking(doc_scores=doc_scores)
            
            # reduce since we already have one positive doc
            # pos_choices = [item[0] for item in ranking[:M]]
            # pos_choices = [ranking[:M]]
          
            # try:
            #     pos_choices.remove(doc_id)
            # except ValueError:
            #     pos_choices = pos_choices[:-1]
            pos_docs = ranking[:M]
           
            # randomize selection of negative docs
            # neg_choices = [item[0] for item in ranking[-2*N:]]
            neg_choices = ranking[-2*N:]
            neg_docs = random.choices(neg_choices, k=N)
            
            ele_json = {
                "query": q,
                "pos_docs": pos_docs,
                "neg_docs": neg_docs
            }
            ele_json_list.append(ele_json)
    
    labelled_dataset_json = {"data": ele_json_list}
    with open("./data/labelled.json", "w", encoding="utf-8") as fp:
        json.dump(labelled_dataset_json, fp, ensure_ascii=False)