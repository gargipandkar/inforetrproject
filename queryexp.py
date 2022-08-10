from collections import defaultdict
from bm25 import BM25
import math
import time
import pandas as pd

from preprocessing import preprocess_text
from metrics import *
from qa import qa_pipeline
from collections import defaultdict
import pprint
from copy import deepcopy
from heapq import nlargest
from nltk.corpus import wordnet

def queryexp(query, topkdocs, docs_tokenized, topm=3):
    '''
    Returns:
        Expanded query
    '''
    

    # Get all unique terms from top k docs
    # Create co-occurrence matrix (association matrix?)
    # Populate matrix
    '''
    for uniqueword in topkdocsvocab
        for doc in topkdocs
            add to association matrix
        
    '''
    association_matrix = defaultdict(dict)
    unique_words = []
    for topk,_ in topkdocs:
        for uniqueword in docs_tokenized[topk]:
            unique_words.append(uniqueword)

    for i in unique_words:
        for j in unique_words:
            association_matrix[i][j] = association_matrix[i].get(j, 0)

    for i in set(unique_words):
        for j in set(unique_words):
            for k,_ in topkdocs:
                association_matrix[i][j] += (docs_tokenized[k].count(i) * docs_tokenized[k].count(j))

    # Normalize matrix (termi-termj frequency / (sum of termii-termjj diagonals - termi-termj frequency))
    normalized_matrix = deepcopy(association_matrix)
    for i, innerdict in association_matrix.items():
        for j, value in innerdict.items():
            normalized_matrix[i][j] = association_matrix[i][j] / (association_matrix[i][i] + association_matrix[j][j] - association_matrix[i][j])
    
    # pp = pprint.PrettyPrinter(depth=6)
    # pp.pprint(normalized_matrix)
    # get relevant words to query from association matrix
    '''
    expandedquery = query
    for word in query
        if word in matrix
            get top m words from matrix
            add to expanded query
        else 
            use wordnet function for synonym
    '''
    expandedquery = deepcopy(query)
    for word in query:
        if word in normalized_matrix:
            topm_words = nlargest(topm, normalized_matrix[word], key = normalized_matrix[word].get)
            expandedquery += topm_words
        else: 
            # use wordnet function for synonym
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    expandedquery.append(lemma.name()) 
    
    return list(set(expandedquery))

# query = ["Boy"]
# ranking = [(0,1),(1,3),(2,1)]
# docs_tokenized = [["I", "Love", "IR", "AI"], ["I", "Hate", "Math"], ["I", "Love", "SUTD"]]
# queryexp(query=query, topkdocs=ranking, docs_tokenized=docs_tokenized)

if __name__ == "__main__":
    doc_collection = pd.read_csv('./data/processed.csv')
    num_docs = len(doc_collection)
        
    docs_tokenized = []
    inv_idx = defaultdict(dict)
    
    for id, row in doc_collection.iterrows():
        doc_data = row['data']
        doc_terms = doc_data.split()
        docs_tokenized.append(doc_terms)
        for term in doc_terms:
            inv_idx[term][id] = inv_idx[term].get(id, 0) + 1
 
    bm25 = BM25(inverted_idx = inv_idx, docs_tokenized=docs_tokenized, num_docs=num_docs)
    
    eval = pd.read_csv('./data/evaluation.csv')[:1]
    eval['Documents'] = eval['Documents'].apply(lambda x: x.strip("[]").replace("'","").split(", "))
    rs = []
    times = []
    for _, row in eval.iterrows():
        q = row['Query']
        query = preprocess_text(q)
        print(f"Processed query: {query}")
        
        # Initial ranking 
        doc_scores = {}
        start = time.time()
        for id in doc_collection.index:
            score = bm25.get_score(query, id)
            doc_scores[id] = score
           
        ranking = bm25.get_ranking(doc_scores=doc_scores, top_K=5)
        query_exp = queryexp(query=query, topkdocs=ranking, docs_tokenized=docs_tokenized)
        end = time.time()
        ir_time = end-start
        print(f"Expanded query: {query_exp}")
        # Redo BM25 with expanded query on subset of documents
        retr = [doc_collection.iloc[id]['filename'] for id, _ in ranking]
        print(f"Initial ranking: {retr}")
       

        maxscore = -1*float("inf")
        result = None
        doc_scores = {}
        start = time.time()
        for id, _ in ranking:
            score = bm25.get_score(query, id)
            doc_scores[id] = score
            if score > maxscore:
                maxscore = score
                result = id
            
        ranking = bm25.get_ranking(doc_scores=doc_scores, top_K=3)
        retr = [doc_collection.iloc[id]['filename'] for id, item in ranking]
        end = time.time()
        ir_time += end-start
        times.append(ir_time)
        print(f"Time taken = {end-start}s")
        print("Retrieved:", retr)
        r = get_relevance_scores(retr, row['Documents'])
        print("Relevance: ", r)
        print(f"Avg. precision = {average_precision(r)}, reciprocal rank = {reciprocal_rank(r)}")
        rs.append(r)
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