import math
import os
import time

import json
import pandas as pd
from collections import defaultdict

from preprocessing import preprocess_text
from metrics import *
from qa import qa_pipeline


'''
Calculate TF
Given a tokenized document (a list of words), return a dictionary
with key as a word and value as the number of times it occurs in the doc. For instance:
input: [ a , b , b , c , d ] 
output: { a : 1, b : 2, c : 1, d : 1}

def tf_(doc):
    frequencies = {}
    for term in doc:
        if term in frequencies:
            frequencies[term]+=1
        else:
            frequencies[term]=1
            
    return frequencies
'''
   
'''
Calculate TF
Given an inverted index and a filename/docID, return a dictionary
with key as a word and value as the number of times it occurs in the doc. For instance:
inputs: {word1 : {doc1: 1 , doc2: 4 ...}, ... } and doc1
output: { word1 : 1, word2 : 2, word3 : 1, word4 : 1}
'''

def tf_(inv_idx, filename):
    frequencies = {}
    for term in inv_idx:
        if filename in inv_idx[term].keys():
            frequencies[term] = inv_idx[term][filename]
    return frequencies

'''
Calculate TF
Given an inverted index and a filename/docID, return a dictionary
with key as a word and value as the number of times it occurs in the doc. For instance:
inputs: {word1 : {doc1: 1 , doc2: 4 ...}, ... } and doc1
output: { word1 : 1, word2 : 2, word3 : 1, word4 : 1}
'''


def tf_(inv_idx, filename):
    frequencies = {}
    for term in inv_idx:
        if filename in inv_idx[term].keys():
            frequencies[term] = inv_idx[term][filename]
    return frequencies


'''  
Calculate DF
Given a list of tokenized documents, return a dictionary with the number of documents (value) 
that a term (key) appears in. For instance:
input: [[ a , b , c ], [ b , c , d ], [ c , d , e ]]
output: { a : 1, b : 2, c : 3, d : 2, e : 1}

def df_(docs):
    df = {}
    for doc in docs:
        for term in doc:
            if term in df:
                df[term]+=1
            else:
                df[term]=1
    return df
'''

'''  
Calculate DF
Given an inverted index, return a dictionary with the number of documents (value) 
that a term (key) appears in. For instance:
input: {word1 : {doc1: 1 , doc2: 4 ...}, ... }
output: { word1 : 1, word2 : 3, word3 : 4, word4 : 2}
'''
def df_(inv_idx):
    df = {}
    for term in inv_idx:
        df[term] = len(inv_idx[term]) 
    return df
 
'''
IDF
input = { 'a': 1, 'b' : 2, 'c' : 3, 'd' : 2, 'e' : 1}
output =  { 'a' : 1.1, 'b' : 0.41, 'c' : 0.0, 'd' : 0.41, 'e' : 1.1}
'''
def idf_(df, corpus_size):
    idf = {}
    for term, freq in df.items():
        idf[term] = round(math.log((corpus_size) / (freq)), 2)
    return idf


'''
BM25
Score a document given a tokenized query, doc, and docs. 
Input:
query= [ b , c , e ] 
doc= [ b , c , d ]
docs= [[ a , b , c ], [ b , c , d ], [ c , d , e ]]

def _score(query, doc, docs, k1=1.5, b=0.75): 
    score = 0.0
    tf = tf_(doc) 
    df = df_(docs)
    idf = idf_(df, len(docs))
    doc_len = len(doc)
    avg_doc_len = sum([len(d) for d in docs])/len(docs)	# calculate average document length
    for term in query:
        if term not in tf.keys(): 
            continue
        
        c_t = idf[term]
        num = (k1+1)*tf[term]
        denom = tf[term] + k1*((1-b) + b*(doc_len/avg_doc_len))
        doc_wt = num/denom
        score += c_t * doc_wt
            
    return score
'''

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
def _score(query, doc, avg_doc_len, inv_idx, filename, df, num_docs, k1=1.5, b=0.75):
    score = 0.0
    tf = tf_(inv_idx, filename)
    idf = idf_(df, num_docs)
    doc_len = len(doc)
    for term in query:
        if term not in tf.keys():
            continue

        c_t = idf[term]
        num = (k1+1)*tf[term]
        denom = tf[term] + k1*((1-b) + b*(doc_len/avg_doc_len))
        doc_wt = num/denom
        score += c_t * doc_wt

    return score

if __name__ == "__main__":
    docs = []
    inv_idx = defaultdict(dict)
    filelist = os.listdir("./data/documents")
    num_docs = 0
    for filename in filelist:
       
        with open("./data/documents/"+filename, "r", encoding="utf-8") as f:
            if filename[-4:] == "json":
                num_docs += 1
                tmp = json.load(f)['data']
                doc_terms = preprocess_text(tmp)
                docs.append(doc_terms)
                for term in doc_terms:
                    doc_id = filename[:-5]
                    inv_idx[term][doc_id] = inv_idx[term].get(doc_id, 0) + 1

    df = df_(inv_idx)
    avg_doc_len = sum([len(d) for d in docs])/num_docs
    
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
        for filename in filelist:
            if filename[-4:] == "json":
                score = _score(query, docs[i], avg_doc_len,
                            inv_idx, filename[:-5], df, num_docs)
                doc_scores[filename] = score
                if score > maxscore:
                    maxscore = score
                    result = i
                i += 1

        ranking = sorted(doc_scores.items(),
                    key=lambda item: item[1], reverse=True)
        try:
            stop_idx = [item[1] for item in ranking].index(0)
        except ValueError:
            stop_idx = len(ranking)
        stop_idx = min(stop_idx, 3)
        retr = [item[0].split('.')[0] for item in ranking[:stop_idx]]
        end = time.time()
        times.append(end-start)
        print(f"Time taken = {end-start}s")
        print("Retrieved:", retr)
        r = get_relevance_scores(retr, row['Documents'])
        print("Relevance: ", r)
        print(f"Avg. precision = {average_precision(r)}, reciprocal rank = {reciprocal_rank(r)}")
        rs.append(r)
        print(f"\nDocument {filelist[result]}")
        
        # get answer
        with open("./data/documents/"+filelist[result], "r", encoding="utf-8") as f:
            tmp = json.load(f)['data']
        qa = {'question': q,
              'context': tmp
              }
        start = time.time()
        ans = qa_pipeline(qa)
        end = time.time()
        print(f"Q - {q}\nA - {ans['answer']}\nTime taken = {end-start}")
        
        print()

    print(mean_average_precision(rs))
    print(mean_reciprocal_rank(rs))
    print(np.mean(times))
