import math

import json
import os

from preprocessing import *

'''
Calculate TF
Given a tokenized document (a list of words), return a dictionary
with key as a word and value as the number of times it occurs in the doc. For instance:
input: [ a , b , b , c , d ] 
output: { a : 1, b : 2, c : 1, d : 1}
'''
   
def tf_(doc):
    frequencies = {}
    for term in doc:
        if term in frequencies:
            frequencies[term]+=1
        else:
            frequencies[term]=1
            
    return frequencies

'''  
Calculate DF
Given a list of tokenized documents, return a dictionary with the number of documents (value) 
that a term (key) appears in. For instance:
input: [[ a , b , c ], [ b , c , d ], [ c , d , e ]]
output: { a : 1, b : 2, c : 3, d : 2, e : 1}
'''
   
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
IDF
input = { 'a': 1, 'b' : 2, 'c' : 3, 'd' : 2, 'e' : 1}
output =  { 'a' : 1.1, 'b' : 0.41, 'c' : 0.0, 'd' : 0.41, 'e' : 1.1}
'''
def idf_(df, corpus_size): 
    idf = {}
    for term, freq in df.items():
        idf[term] = round(math.log((corpus_size) / (freq)),2) 
    return idf

'''
BM25
Score a document given a tokenized query, doc, and docs. 
Input:
query= [ b , c , e ] 
doc= [ b , c , d ]
docs= [[ a , b , c ], [ b , c , d ], [ c , d , e ]]
'''
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

if __name__=="__main__":
    query = input("Type in some query:")
    query = preprocess(query)
    docs = []
    filelist = os.listdir("./data/documents")
    for filename in filelist:
        with open("./data/documents/"+filename, "r", encoding="utf-8") as f:
            tmp = json.load(f)['data']
            docs.append(preprocess(" ".join(tmp.split()[:50])))

    maxscore = -1*float("inf")
    result = None
    num_docs = len(filelist)
    for i in range(num_docs):
        score = _score(query, docs[i], docs)
        print(score, end=' ')
        if score>maxscore:
            maxscore = score
            result = i
        
    print(f"\nDocument {filelist[result]}")
    