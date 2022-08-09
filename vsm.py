import math
import os
import time

import json
import pandas as pd
from collections import defaultdict, Counter
import numpy as np

from preprocessing import preprocess_text
from metrics import *
from qa import qa_pipeline

from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.models.fasttext import load_facebook_vectors, load_facebook_model
import gensim.downloader
from gensim.test.utils import datapath
from tqdm import tqdm
import nltk

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

def cosinesimilarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
if __name__ == "__main__":
    doc_collection = pd.read_csv('./data/processed.csv')
    num_docs = len(doc_collection)

    # Use pre-trained Word2vec model
    # Word2Vec:
    # print(list(gensim.downloader.info()['models'].keys()))
    # model = gensim.downloader.load('glove-wiki-gigaword-50')
    # FastText:
    # FastText model is able to handle unknown words
    # Use FastText: https://radimrehurek.com/gensim/models/fasttext.html
    # cap_path = datapath("crime-and-punishment.bin")
    # model = load_facebook_model(cap_path)

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
    fname = './data/vsm_docs.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(doc_vec_map, f)

    # eval = pd.read_csv('./data/evaluation.csv')[:1]
    # eval['Documents'] = eval['Documents'].apply(lambda x: x.strip("[]").replace("'","").split(", "))
    # rs = []
    # times = []
    # for _, row in eval.iterrows():
    #     q = row['Query']
    #     query = preprocess_text(q)
    #     print(f"Processed query: {query}")

    #     query_vec = sent2vec(model.wv, query)
        
    #     maxscore = -1*float("inf")
    #     result = None

    #     doc_scores = {}
    #     start = time.time()
    #     for i, doc_vec in enumerate(all_doc_vec):
    #         score = cosinesimilarity(query_vec, doc_vec)
    #         doc_scores[i] = score
    #         if score > maxscore:
    #             maxscore = score
    #             result = i

    #     ranking = sorted(doc_scores.items(),
    #                 key=lambda item: item[1], reverse=True)
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