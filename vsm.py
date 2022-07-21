import math
import os
import time

import json
import pandas as pd
from collections import defaultdict
import numpy as np

from preprocessing import preprocess_text
from metrics import *
from qa import qa_pipeline

from gensim.models import Word2Vec, Doc2Vec, FastText
import gensim.downloader
from tqdm import tqdm

def sent2vec(glove_vectors, sentence):
    '''
    Args:
        A list of words
    Returns:
        A sentence vector by concat/average/sum over word vectors
    '''
    sent_vec = None
    unk_vector = np.zeros(50)
    for word in sentence:
        word_vec = glove_vectors[word]
        if sent_vec is None:
            sent_vec = word_vec
        else:
            sent_vec = np.sum([sent_vec,word_vec])
        
    # TODO: we can try weighted average
    sent_len = len(sent_vec)
    sent_vec = sent_vec/sent_len
    return sent_vec

def cosinesimilarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

if __name__ == "__main__":
    doc_collection = pd.read_csv('./data/documents.csv')
    num_docs = len(doc_collection)

    # Use pre-trained Word2vec model
    print(list(gensim.downloader.info()['models'].keys()))
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    # TODO: Train FastText model to be able to handle unknown words
    # https://radimrehurek.com/gensim/models/fasttext.html

    # Tokenize and preprocess documents - TODO: save to dataframe
    docs_tokenized = []
    for id, row in tqdm(doc_collection.iterrows()):
        doc_data = row['data']
        doc_terms = preprocess_text(doc_data)
        docs_tokenized.append(doc_terms)
    
    # Change to vector - need to add padding (?)
    all_doc_vec = []
    for doc in docs_tokenized:
        sent_vec = sent2vec(glove_vectors, doc)
        print(sent_vec)
        print(sent_vec.shape)
        all_doc_vec.append(sent2vec(glove_vectors, doc))

    eval = pd.read_csv('./data/evaluation.csv')
    eval['Documents'] = eval['Documents'].apply(lambda x: x.strip("[]").replace("'","").split(", "))
    rs = []
    times = []
    for _, row in eval.iterrows():
        q = row['Query']
        query = preprocess_text(q)
        print(f"Processed query: {query}")

        query_vec = sent2vec(glove_vectors, query)
        
        maxscore = -1*float("inf")
        result = None

        doc_scores = {}
        start = time.time()
        for i, doc_vec in enumerate(all_doc_vec):
            score = cosinesimilarity(query_vec, doc_vec)
            doc_scores[i] = score
            if score > maxscore:
                maxscore = score
                result = i

        ranking = sorted(doc_scores.items(),
                    key=lambda item: item[1], reverse=True)
        try:
            stop_idx = [item[1] for item in ranking].index(0)
        except ValueError:
            stop_idx = len(ranking)
        stop_idx = min(stop_idx, 3)
        retr = [doc_collection.iloc[item[0]]['filename'] for item in ranking[:stop_idx]]
        end = time.time()
        times.append(end-start)
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