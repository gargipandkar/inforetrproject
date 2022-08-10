'''
Using pretrained Dense Passage Retrieval encoders
'''

from sentence_transformers import SentenceTransformer, util
import torch 

import numpy as np 
import pandas as pd

if __name__=="__main__":
   
    encoder = SentenceTransformer('./data/semanticsearch')

    # docs = [
    #     "London [SEP] London is the capital and largest city of England and the United Kingdom.",
    #     "Paris [SEP] Paris is the capital and most populous city of France.",
    #     "Berlin [SEP] Berlin is the capital and largest city of Germany by both area and population."
    # ]
    
    # query = "What is the capital of England?"
    
    docs = pd.read_csv('./data/documents.csv')
    doc_embeddings = encoder.encode(docs['data'])
    
    import pickle
    fname = './data/ss_docs.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(doc_embeddings, f)
        
    query = "Where can I find good vegetarian food?"
    query_embedding = encoder.encode(query)

    scores = util.dot_score(query_embedding, doc_embeddings)
    _, topk = torch.topk(scores, k=3, sorted=True)
    topk = topk.flatten().tolist()
    print("Query:", query)
    print("Docs:", [docs.iloc[i]['filename'] for i in topk])
