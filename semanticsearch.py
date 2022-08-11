'''
Using pretrained Dense Passage Retrieval encoders
'''

from sentence_transformers import SentenceTransformer, util
import torch 

import numpy as np 
import pandas as pd

if __name__=="__main__":
   
    encoder = SentenceTransformer('./data/checkpoint')
    
    docs = pd.read_csv('./data/documents.csv')
    doc_embeddings = encoder.encode(docs['data'])
    
    import pickle
    fname = './data/ss_docs.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(doc_embeddings, f)
        
    # Test example
    query = "Is there a safari in Singapore?"
    query_embedding = encoder.encode(query)

    scores = util.dot_score(query_embedding, doc_embeddings)
    _, topk = torch.topk(scores, k=3, sorted=True)
    topk = topk.flatten().tolist()
    print("Query:", query)
    print("Docs:", [docs.iloc[i]['filename'] for i in topk])
