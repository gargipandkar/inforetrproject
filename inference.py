from evaluate import Engine
from qa import qa_pipeline

import pandas as pd
import time

if __name__ =="__main__":
    print("Starting up...")
    doc_collection = pd.read_csv('./data/documents.csv')
    method = input("Select IR method: ")
    qe = False
    if method in ["bm25", "wordrep"]:
        qe = input("Query expansion (y/n): ")
        if qe.lower() == "y": qe = True
    
    engine = Engine(collection=doc_collection, method=method, qe=qe)
    
    while True:
        q = input("\nEnter query: ")
        start = time.time()
        ranking = engine.retrieve(query=q)
        end = time.time()
        
        result, _ = ranking[0]
        top_doc = doc_collection.iloc[result]
        print(f"\nDocument: {top_doc['filename']}\nFound at: {top_doc['uri']}")
        print(f"\nTime taken = {end-start}s")
        
        qa_input = {'question': q,
                    'context': top_doc['data']
                    }
        start = time.time()
        ans = qa_pipeline(qa_input)
        end = time.time()
        print(f"\nAnswer: {ans['answer']}\n\nTime taken = {end-start}")
