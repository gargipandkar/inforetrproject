'''
Neural Language Modeling

Given a document, generate queries 

Reference: https://github.com/castorini/docTTTTTquery
Using pretrained T5 instead of training an LSTM

When indexing, index top-k predicted queries for all documents
At runtime, check similarity between incoming query and indexed queries
To further reduce latency, index embedding of predicted queries 
'''
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from nltk import word_tokenize, sent_tokenize

torch.manual_seed(42)

import pandas as pd 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
model.to(device)

if __name__ == "__main__":
    docs = pd.read_csv('./data/documents.csv')
    pred_queries = {}
    for id, doc in docs.iterrows():
        # print(doc['title'])
        doc_text = doc['data']
        input_ids = tokenizer.encode(doc_text, truncation=True, return_tensors='pt').to(device)
        outputs = model.generate(
            input_ids=input_ids,
            max_length=25,
            repetition_penalty=2.5,
            do_sample=True,
            top_k=3,
            num_return_sequences=1)

        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_queries[id] = predicted
        # print(f'generated: {predicted}\n')

        import pickle
        fname = './data/nlm_queries.pickle'
        with open(fname, 'wb') as f:
            pickle.dump(pred_queries, f)