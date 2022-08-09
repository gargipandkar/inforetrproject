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

torch.manual_seed(42)

import pandas as pd 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
model.to(device)

# doc_text = "INSERT TEXT HERE"
# input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
# k = 5
# outputs = model.generate(
#     input_ids=input_ids,
#     max_length=32,
#     repetition_penalty=2.5,
#     do_sample=True,
#     top_k=k,
#     num_return_sequences=3)

# for i in range(3):
#     print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')

docs = pd.read_csv('./data/documents.csv')[:3]

for _, doc in docs.iterrows():
    doc_text = doc['data'][:256]
    print(doc['title'])
    input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
    k = 5
    outputs = model.generate(
        input_ids=input_ids,
        max_length=32,
        repetition_penalty=2.5,
        do_sample=True,
        top_k=k,
        num_return_sequences=3)

    for i in range(3):
        print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')
    print()