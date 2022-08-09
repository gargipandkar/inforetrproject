'''
Using pretrained Dense Passage Retrieval encoders
'''

from sentence_transformers import SentenceTransformer, util, InputExample, losses
import torch 
from torch.utils.data import DataLoader
from torch import nn, Tensor
from typing import Iterable, Dict

import numpy as np 
import pandas as pd
import json

# class DPR(nn.Module):
    
#     def __init__(self, question_model: SentenceTransformer, 
#                  context_model: SentenceTransformer):
#         super(DPR, self).__init__()

#         self.question_model = question_model
#         self.context_model = context_model

#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         embeddings = [self.question_model.encode(sentence_features[0]),self.context_model.encode(sentence_features[1])]
#         output = torch.matmal(embeddings[0], embeddings[1])
#         return output

def getTrainExamples(filename):
    f = open(filename)
    data = json.load(f)
    return data

if __name__=="__main__":
    
    # docs = [
    #     "London [SEP] London is the capital and largest city of England and the United Kingdom.",
    #     "Paris [SEP] Paris is the capital and most populous city of France.",
    #     "Berlin [SEP] Berlin is the capital and largest city of Germany by both area and population."
    # ]
    
    # query1 = "What is the capital of England?"
    # query2 = "What is the capital of Singapore?"

    # query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
    # doc_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    
    # # dpr_model = DPR(query_encoder, doc_encoder)
    # em1 = query_encoder.encode(query1)
    # em2 = doc_encoder.encode(query2)
    # print(em1.shape, em2.shape)

    # output = torch.dot(Tensor(em1),Tensor(em2))
    # print(output)

    #Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer('msmarco-roberta-base-v3')

    #Define your train examples. 
    data = getTrainExamples("./data/labelled.json")
    df = pd.read_csv("./data/documents.csv")
    
    train_examples = []
    for item in data["data"]:
        qry = item["query"]
        for pos_doc in item["pos_docs"]:
            doc_id = pos_doc[0]
            # score = pos_doc[1]
            inp_example = InputExample(texts=[qry,df["data"][doc_id]], label=1.0)
            train_examples.append(inp_example)

        for neg_doc in item["neg_docs"]:
            doc_id = neg_doc[0]
            # score = neg_doc[1]
            inp_example = InputExample(texts=[qry,df["data"][doc_id]], label=0.0)
            train_examples.append(inp_example)

    #Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    #Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=100, output_path = "./outputs", checkpoint_path="./checkpoints", checkpoint_save_steps=5)