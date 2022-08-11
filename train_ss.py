'''
Training script for semantic search model
'''

from sentence_transformers import SentenceTransformer, util, InputExample, losses
import torch 
from torch.utils.data import DataLoader
from torch import nn, Tensor
from typing import Iterable, Dict

import numpy as np 
import pandas as pd
import json

def getTrainExamples(filename):
    f = open(filename)
    data = json.load(f)
    return data

def train_ss(model, data, df, epochs= 100, output_path = "./outputs", checkpoint_path="./checkpoint", checkpoint_save_steps=5,):
    
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
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, output_path = output_path, checkpoint_path=checkpoint_path, checkpoint_save_steps=checkpoint_save_steps)    

if __name__=="__main__":

    #Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer('msmarco-roberta-base-v3')
    #Define your train examples. 
    data = getTrainExamples("./data/labelled.json")
    df = pd.read_csv("./data/documents.csv")

    train_ss(model, data, df)