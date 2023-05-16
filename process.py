# Deep learning libraries 
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, Module, Linear, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Basic libraries 
import os
from os.path import join, exists
import pickle
import sys
import numpy as np
import logging
import warnings
import random as rd
import json
from argparse import ArgumentParser
from tqdm import tqdm
import math
import time
from datetime import datetime

prefix1 = "[persona]: "
prefix2 = "[query]: "
prefix3 = "generate a dialog response to [query] based on [persona]: " 

def read_dataset(path):
    query, response = [], []
    pre_state, cur_state = "dialog", "dialog"
    
    with open(path, "r", encoding="utf-8") as src:
        for line in src: 
            if "persona:" in line: 
                pre_state = cur_state
                cur_state = "persona"
            else:
                pre_state = cur_state
                cur_state = "dialog"

            if pre_state == "dialog" and cur_state == "persona":
                cur_text = ""

            if cur_state == "persona":
                cur_text += (line[21:].strip("\n") + " ")
            elif cur_state == "dialog":
                line = line[line.find(' '):]
                query.append(prefix3 + prefix1 + cur_text + prefix2 + line.split('\t')[0])
                response.append(line.split('\t')[1])
            else:
                raise (ValueError)
            
    return query, response


# return tokenized and write them into files 
def preprocess(train_path, val_path, save_path='../data/convai2_prepared/'):
    # load the data 
    train_query, train_response = read_dataset(train_path) 
    test_query, test_response = read_dataset(val_path)
    
    # load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # tokenize queries and save as dic
    train_query_tokenized = tokenizer(train_query, truncation=True, padding=True, max_length=96)
    train_query_tokenized = {key: val for key, val in train_query_tokenized.items()}
    
    test_query_tokenized = tokenizer(test_query, truncation=True, padding=True, max_length=96)
    test_query_tokenized = {key: val for key, val in test_query_tokenized.items()}
    
    # randomly select response from train dataset as a negative sample for each query
    train_random_res = [train_response[rd.randint(0, len(train_response) - 1)] for _ in train_response]
    
    assert len(train_query) == len(train_response)
    assert len(test_query) == len(test_response)
    
    # tokenize responses and save as dic
    train_response_tokenized = tokenizer(train_response, truncation=True, padding=True, max_length=32)
    train_response_tokenized = {key: val for key, val in train_response_tokenized.items()}
    
    test_response_tokenized = tokenizer(test_response, truncation=True, padding=True, max_length=32)
    test_response_tokenized = {key: val for key, val in test_response_tokenized.items()}
    
    train_random_res_tokenized = tokenizer(train_random_res, truncation=True, padding=True, max_length=32)
    train_random_res_tokenized = {key: val for key, val in train_random_res_tokenized.items()}
    
    # save_path
    print(f"Saving tokenized dict at {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # save the files 
    with open(save_path+'train_query.json','w') as train_query:
        print("Dump train_query")
        print(len(train_query_tokenized['input_ids']))
        json.dump(train_query_tokenized, train_query)
    with open(save_path+'test_query.json','w') as test_query:
        print("Dump test_query")
        print(len(test_query_tokenized['input_ids']))
        json.dump(test_query_tokenized, test_query)
   
    with open(save_path+'train_response.json','w') as train_response:
        print("Dump train_response")
        print(len(train_response_tokenized['input_ids']))
        json.dump(train_response_tokenized, train_response)
    with open(save_path+'test_response.json','w') as test_response:
        print("Dump test_response")
        print(len(test_response_tokenized['input_ids']))
        json.dump(test_response_tokenized, test_response)
    with open(save_path+'train_random_res.json','w') as train_random_res:
        print("Dump train_random_res")
        print(len(train_random_res_tokenized['input_ids']))
        json.dump(train_random_res_tokenized, train_random_res)
        
def load_dataset(path):

    with open(path + 'train_query.json') as train_query:
        print("Load train_query")
        tmp = train_query.readline()
        train_query_tokenized = json.loads(tmp)
    with open(path + 'test_query.json') as val_query:
        print("Load val_query")
        tmp = val_query.readline()
        val_query_tokenized = json.loads(tmp)

    with open(path + 'train_response.json') as train_response:
        print("Load train_response")
        tmp = train_response.readline()
        train_response_tokenized = json.loads(tmp)

    with open(path + 'test_response.json') as val_response:
        print("Load val_response")
        tmp = val_response.readline()
        val_response_tokenized = json.loads(tmp)
        
    with open(path + 'train_random_res.json') as train_random_res:
        print("Load train_random_res")
        tmp = train_random_res.readline()
        train_random_res_tokenized = json.loads(tmp)
        
        
    train_dataset = ConvAI2Dataset(train_query_tokenized, train_response_tokenized, train_random_res_tokenized)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=2)

    val_dataset = ConvAI2Dataset(val_query_tokenized, val_response_tokenized)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)

    return train_loader, val_loader


class ConvAI2Dataset(torch.utils.data.Dataset):
    def __init__(self, queries, labels, neg_samples = {}):
        self.queries = queries
        self.labels = labels
        if neg_samples:
            self.neg_samples = neg_samples
        else:
            self.neg_samples = {'input_ids': [], 'attention_mask': []}
     
        
    def __getitem__(self, idx):
        query = {
            key: torch.tensor(val[idx])
            for key, val in self.queries.items()
        }
        response = {
            key: torch.tensor(val[idx])
            for key, val in self.labels.items()
        }
        random_response = {
            key: torch.tensor(val[idx] if val else [])
            for key, val in self.neg_samples.items()
        }
        return {'query': query, 'response': response, 'random_response': random_response}

    def __len__(self):
        return len(self.labels['input_ids'])
