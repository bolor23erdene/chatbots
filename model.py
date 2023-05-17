import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, Module, Linear, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
import random as rd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import math
import time
from datetime import datetime
import os
from os.path import join, exists
import transformers
import pickle
import sys
import numpy as np
import logging
import warnings

class JointT5Model(Module):
    # load seq2seq T5 model 
    def __init__(self, model_path, output_hidden_states=True, hidden_size=768):
        super(JointT5Model, self).__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(model_path, output_hidden_states=output_hidden_states)
        self.hidden_size = hidden_size
        self.ranker = Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, labels, neg_samples, label_length, neg_length):
        # implement generation loss 
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        generation_loss = outputs.loss

        # implement ranking loss 
        ranking_loss = 0

        loss = generation_loss + ranking_loss
        return {'logits': outputs.logits, 'loss': loss}
    
    def evaluate(self, input_ids, attention_mask, labels):
        return self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels)