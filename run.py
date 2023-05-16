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
from process import preprocess, load_dataset

# goal is to create dataset 
#query:  ["generate a dialog response to [query] based on [persona]: [persona]: i like to remodel homes. i like to go hunting. i like to shoot a bow. my favorite holiday is halloween. [query]:  hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape ."]
#response:  ['you must be very fast . hunting is one of my favorite hobbies .']
train_path, test_path = "./convai2/train_other_revised_no_cands.txt", "./convai2/valid_self_original_no_cands.txt"
preprocess(train_path, test_path)

data_path = './convai2_tokenized'
load_dataset(data_path)