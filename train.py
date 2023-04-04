#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['http_proxy'] = "http://proxygate2.ctripcorp.com:8080"
os.environ['https_proxy'] = "http://proxygate2.ctripcorp.com:8080"


# In[ ]:


import pickle
import importlib

from sklearn.model_selection import train_test_split

import pandas as pd
from datasets import Dataset

import torch
from torch.utils.data import DataLoader, distributed
from torch import optim, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from transformers import AutoTokenizer
from tqdm.auto import tqdm

from utils.checkpoint import checkpoint
from utils.makeModel import make_model
from utils.run import Fit
from Transformer.Module import WarmUpOpt, LabelSmoothing


# In[ ]:


class CFG:
    source_max_segment = 40
    source_max_length = 50
    target_max_length = 300
    pretrained_model_name_or_path = 'facebook/bart-base'
    gradient_clipper = 5
    smoothing = 0.1
    checkpoint_path = 'model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 512
    num_head = 8
    num_layer_encoder = 6
    num_layer_decoder = 6
    num_layer_graph_encoder = 6
    d_ff = 2048
    dropout_embed = 0.1
    dropout_sublayer = 0.1
    learning_rate = 1e-04
    beta_1 = 0.9
    beta_2 = 0.98
    eps = 1e-09
    weight_decay = 1e-05
    warmup_steps = 4000
    min_learning_rate = 0
    factor = 1
    batch_size = 16
    epoch = 100

cfg = CFG()


# In[ ]:


train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/valid.csv")
test_df = pd.read_csv("data/test.csv")


# In[ ]:


# train_df = train_df[:1000]
# valid_df = valid_df[:200]
# test_df = test_df[:200]


# In[ ]:


all_df = pd.concat([train_df, valid_df, test_df])


# In[ ]:


train_df, test_df = train_test_split(all_df, test_size=0.2, random_state=42)


# In[ ]:


train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# In[ ]:


train_df


# In[ ]:





# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)


# In[ ]:


def get_dataset(df, cfg):
    D = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        node = eval(row['node'])
        edge = eval(row['edge'])
        docstring = row['docstring']
        source_input_ids = []
        source_attention_mask = []
        target_input_ids = []
        target_attention_mask = []
        for segment in node[:cfg.source_max_segment]:
            segment_tokens = tokenizer(segment, truncation=True, padding="max_length", max_length=cfg.source_max_length)
            source_input_ids.append(segment_tokens['input_ids'])
            source_attention_mask.append(segment_tokens['attention_mask'])
        if len(source_input_ids) < cfg.source_max_segment:
            for i in range(len(source_input_ids), cfg.source_max_segment):
                segment_tokens = tokenizer("", truncation=True, padding="max_length", max_length=cfg.source_max_length)
                source_input_ids.append(segment_tokens['input_ids'])
                source_attention_mask.append(segment_tokens['attention_mask'])
        graph = [[0] * cfg.source_max_segment for _ in range(cfg.source_max_segment)]

        for l, r in edge:
            if l < cfg.source_max_segment and r < cfg.source_max_segment:
                graph[l][r] = 1

        segment_tokens = tokenizer(docstring, truncation=True, padding="max_length", max_length=cfg.target_max_length)
        D.append({
            "source_input_ids": torch.LongTensor(source_input_ids),
            "graph": torch.LongTensor(graph),
            "target_input_ids": torch.LongTensor(segment_tokens['input_ids']),
        })

    dataset = Dataset.from_list(D)
    dataset.set_format(type="torch")

    return dataset


# In[ ]:


cfg.PAD_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
cfg.BOS_index = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
cfg.EOS_index = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

cfg.vocab_size = tokenizer.vocab_size

criterion = LabelSmoothing(smoothing=cfg.smoothing, ignore_index=cfg.PAD_index)

check_point = None
check_point = checkpoint(save_path=cfg.checkpoint_path)
params = {}
params['checkpoint'] = check_point
params['criterion'] = criterion

model_state_dict = None
optim_state_dict = None
start_epoch = 0
model_state_dict, optim_state_dict, start_epoch = params['checkpoint'].restore()

model = make_model(cfg)

optimizer = WarmUpOpt(
    optimizer=optim.Adam(params=model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta_1, cfg.beta_2), eps=cfg.eps,
                         weight_decay=cfg.weight_decay),
    d_model=cfg.embedding_dim,
    warmup_steps=cfg.warmup_steps,
    min_learning_rate=cfg.min_learning_rate,
    factor=cfg.factor,
    state_dict=optim_state_dict)

if model_state_dict is not None:
    model.load_state_dict(model_state_dict)
params['model'] = model
params['optimizer'] = optimizer


# In[ ]:


train_dataset = get_dataset(train_df, cfg)
test_dataset = get_dataset(test_df, cfg)

for d in train_dataset:
    print(d)
    break


# In[ ]:


train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

for d in train_loader:
    print(d['source_input_ids'].shape)
    print(d['graph'].shape)
    print(d['target_input_ids'].shape)
    break


# In[ ]:


params['train_data'] = train_loader

fit = Fit(cfg, params)


# In[ ]:


fit(start_epoch=start_epoch)


# In[ ]:





# In[ ]:





# In[ ]:




