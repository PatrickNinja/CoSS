# -*- coding: utf-8 -*-

import pandas as pd
import re
import json
from pathlib import Path
from typing import List, Optional
from tqdm.auto import tqdm
import json
import random
import json

def split_train_test_valid(df_flow):

    # define the ratios 8:1:1
    train_len = int(len(df_flow) * 0.8)
    test_len = int(len(df_flow) * 0.1)

    # split the dataframe
    idx = list(df_flow.index)
    random.shuffle(idx)  
    df_trn = df_flow.loc[idx[:train_len]]
    df_tst = df_flow.loc[idx[train_len:train_len+test_len]]
    df_val = df_flow.loc[idx[train_len+test_len:]] 

    return df_trn, df_tst, df_val





if __name__ == "__main__":

    df_trn, df_tst, df_val = split_train_test_valid(df_sl)
    df_trn['code_tokens'] = df_trn.mthd.apply(lambda x: x.split())
    df_trn['docstring_tokens'] = df_trn.cmt.apply(lambda x: x.split())
    with open('processed_data/solidity/train.jsonl','w') as f:
        for _, row in df_trn.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    df_val['code_tokens'] = df_val.mthd.apply(lambda x: x.split())
    df_val['docstring_tokens'] = df_val.cmt.apply(lambda x: x.split())
    with open('processed_data/solidity/valid.jsonl','w') as f:
        for _, row in df_val.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    df_tst['code_tokens'] = df_tst.mthd.apply(lambda x: x.split())
    df_tst['docstring_tokens'] = df_tst.cmt.apply(lambda x: x.split())
    with open('processed_data/solidity/test.jsonl','w') as f:
        for _, row in df_tst.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    df_val.to_csv('processed_data/solidity/valid.csv', index = False)