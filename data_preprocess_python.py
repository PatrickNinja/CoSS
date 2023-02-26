# -*- coding: utf-8 -*-


import numpy as np

from collections import Counter
from statistics import mean, median, stdev
from transformers import AutoTokenizer
import pandas as pd
import re
import json
from pathlib import Path
from typing import List, Optional
from tqdm.auto import tqdm
import json
import random

if __name__ == "__main__":

  with open('./dataset/python_dataset/data_ps.all.train', errors='ignore') as f:
      lines_mthd_trn = f.readlines()
  with open('./dataset/python_dataset/data_ps.descriptions.train', errors='ignore') as f:
      lines_cmt_trn = f.readlines()

  with open('./dataset/python_dataset/data_ps.all.test', errors='ignore') as f:
      lines_mthd_tst = f.readlines()
  with open('./dataset/python_dataset/data_ps.descriptions.test', errors='ignore') as f:
      lines_cmt_tst = f.readlines()

  with open('./dataset/python_dataset/data_ps.all.valid', errors='ignore') as f:
      lines_mthd_val = f.readlines()
  with open('./dataset/python_dataset/data_ps.descriptions.valid', errors='ignore') as f:
      lines_cmt_val = f.readlines()

  trn = {"mthd":lines_mthd_trn, "cmt":lines_cmt_trn}
  tst = {"mthd":lines_mthd_tst, "cmt":lines_cmt_tst}
  val = {"mthd":lines_mthd_val, "cmt":lines_cmt_val}

  df_trn = pd.DataFrame(trn) 
  df_tst = pd.DataFrame(tst) 
  df_val = pd.DataFrame(val) 

  df_trn['code_tokens'] = df_trn.mthd.apply(lambda x: x.split())
  df_trn['docstring_tokens'] = df_trn.cmt.apply(lambda x: x.split())
  with open('processed_data/python/train.jsonl','w') as f:
      for _, row in df_trn.iterrows():
          f.write(json.dumps(row.to_dict()) + '\n')

  df_tst['code_tokens'] = df_tst.mthd.apply(lambda x: x.split())
  df_tst['docstring_tokens'] = df_tst.cmt.apply(lambda x: x.split())
  with open('processed_data/python/test.jsonl','w') as f:
      for _, row in df_tst.iterrows():
          f.write(json.dumps(row.to_dict()) + '\n')

  df_val['code_tokens'] = df_val.mthd.apply(lambda x: x.split())
  df_val['docstring_tokens'] = df_val.cmt.apply(lambda x: x.split())
  with open('processed_data/python/valid.jsonl','w') as f:
      for _, row in df_val.iterrows():
          f.write(json.dumps(row.to_dict()) + '\n')
  df_val.to_csv("processed_data/python/valid.csv", index=False)





