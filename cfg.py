#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# !unzip progex-v3.4.5.zip


# In[ ]:





# In[ ]:


import os
import pandas as pd
import shutil
from tqdm.auto import tqdm


# In[ ]:


train = pd.read_json("data/java_train_0.jsonl", lines=True)


# In[ ]:


# train = train[:1000]


# In[ ]:


for index, row in tqdm(train.iterrows(), total=len(train)):
    if os.path.exists(f"data/train/{index}"):
        shutil.rmtree(f"data/train/{index}")
    os.mkdir(f"data/train/{index}")
    code = "public class " + f"Java{index}" + " {\n"
    code += row["original_string"]
    code += "\n}"
    with open(f"data/train/{index}/Java{index}.java", "w") as f:
        f.write(code)
    # 要执行的命令
    command = f"java -jar progex-v3.4.5/progex.jar -cfg data/train/{index} -outdir data/train/{index}"
    # 执行命令
    output = os.system(command)


# In[ ]:


valid = pd.read_json("data/java_valid_0.jsonl", lines=True)


# In[ ]:


for index, row in tqdm(valid.iterrows(), total=len(valid)):
    if os.path.exists(f"data/valid/{index}"):
        shutil.rmtree(f"data/valid/{index}")
    os.mkdir(f"data/valid/{index}")
    code = "public class " + f"Java{index}" + " {\n"
    code += row["original_string"]
    code += "\n}"
    with open(f"data/valid/{index}/Java{index}.java", "w") as f:
        f.write(code)
    # 要执行的命令
    command = f"java -jar progex-v3.4.5/progex.jar -cfg data/valid/{index} -outdir data/valid/{index}"
    # 执行命令
    output = os.system(command)


# In[ ]:


test = pd.read_json("data/java_test_0.jsonl", lines=True)


# In[ ]:


for index, row in tqdm(test.iterrows(), total=len(test)):
    if os.path.exists(f"data/test/{index}"):
        shutil.rmtree(f"data/test/{index}")
    os.mkdir(f"data/test/{index}")
    code = "public class " + f"Java{index}" + " {\n"
    code += row["original_string"]
    code += "\n}"
    with open(f"data/test/{index}/Java{index}.java", "w") as f:
        f.write(code)
    # 要执行的命令
    command = f"java -jar progex-v3.4.5/progex.jar -cfg data/test/{index} -outdir data/test/{index}"
    # 执行命令
    output = os.system(command)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




