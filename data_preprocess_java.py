# -*- coding: utf-8 -*-


import pandas as pd
import re
import json
from pathlib import Path
from typing import List, Optional
from tqdm.auto import tqdm

def jsonl_list_to_dataframe(file_list, columns=['code', 'docstring']):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    return pd.concat([pd.read_json(f,
                                   orient='records', 
                                   compression='gzip',
                                   lines=True)[columns] 
                      for f in file_list], sort=False)

def get_dfs(path: Path) -> List[pd.DataFrame]:
    """Grabs the different data splits and converts them into dataframes"""
    dfs = []
    for split in ["train", "valid", "test"]:
        files = sorted((path/split).glob("**/*.gz"))
        df = jsonl_list_to_dataframe(files).rename(columns = {'code': 'mthd', 'docstring': 'cmt'})
        dfs.append(df)
        
    return dfs


def is_ascii(s):
    '''
    Determines if the given string contains only ascii characters

    :param s: the string to check
    :returns: whether or not the given string contains only ascii characters
    '''
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def get_inline_pairs(mthd):
    '''
    Get all pairs of inline comments and corresponding code snippets

    :param mthd: the method to retrieve the pairs of comments and corresponding
    code snippets from
    :returns: all pairs of comments and corresponding code snippets
    '''
    pairs = [[]]

    comment = False
    bracket = False
    indent_lvl = -1
    lines = mthd.split("\n")
    for line in lines:
        if "//" in line and not bracket and not "://" in line:
            pairs[-1].append(line)
            if '\t' in line:
                indent_lvl = line.count('\t')
            else:
                indent_lvl = line.split("//")[0].count(' ')
            comment = True
            bracket = False
        elif comment:
            if '{' in line and not bracket:
                bracket = True
                pairs[-1].append(line)
            elif '}' in line:
                line_indent = -1
                if '\t' in line:
                    line_indent = line.count('\t')
                else:
                    line_indent = line.split("//")[0].count(' ')
                if indent_lvl == line_indent:
                    pairs[-1].append(line)
                if not bracket:
                    pairs.append([])
                    comment = False
                    bracket = False
            elif line.isspace() or line == '' and not bracket:
                pairs.append([])
                comment = False
            else:
                pairs[-1].append(line)
    
    # Convert pairs into proper format of (code snippet, inline comment) dataframe
    code_snippets   = []
    comments        = []
    for pair in pairs:
        if pair and len(pair) < 5:
            code    = []
            comment = []
            skip = False
            for line in pair:
                if "TODO" in line: break
                if "//" in line:
                    comment.append(line.replace('//', ''))
                else:
                    code.append(line)
            if len(code) > 1 and len(comment) > 0:
                        code_snippets.append('\n'.join(code))
                        comments.append('\n'.join(comment))

    pairs = pd.DataFrame(zip(code_snippets, comments), columns = ["mthd", "cmt"])
    return pairs


def add_inline(df: pd.DataFrame) -> pd.DataFrame:

    new_df = df[df['mthd'].str.contains("//")]
    all_pairs = []
    for mthd in tqdm(new_df.mthd.values):
        pairs = get_inline_pairs(mthd)
        all_pairs.append(pairs)

    df_pairs = pd.concat([pairs for pairs in all_pairs])
    return pd.concat([df, df_pairs])

def has_code(cmt: str) -> bool:
    if '<code>' in cmt: return True
    else: return False

def remove_jdocs(df: pd.DataFrame) -> pd.DataFrame:

    methods = []
    comments = []
    for i, row in tqdm(list(df.iterrows())):
        comment = row["cmt"]
        # Remove {} text in comments from https://stackoverflow.com/questions/14596884/remove-text-between-and-in-python/14598135
        comment = re.sub("([\{\[]).*?([\)\}])", '', comment)
        
        
        cleaned = []
        for line in comment.split('\n'):
            if "@" in line: break
            cleaned.append(line)
        comments.append('\n'.join(cleaned))
        methods.append(row["mthd"])
    new_df = pd.DataFrame(zip(methods, comments), columns = ["mthd", "cmt"])

    return new_df

def clean_html(cmt: str) -> str:

    result = re.sub(r"<.?span[^>]*>|<.?code[^>]*>|<.?p[^>]*>|<.?hr[^>]*>|<.?h[1-3][^>]*>|<.?a[^>]*>|<.?b[^>]*>|<.?blockquote[^>]*>|<.?del[^>]*>|<.?dd[^>]*>|<.?dl[^>]*>|<.?dt[^>]*>|<.?em[^>]*>|<.?i[^>]*>|<.?img[^>]*>|<.?kbd[^>]*>|<.?li[^>]*>|<.?ol[^>]*>|<.?pre[^>]*>|<.?s[^>]*>|<.?sup[^>]*>|<.?sub[^>]*>|<.?strong[^>]*>|<.?strike[^>]*>|<.?ul[^>]*>|<.?br[^>]*>", "", cmt)
    return result



if __name__ == "__main__":

    path = Path('./dataset')
    df_trn, df_val, df_tst = get_dfs(path/"java/final/jsonl")
    #choose 1% data to do the test
    #sample = 0.01

    #df_trn = df_trn.sample(frac = sample)
    #df_val = df_val.sample(frac = sample)
    #df_tst = df_tst.sample(frac = sample)

    df_trn = df_trn[df_trn['mthd'].apply(lambda x: is_ascii(x))]
    df_val = df_val[df_val['mthd'].apply(lambda x: is_ascii(x))]
    df_tst = df_tst[df_tst['mthd'].apply(lambda x: is_ascii(x))]

    df_trn = df_trn[df_trn['cmt'].apply(lambda x: is_ascii(x))]
    df_val = df_val[df_val['cmt'].apply(lambda x: is_ascii(x))]
    df_tst = df_tst[df_tst['cmt'].apply(lambda x: is_ascii(x))]

    df_trn = add_inline(df_trn)
    df_val = add_inline(df_val)
    df_tst = add_inline(df_tst)

    df_trn = df_trn[df_trn.apply(lambda row: len(row.mthd) > len(row.cmt), axis = 1)]
    df_val = df_val[df_val.apply(lambda row: len(row.mthd) > len(row.cmt), axis = 1)]
    df_tst = df_tst[df_tst.apply(lambda row: len(row.mthd) > len(row.cmt), axis = 1)]


    df_trn = df_trn[~df_trn['cmt'].apply(lambda x: has_code(x))]
    df_val = df_val[~df_val['cmt'].apply(lambda x: has_code(x))]
    df_tst = df_tst[~df_tst['cmt'].apply(lambda x: has_code(x))]

    df_trn = remove_jdocs(df_trn);
    df_val = remove_jdocs(df_val);
    df_tst = remove_jdocs(df_tst);

    df_trn.cmt = df_trn.cmt.apply(clean_html)
    df_val.cmt = df_val.cmt.apply(clean_html)
    df_tst.cmt = df_tst.cmt.apply(clean_html)

    df_trn = df_trn.applymap(lambda x: ' '.join(x.split()).lower())
    df_val = df_val.applymap(lambda x: ' '.join(x.split()).lower())
    df_tst = df_tst.applymap(lambda x: ' '.join(x.split()).lower())

    df_trn = df_trn[~(df_trn['cmt'] == '')]
    df_val = df_val[~(df_val['cmt'] == '')]
    df_tst = df_tst[~(df_tst['cmt'] == '')]

    df_trn = df_trn[~df_trn['cmt'].duplicated()]
    df_val = df_val[~df_val['cmt'].duplicated()]
    df_tst = df_tst[~df_tst['cmt'].duplicated()]



    df_trn['code_tokens'] = df_trn.mthd.apply(lambda x: x.split())
    df_trn['docstring_tokens'] = df_trn.cmt.apply(lambda x: x.split())
    with open('processed_data/java/train.jsonl','w') as f:
        for _, row in df_trn.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')

    df_val['code_tokens'] = df_val.mthd.apply(lambda x: x.split())
    df_val['docstring_tokens'] = df_val.cmt.apply(lambda x: x.split())
    with open('processed_data/java/valid.jsonl','w') as f:
        for _, row in df_val.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    df_val.to_csv("processed_data/java/valid.csv", index=False)



    df_tst['code_tokens'] = df_tst.mthd.apply(lambda x: x.split())
    df_tst['docstring_tokens'] = df_tst.cmt.apply(lambda x: x.split())
    with open('processed_data/java/test.jsonl','w') as f:
        for _, row in df_tst.iterrows():
            f.write(json.dumps(row.to_dict()) + '\n')
    
    print(len(df_trn), len(df_val), len(df_tst))