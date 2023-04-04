
import torch
import tqdm
import torch.nn as nn
import pandas as pd
from model import Seq2Seq
from transformers import RobertaConfig, RobertaModel, AutoTokenizer
from run import convert_examples_to_features, Example
from pathlib import Path
import pandas as pd

class Args:
    max_source_length = 256
    max_target_length = 48

args = Args()

def get_preds(df: pd.DataFrame):
    ps = []
    for idx, row in df.iterrows():
        examples = [
            Example(idx, source = row.mthd, target = row.cmt)
        ]
        eval_features = convert_examples_to_features(
            examples, tokenizer, args, stage='test'
        )
        source_ids = torch.tensor(eval_features[0].source_ids, dtype = torch.long).unsqueeze(0).to('cuda')
        source_mask = torch.tensor(eval_features[0].source_mask, dtype = torch.long).unsqueeze(0).to('cuda')

        with torch.no_grad():
            preds = model(source_ids = source_ids, source_mask = source_mask)  
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                ps.append(text)
    
    return ps
    
if __name__ == "__main__":  

    pretrained_module = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_module)
    config = RobertaConfig.from_pretrained(pretrained_module)
    encoder = RobertaModel.from_pretrained(pretrained_module, config = config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder = encoder,decoder = decoder,config=config,
                    beam_size=10, max_length=48,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    model.load_state_dict(torch.load(Path(f'model/solidity')/"checkpoint-last/pytorch_model.bin"))
    model.to('cuda')

    df_val = pd.read_csv("processed_data/solidity/valid.csv")

    df_val = df_val.reset_index()
    preds = get_preds(df_val.head(10))
    for idx, row in df_val.head(10).iterrows():
        print('Code:', row.mthd)
        print('Original Comment:', row.cmt)
        print('Generated Comment:', preds[idx])
        print('='*40)