#!/usr/bin/env python

"""
    obtext/main.py
"""

import sys
import torch
import argparse
import numpy as np
from scipy.stats import ortho_group

BERT_DIM = 768

class Encoder:
    def __init__(self, layer=-1, cuda=True):
        self.tokenizer = torch.hub.load(
            'huggingface/pytorch-pretrained-BERT',
            'bertTokenizer',
            'bert-base-cased',
            do_basic_tokenize=False,
            do_lower_case=False,
        )
        
        self.model = torch.hub.load(
            'huggingface/pytorch-pretrained-BERT',
            'bertModel',
            'bert-base-cased'
        ).eval()
        
        self.layer = -1
        
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()
    
    def encode_one(self, text):
        text = '[CLS] ' + text.strip()
        toks = self.tokenizer.tokenize(text)
        toks = self.tokenizer.convert_tokens_to_ids(toks)
        segs = [0] * len(toks)
        
        toks = torch.tensor([toks])
        segs = torch.tensor([segs])
        
        if self.cuda:
            toks = toks.cuda()
            segs = segs.cuda()
        
        with torch.no_grad():
            emb, _ = self.model(toks, segs)
        
        return emb[0][self.layer]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model = Encoder(cuda=args.cuda)
    
    _ = np.random.seed(args.seed)
    _ = torch.manual_seed(args.seed + 1)
    _ = torch.cuda.manual_seed(args.seed + 2)
    
    rot = ortho_group.rvs(BERT_DIM)
    np.save('.rot', rot)
    
    inp = sys.stdin
    for i, line in enumerate(inp):
        
        # Compute BERT encodings
        emb = model.encode_one(line)
        
        # Apply random rotation to make decoding more difficult
        emb = emb.cpu().numpy() @ rot
        
        # Write
        np.savetxt(sys.stdout, emb, fmt='%f')