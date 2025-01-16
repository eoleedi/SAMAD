import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from transformers.modeling_outputs import TokenClassifierOutput
from transformer_model import *

from content_module import content_BLSTM
from delivery_module import *
from langUse_module import *

class MultiFeatureModel(nn.Module):
    def __init__(self, device, embed_dim=256, num_labels=5, hidden_dim=128):
        super(MultiFeatureModel, self).__init__()
        self.device = device
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.content_BLSTM = content_BLSTM().to(device)
        self.content_BLSTM.bert_freeze_feature_extractor()
        self.Delivery_BLSTM = Delivery_BLSTM(hidden_dim=13)
        self.Language_BLSTM = Language_BLSTM(POS, morph, DEP, hidden_dim).to(device)
        self.cross_attention_cd = QKVTransformer(3, 256, 4) # num_layers, d_model, num_heads
        self.cross_attention_cl = QKVTransformer(3, 256, 4) # num_layers, d_model, num_heads

        self.self_attention_content = Transformer(3, 512, 4) # num_layers, d_model, num_heads

        self.num_features = 256

        # Projection layers
        self.proj1 = nn.Linear(self.num_features*2, self.num_features*4)
        self.proj2 = nn.Linear(self.num_features*4, self.num_features*2)
        self.out_layer = nn.Linear(self.num_features*2, num_labels)


    def forward(self,
                padded_res_emb,
                attention_mask_res,
                padded_pro_emb,
                attention_mask_pro,
                padded_seq,
                masked_seq,
                delivery_tra,
                labels=None):
        
        # 1. Get the embedding
        x_content = self.content_BLSTM(padded_res_emb, padded_pro_emb, attention_mask_res, attention_mask_pro) # [8, 768], [8, 1]
        x_delivery = self.Delivery_BLSTM(delivery_tra)           # [8, 256], [batch_size, 1]
        x_langUse = self.Language_BLSTM(padded_seq, masked_seq)
        
        # 2. Cross-attention 
        trans_c_with_d = self.cross_attention_cd(x_content, x_delivery, x_delivery) # [2, 254, 256]
        trans_c_with_l = self.cross_attention_cl(x_content, x_langUse, x_langUse) # 2, 254, 256]
        # print(f'C->D: {trans_c_with_d.size()}, C-L: {trans_c_with_l.size()}')     # [batch, 256, 256]
        # 3.1 concate through feature size
        h_content_states = torch.cat([trans_c_with_d, trans_c_with_l], dim=2) # 2, 254, 512] 用feature size連在一起
        # print(f'Content states: {h_content_states.size()}') # [B, length, 512]
        # 3.2 self-attention + Take the last output for prediction
        h_content = self.self_attention_content(h_content_states) # [2, 256, 512]
        # print(f'After self attention states: {h_content.size()}')


        # 6. projection
        # A residual block
        last_hs = h_content[:,-1,:]
        # print('Last Hidden', last_hs.size())
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs))))
        last_hs_proj += last_hs
        # print(f'After Residual: {last_hs_proj.size()}') # [2, 4751, 256]

        # 7. predictor
        output = self.out_layer(last_hs_proj)
        # print(f'output layer: {output.size()}')

        overall_loss = None
        loss_fct = nn.CrossEntropyLoss()
        if labels is not None:
            labels = labels.long()
            loss_holistic = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
            overall_loss = loss_holistic

        # soft label
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss_holistic = loss_fct(output.view(-1, self.num_labels), labels)
        #     overall_loss = loss_holistic

        # train.py
        return TokenClassifierOutput(loss=overall_loss, logits=output) 
        # return output
