import torch
from torch import nn
from torch.nn import functional as F
from .Module import Linear, SublayerConnection
from copy import deepcopy
from math import inf


class GAT(nn.Module):
    def __init__(self, d_model, num_head, num_layer, dropout):
        super().__init__()
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_model // 2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.d_model = d_model
        self.scale = d_model ** 0.5
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.w_head = Linear(d_model, d_model)
        self.w_tail = Linear(d_model, d_model)
        self.w_info = Linear(d_model, d_model)
        self.w_comb = Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, inputs, graph):
        batch_size = inputs.size(0)
        num_seq = inputs.size(1)
        _, (memory, _) = self.lstm(inputs.view(batch_size * num_seq, 1, -1))
        memory = memory.view(batch_size, num_seq, -1)
        memory = self.norm(self.dropout(memory))
        graph = (graph.unsqueeze(1) == 0)
        for _ in range(self.num_layer):
            # inputs = inputs.view(batch_size, num_seq, -1)
            head = self.w_head(inputs).view(batch_size, num_seq, self.num_head, -1)
            tail = self.w_tail(memory).view(batch_size, num_seq, self.num_head, -1)
            info = self.w_info(memory).view(batch_size, num_seq, self.num_head, -1)
            head.transpose_(-2, -3).contiguous()
            tail.transpose_(-2, -3).contiguous()
            info.transpose_(-2, -3).contiguous()
            score = torch.matmul(head, tail.transpose_(-1, -2).contiguous()) / self.scale
            score.masked_fill_(graph, -inf)
            attn_weight = F.softmax(score, dim=-1)
            attn_weight = attn_weight.masked_fill(graph, 0.)
            attn_vector = torch.matmul(attn_weight, info)
            attn_vector.transpose_(-2, -3)
            attn_vector = self.w_comb(attn_vector.contiguous().view(batch_size, num_seq, -1))
            attn_vector = attn_vector.view(2, batch_size * num_seq, -1)
            _, (memory, _) = self.lstm(inputs.view(batch_size * num_seq, 1, -1), (attn_vector, attn_vector))
            memory = self.norm(self.dropout(memory.view(batch_size, num_seq, -1)))

        return memory
