import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from models.fc import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Attention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def logits(self, v, q):
        batch, kk, vdim = v.size()
        v_proj = self.v_proj(v)
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, kk, 1)
        joint_repr = v_proj * q_proj 
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr) 
        return logits

    def forward(self, v, q):
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w
