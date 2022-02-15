import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.attention import Attention
from models.classifier import SimpleClassifier
from models.fc import FCNet

from models.patch_transformer import Patch_Transformer

from transformers import AutoModel
bert_models = {'bert-base-uncased': 'bert-base-uncased',
               'bert-tiny':   'google/bert_uncased_L-2_H-128_A-2',
               'bert-mini':   'google/bert_uncased_L-4_H-256_A-4',
               'bert-small':  'google/bert_uncased_L-4_H-512_A-8',
               'bert-medium': 'google/bert_uncased_L-8_H-512_A-8',
               'bert-base':   'google/bert_uncased_L-12_H-768_A-12'}


class Transformer_Bert_Model(nn.Module):
    def __init__(self, bert_net, w_net, v_trm_net, v_att, q_net, v_net, c_net, qc_net, classifier, patch_dim):
        super(Transformer_Bert_Model, self).__init__()
        self.bert_net = bert_net # BERT
        self.w_net = w_net
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.c_net = c_net
        self.qc_net = qc_net
        self.classifier = classifier
        self.v_trm_net = v_trm_net
        self.patch_dim = patch_dim

    def patch_net(self, v, q_emb):
        v = self.v_trm_net(v) # [N,num_patch+1,num_hid]
        att = self.v_att(v, q_emb) # [N,num_patch+1,1]
        v_emb = (att * v).sum(1) # [N,num_hid]
        return v_emb

    def bert(self, input_ids):
        last_hidden_states = self.bert_net(input_ids) # [N,59,num_hid]
        q_features = last_hidden_states[0][:, 0, :] # [N,num_hid]
        return q_features

    def merge_img_que(self, v, q):
        q_emb = self.bert(q)
        q_emb = self.w_net(q_emb)
        v_emb = self.patch_net(v, q_emb) 
        q_repr = self.q_net(q_emb) # [batch, num_hid]
        v_repr = self.v_net(v_emb) # [batch, num_hid]
        joint_repr = q_repr * v_repr
        return joint_repr

    def forward(self, v, q, c):
        """
        v: [batch, num_pats, v_dim]
        q: [batch_size, seq_length]
        c: [batch_size, c_num, v_dim]
        return: logits, not probs
        """
        batch, c_num, v_dim = c.size()
        joint_vq = self.merge_img_que(v, q).unsqueeze(1)
        c_repr = self.c_net(c.view(batch, -1)).view(batch, c_num, -1)
        qc_repr = torch.cat((joint_vq, c_repr), 1)
        qc_repr= qc_repr.view(batch, -1)
        joint_repr = self.qc_net(qc_repr)
        joint_repr =  joint_repr * joint_vq.squeeze(1)
        logits = self.classifier(joint_repr)
        return logits


def build_patch_transformer_ques_bert(dataset, num_hid, lang_model, num_heads, num_layers, num_patches, patch_emb_dim):
    assert dataset.c_num == 5 # max choice number

    # bert model for questions
    bert_dims = {'bert-tiny':128, 'bert-mini':256, 'bert-small':512, 'bert-medium':512, 'bert-base':768, 'bert-base-uncased':768}
    q_dim = bert_dims[lang_model]
    bert_net = AutoModel.from_pretrained(bert_models[lang_model])
    w_net = FCNet([q_dim, num_hid])

    # transformer model for images
    v_trm_net = Patch_Transformer(num_patches, num_heads, num_layers, patch_emb_dim)

    # attention net
    v_att = Attention(patch_emb_dim, num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([patch_emb_dim, num_hid])
    c_net = FCNet([dataset.v_dim*dataset.c_num, num_hid*dataset.c_num])
    qc_net = FCNet([num_hid * (dataset.c_num + 1), num_hid])

    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.c_num, 0.5)
    return Transformer_Bert_Model(bert_net, w_net, v_trm_net, v_att, q_net, v_net, c_net, qc_net, classifier, patch_emb_dim)
