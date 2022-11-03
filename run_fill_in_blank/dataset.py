from __future__ import print_function
import os
import sys
import json
import _pickle as cPickle # python3
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils

from transformers import AutoTokenizer
bert_models = {'bert-base-uncased': 'bert-base-uncased',
               'bert-tiny':   'google/bert_uncased_L-2_H-128_A-2',
               'bert-mini':   'google/bert_uncased_L-4_H-256_A-4',
               'bert-small':  'google/bert_uncased_L-4_H-512_A-8',
               'bert-medium': 'google/bert_uncased_L-8_H-512_A-8',
               'bert-base':   'google/bert_uncased_L-12_H-768_A-12'}


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print(datetime.now().isoformat(), " ",'dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print(datetime.now().isoformat(), " ",'\nloading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word) # initialize the instance
        print(datetime.now().isoformat(), " ",'vocabulary number in the dictionary:', len(idx2word))
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _simplify_question(ques):
    """
    Simplify question: remove verbose sentences in the question.
    """
    sentences = ques.split(". ")
    if len(sentences) > 1 and "Count" in sentences[0] and " by " in sentences[0]:
        ques = ". ".join(sentences[1:])
        return ques
    else:
        return ques


def _load_dataset(dataroot, name, task, ans2label, label2ans, test_ids = []):
    """
    Load the IconQA dataset.
    - dataroot: root path of dataset
    - name: 'train', 'val', 'test', 'traninval', 'minitrain', 'minival', 'minitest'
    - task: 'fill_in_blank'
    """
    problems =  json.load(open(os.path.join(dataroot, 'iconqa_data', 'problems.json')))
    pid_splits = json.load(open(os.path.join(dataroot, 'iconqa_data', 'pid_splits.json')))

    pids = pid_splits['%s_%s' % (task, name)]
    print(datetime.now().isoformat(), " ","problem number for %s_%s:" % (task, name), len(pids))

    entries = []
    if name == 'test' and len(test_ids) > 0:        
        #print(pids[0:10])    
        #print(test_ids[0:10])
        pids = list(filter(lambda p: p in test_ids, pids))        
        print(datetime.now().isoformat(), " ", "Test data is filtered to ", len(pids), " items.")
    else:
        print(datetime.now().isoformat(), " ", "Test data filtering is NOT applied.")

    for pid in pids:
        prob = {}
        prob['question_id'] = pid
        prob['image_id'] = pid
        prob['question'] = _simplify_question(problems[pid]['question'])
        prob['ques_type'] = problems[pid]['ques_type']

        utils.assert_eq(task, prob['ques_type'])

        # answer to label
        if 'test' not in name: # train, val
            ans = problems[pid]['answer']
            prob['answer'] = ans
            prob['answer_label'] = ans2label[ans]
            utils.assert_eq(ans, label2ans[prob['answer_label']])
        else: # test
            ans = problems[pid]['answer']
            prob['answer'] = ans
            prob['answer_label'] = None

        entries.append(prob)

    return entries


class IconQAFeatureDataset(Dataset):
    def __init__(self, name, task, feat_label, dataroot, dictionary, lang_model, max_length, test_ids = []):
        super(IconQAFeatureDataset, self).__init__ ()
        assert name in ['train', 'val', 'test', 'traninval', 'minitrain', 'minival', 'minitest']
        assert 'bert' in lang_model

        self.dictionary = dictionary
        self.lang_model = lang_model
        self.max_length = max_length # max question word length

        # load answers
        ans2label_path = os.path.join(dataroot, 'trainval_'+task+'_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'trainval_'+task+'_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb')) # dict
        self.label2ans = cPickle.load(open(label2ans_path, 'rb')) # list
        self.num_ans_candidates = len(self.ans2label) # 3129

        # load and tokenize the questions
        self.entries = _load_dataset(dataroot, name, task, self.ans2label, self.label2ans, test_ids)
        if 'bert' in self.lang_model:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_models[self.lang_model]) # For Bert
        self.tokenize()
        self.tensorize()

        # load image features
        h5_path = os.path.join(dataroot, 'patch_embeddings', feat_label, 'iconqa_%s_%s_%s.pth' % (name, task, feat_label))
        print(datetime.now().isoformat(), " ",'\nloading features from h5 file:', h5_path)
        self.features = torch.load(h5_path)
        self.v_dim = list(self.features.values())[0].size()[1] # [num_patch,2048]
        print(datetime.now().isoformat(), " ","visual feature dim:", self.v_dim)

    def tokenize(self):
        """
        Tokenize the questions.
        This will add q_token in each entry of the dataset.
        """
        print(datetime.now().isoformat(), " ",'max question token length is:', self.max_length)

        for entry in self.entries:
            if 'bert' in self.lang_model:
                tokens = self.tokenizer(entry['question'])['input_ids']
                tokens = tokens[-self.max_length:]
                if len(tokens) < self.max_length:
                    tokens = tokens + [0] * (self.max_length - len(tokens))

            entry['q_token'] = tokens
            assert len(entry['q_token']) == self.max_length

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer_label = np.array(entry['answer_label'])
            if answer_label:
                answer_label = torch.from_numpy(answer_label)
                entry['answer_label'] = answer_label
            else:
                entry['answer_label'] = None

    def __getitem__(self, index):
        entry = self.entries[index]

        features = self.features[int(entry['image_id'])]
        
        question_id = entry['question_id']
        question = entry['q_token']
        answer_label = entry['answer_label']

        target = torch.zeros(self.num_ans_candidates)
        if answer_label is not None:
            target[answer_label] = 1.0

        return features, question, target, question_id

    def __len__(self):
        return len(self.entries)
