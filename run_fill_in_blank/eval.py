import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils
from dataset import Dictionary, IconQAFeatureDataset
import base_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str).replace('_ ', '')


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def compute_test_acc(pred_logits, qIds, dataloader):
    acc = 0
    results = {}
    results_detail = {}
    N = len(dataloader.dataset)
    assert N == qIds.size(0)

    for i, data in enumerate(dataloader.dataset.entries):
        pid = data['question_id']
        gt_ans = data['answer']
        pred_ans = get_answer(pred_logits[i], dataloader)
        results[pid] = pred_ans
        is_correct = pred_ans == gt_ans
        results_detail[pid] = is_correct
        if is_correct:
            acc += 1
    acc = 100 * acc / N
    return acc, results, results_detail


@torch.no_grad()
def evaluate(model, dataloader, output):
    print("\nThe model is testing")
    utils.create_dir(output)
    model.train(False)

    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    pbar = tqdm(total=len(dataloader))

    for i, (v, q, a, qid) in enumerate(dataloader):
        pbar.update(1)
        batch_size = v.size(0)

        v = v.to(device)
        q = q.to(device)
        logits = model(v, q)

        ques_ids = torch.IntTensor([int(ques_id) for ques_id in qid])
        pred[idx:idx + batch_size, :].copy_(logits.data)
        qIds[idx:idx + batch_size].copy_(ques_ids)
        idx += batch_size

        if args.debug:
            print("\nQuestion id:", qid[0])
            print(get_question(q.data[0], dataloader))
            print(get_answer(logits.data[0], dataloader))

    pbar.close()

    acc, results, results_detail = compute_test_acc(pred, qIds, dataloader)
    print("\nTest acc: %.3f" % acc)

    # save results to json file
    data = {}
    data['accuracy'] = acc
    data['args'] = {}
    for arg in vars(args):
        data['args'][arg] = getattr(args, arg)
    data['results'] = results
    data['results_detail'] = results_detail
    with open("{}/{}_{}.json".format(args.output, args.label, args.model), 'w') as f:
        json.dump(data, f, indent = 2, separators=(',', ': '))

    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--task', type=str, default='fill_in_blank')
    # input and output
    parser.add_argument('--feat_label', type=str, default='resnet101_pool5_79_icon',
                        help = 'resnet101_pool5_79_icon: icon pretrained model')
    parser.add_argument('--input', type=str, default='../data')
    parser.add_argument('--model_input', type=str, default='../saved_models/fill_in_blank')
    parser.add_argument('--output', type=str, default='../results/fill_in_blank')
    # model and label
    parser.add_argument('--model', type=str, default='patch_transformer_ques_bert')
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--check_point', type=str, default='best_model.pth')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--debug", default=False, help='debug mode or not')
    # transformer
    parser.add_argument('--num_patches', type=int, default=79)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--patch_emb_dim', type=int, default=768)
    # language model
    parser.add_argument('--lang_model', type=str, default='bert-small',
                        choices=['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'bert-base', 'bert-base-uncased'])
    parser.add_argument('--max_length', type=int, default=34)

    # filter testing data
    parser.add_argument('--test_ids', type=str, default=[], nargs='*')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))

    torch.backends.cudnn.benchmark = True

    # dataset
    dictionary = Dictionary.load_from_file(args.input + '/dictionary.pkl') # load dictionary
    eval_dset = IconQAFeatureDataset('test', args.task, args.feat_label, args.input,
                                  dictionary, args.lang_model, args.max_length, args.test_ids) # generate test data
    batch_size = args.batch_size

    # data loader
    test_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)

    # build the model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.lang_model,
                                             args.num_heads, args.num_layers, 
                                             args.num_patches, args.patch_emb_dim)

    # load the trained model
    model_path = os.path.join(args.model_input, args.label, args.check_point)
    print('\nloading %s' % model_path)
    model_data = torch.load(model_path)
    model.load_state_dict(model_data.get('model_state', model_data), strict=False) # ignore missing key(s) in state_dict

    # GPU
    device = torch.device('cuda:' + args.gpu)
    model.to(device)

    # model testing
    evaluate(model, test_loader, args.output)
