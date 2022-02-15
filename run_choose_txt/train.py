import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils
from dataset import Dictionary, IconQAFeatureDataset
import base_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model, dataloader):
    score = 0
    num_data = 0

    for v, q, c, a, qid in iter(dataloader):
        with torch.no_grad():
            v = v.to(device)
            q = q.to(device)
            c = c.to(device)
            a = a.to(device)

            pred = model(v, q, c)
            batch_score = compute_score_with_logits(pred, a).sum() # accurate answer number
            score += batch_score
            num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    assert num_data == len(dataloader.dataset)
    return score


def train(model, train_loader, eval_loader, num_epochs, output, save_all, patience, lr):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters(), lr=lr)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    best_epoch = 0
    eval_scores = []

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        # for each mini-bach data
        for i, (v, q, c, a, pid) in enumerate(train_loader):
            v = v.to(device)
            q = q.to(device)
            c = c.to(device)
            a = a.to(device)

            pred = model(v, q, c)
            loss = instance_bce_with_logits(pred, a)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            # logits score
            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        # evaluation
        model.train(False)
        eval_score = evaluate(model, eval_loader)
        model.train(True)

        # save the model
        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
            best_epoch = epoch
            
        if save_all:
            model_path = os.path.join(output, 'model_{}_{}.pth'.format(epoch, "%.3f" % (100*eval_score)))
            torch.save(model.state_dict(), model_path)

        # print results
        logger.write('epoch %d, time: %.2f\t'             % (epoch, time.time()-t) +
                     'train_loss: %.3f, acc: %.3f,  '     % (total_loss, train_score) +
                     'val_acc: [%.3f] (best: [%.3f] @%d)' % (100*eval_score, 100*best_eval_score, best_epoch))

        # early stopping
        eval_scores.append(eval_score)
        if epoch > max(20, patience) and max(eval_scores[-patience:]) < best_eval_score: # last N epochs
            print("Early Stopping!\n")
            break

    logger.write('\tBEST evaluation score: %.3f @ %d' % (100*best_eval_score, best_epoch))


def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--task', type=str, default='choose_txt')
    parser.add_argument('--save_all', type=bool, default=False, help='save all model checkpoints or not')
    # input and output
    parser.add_argument('--feat_label', type=str, default='resnet101_pool5_79_icon',
                        help = 'resnet101_pool5_79_icon: icon pretrained model')
    parser.add_argument('--input', type=str, default='../data')
    parser.add_argument('--output', type=str, default='../saved_models/choose_txt')
    # data splits
    parser.add_argument('--train_split', type=str, default='train', choices=['minitrain', 'train', 'trainval', 'minival', 'val'],
                        help = 'minitrain: quick model validation')
    parser.add_argument('--val_split', type=str, default='val', choices=['minival', 'val'],
                        help = 'minival: quick model validation')
    # model and label
    parser.add_argument('--model', type=str, default='patch_transformer_ques_bert')
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--gpu', type=str, default='0')
    # transformer
    parser.add_argument('--num_patches', type=int, default=79)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--patch_emb_dim', type=int, default=768)
    # language model
    parser.add_argument('--lang_model', type=str, default='bert-small',
                        choices=['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'bert-base', 'bert-base-uncased'])
    parser.add_argument('--max_length', type=int, default=34, help='34 for choose_txt task')

    args = parser.parse_args()

    # print and save the args
    output = os.path.join(args.output, args.label)
    utils.create_dir(output)
    logger = utils.Logger(output + '/args.txt')
    for arg in vars(args):
        logger.write('%s: %s' % (arg, getattr(args, arg)))
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed) # CPU random seed
    torch.cuda.manual_seed(args.seed) # GPU random seed
    torch.backends.cudnn.benchmark = True

    # dataset
    dictionary = Dictionary.load_from_file(args.input + '/dictionary.pkl') # load dictionary
    train_dset = IconQAFeatureDataset(args.train_split, args.task, args.feat_label, args.input,
                                   dictionary, args.lang_model, args.max_length) # generate train data
    eval_dset = IconQAFeatureDataset(args.val_split, args.task, args.feat_label, args.input,
                                  dictionary, args.lang_model, args.max_length) # generate val data
    batch_size = args.batch_size

    # data loader
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)

    # build the model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args.lang_model,
                                             args.num_heads, args.num_layers, 
                                             args.num_patches, args.patch_emb_dim)
    # GPU
    device = torch.device('cuda:' + args.gpu)
    model.to(device)

    # model training
    print("\n------------------------- The model is training: -------------------------")
    output_path = os.path.join(args.output, args.label)
    train(model, train_loader, eval_loader, args.epochs, output_path, args.save_all, args.patience, args.lr)
    print("\n------------------------- Done! -------------------------")
