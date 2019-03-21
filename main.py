import argparse
import time
import torch
from Models import get_model
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import os
import csv
import nltk
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import pickle as pickle
import torch.nn as nn

#from activitynet import DataLoader
from DataLoader import DataLoader
from Vocabulary import Vocabulary

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_model(model, trainloader, evalloader, opt):
    for epoch in range(opt.epochs):
        total_loss = 0
        for i, (src, trg) in enumerate(trainloader.batch_data_generator()):
            trg_input = trg[:, :-1] # not include the end of sentence
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.long())
            #if i % opt.log_frequency == 0:
            #    print("Epoch [{}][{}] Batch [{}] Loss = {}".format(epoch, opt.epochs, i, loss.item()))
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            total_loss += loss.item()

        print("Epoch: [{}]/[{}], Loss: {}".format(epoch, opt.epochs, total_loss))
        if epoch % opt.save_freq == opt.save_freq - 1:
            torch.save(model.state_dict(), "{}/i3d_model_{}.pth".format(opt.model_save_dir, epoch))
            eval_model(model, evalloader, opt)
			

def eval_model(model, evalloader, opt):
    total_loss = 0
    for i, (src, trg) in enumerate(evalloader.batch_data_generator()):
        with torch.no_grad():
            trg_input = trg[:, :-1] # not include the end of sentence
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.long())
            total_loss += loss.item()
            
            # Uncomment to print the sentences
            batch, seq, word = preds.size()
            sentences = []
            for j in range(batch):
                sentence = ' '.join(evalloader.get_sentence_from_tensor(preds[j]))
                sentences.append(sentence)
            print(sentences[-1])

    print("Total Validation loss: {}".format(total_loss)) 
		  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='train')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2000)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=27)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-vid_feat_size', type=int, default=512)
    parser.add_argument('-save_freq', type=int, default=2)
    parser.add_argument('-model_save_dir', default='model')
    parser.add_argument('-log_frequency', default=20)
    # DataLoader
    parser.add_argument('-num_train_set', type=int, default=1300)
    parser.add_argument('-video_features_file', default='data/features_video_rgb_pca_i3d.npz')
    parser.add_argument('-video_descriptions_file', default='data/video_descriptions.pickle')
    parser.add_argument('-vocab_file', default='data/vocab.pickle')
    parser.add_argument('-video_descriptions_csv', default='data/video_description.csv')
    parser.add_argument('-target_feature_size', type=int, default=14238)
 
    opt = parser.parse_args()

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_model(opt, opt.vid_feat_size, opt.target_feature_size)
    model = nn.DataParallel(model)

    if opt.mode == 'train':
        print("Training model for num_epochs - {}, vocab_size - {}...".format(opt.epochs, opt.target_feature_size))
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
        if opt.SGDR == True:
            opt.sched = CosineWithRestarts(opt.optimizer, T_max = 10)
        model.train()
        trainloader = DataLoader(opt=opt, train=True)
        evalloader = DataLoader(opt=opt, train=False)
        train_model(model, trainloader, evalloader, opt)
    elif opt.mode == 'eval':
        print("Evaluating model...")
        model.load_state_dict(torch.load(opt.model_save_dir + '/model_595.pth'))
        model.eval()
        print("Transformer model loaded")
        evalloader = DataLoader(opt=opt, train=False)
        eval_model(model, evalloader, opt)
    else:
        print("Wrong option. Give either 'train' or 'eval' as input to -mode")

if __name__ == "__main__":
    main()
