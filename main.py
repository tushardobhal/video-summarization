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

from DataLoader import DataLoader
from Vocabulary import Vocabulary

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_model(model, opt):
    print("Training model for num_epochs - {}, vocab_size - {}...".format(opt.epochs, opt.target_feature_size))
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
        
    model.train()
    trainloader = DataLoader(opt=opt, train=True)
    
    for epoch in range(opt.epochs):
        total_loss = 0
        for i, (src, trg) in enumerate(trainloader.batch_data_generator()):
            trg_input = trg[:, :-1] # not include the end of sentence
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.long())
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            total_loss += loss.item()

        print("Epoch [{}]/[{}], Loss = {}".format(epoch, opt.epochs, total_loss))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), "{}/model_{}.pth".format(opt.model_save_dir, epoch))
			

def eval_model(model, opt):
    print("Evaluating model...")
    model.load_state_dict(torch.load(opt.model_save_dir + '/model_595.pth'))
    model.eval()
    print("Transformer model loaded")
    evalloader = DataLoader(opt=opt, train=False)

    total_loss = 0
    for i, (src, trg) in enumerate(evalloader.batch_data_generator()):
        with torch.no_grad():
            trg_input = trg[:, :-1] # not include the end of sentence
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.long())
            total_loss += loss.item()
            print("Loss - {}, Preds size - {}".format(total_loss, preds.size()))

            batch, seq, word = preds.size()
            for j in range(batch):
                sentence = ' '.join(evalloader.get_sentence_from_tensor(preds[j]))
                # print(sentence)

    print("Total Validation loss is - {}".formattotal_loss()) 
		  

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2000)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.2)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-vid_feat_size', type=int, default=512)
    parser.add_argument('-save_freq', type=int, default=20)
    parser.add_argument('-model_save_dir', default='model')
    # DataLoader
    parser.add_argument('-num_train_set', type=int, default=1300)
    parser.add_argument('-video_features_file', default='data/features_video_pca.npz')
    # parser.add_argument('-video_descriptions_file', default='data/video_descriptions_10_sentence.pickle')
    parser.add_argument('-video_descriptions_file', default='data/video_descriptions.pickle')
    # parser.add_argument('-vocab_file', default='data/vocab_10_sentence.pickle')
    parser.add_argument('-vocab_file', default='data/vocab.pickle')
    parser.add_argument('-video_descriptions_csv', default='data/video_description.csv')
    parser.add_argument('-target_feature_size', type=int, default=14238)
 
    opt = parser.parse_args()

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = get_model(opt, opt.vid_feat_size, opt.target_feature_size)
    model = nn.DataParallel(model)

    if opt.mode == 'train':
        train_model(model, opt)
    elif opt.mode == 'eval':
        eval_model(model, opt)
    else:
        print("Wrong option. Give either 'train' or 'eval' as input to -mode")


def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), '{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open('{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open('{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
