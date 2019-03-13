import argparse
import time
import torch
from utransformer import UTransformer
import torch.nn as nn
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import os
import csv
import nltk
import numpy as np
from tqdm import tqdm
import pickle as pickle
from tensorboardX import SummaryWriter

#from DataLoader import DataLoader
from activitynet import DataLoader
from Vocabulary import Vocabulary

import pdb

writer = SummaryWriter('runs')

def train_model(model, opt, trainloader, evalloader):
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
    
    for epoch in range(opt.epochs):
        print("epoch: ", epoch)

        total_loss = 0
        if epoch % 10 == 9:
            opt.optimizer.param_groups[0]['lr'] *= 0.98

        for i, (src, trg, vid) in enumerate(trainloader.batch_data_generator()):
          trg_input = trg[:, :-1] # not include the end of sentence

          preds = model(src, trg_input)
          ys = trg[:, 1:].contiguous().view(-1)
          opt.optimizer.zero_grad()
          loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.long())
          loss.backward()
          opt.optimizer.step()
          if opt.SGDR == True: 
              opt.sched.step()

          total_loss += loss.item()

        avg_loss = total_loss / (i + 1)
        epoch_time = (time.time() - start)
        print("%dm %ds: loss = %.3f\n" %(epoch_time//60, epoch_time%60, avg_loss))
        writer.add_scalar('train/train_loss', avg_loss, epoch + 1)

        if epoch % opt.save_freq == 0:
          torch.save(model.state_dict(), 'weights/ut_activitynet_c3d')
          cptime = time.time()
          eval_model(model, evalloader, opt, epoch)
          print("model saved at epoch ", epoch)

    # save final weights
    torch.save(model.state_dict(), 'weights/ut_model_activitynet')

def eval_model(model, evalloader, opt, epoch):
    total_loss = 0
    for i, (src, trg, vid) in enumerate(evalloader.batch_data_generator()):
        with torch.no_grad():
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            preds = model(src, trg_input)
            ys = trg[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys.long())
            total_loss += loss.item()

            # Uncomment to print the sentences
            batch, seq, word = preds.size()
            sentences = []
            for j in range(batch):
                try:
                    sentence = ' '.join(evalloader.get_sentence_from_tensor(preds[j]))
                    sentences.append(sentence)
                except UnicodeEncodeError:
                    print("Error")
            try:
                print(vid[0], sentences[0])
                print(" GT:- ", evalloader.get_words_from_index(trg[0]))
                #print(" GT:- ", evalloader.get_sentence_from_tensor(trg[0]))
                print(50 * '-')
            except UnicodeDecodeError:
                print("Error")
       
    writer.add_scalar('val/val_loss', loss.item(), epoch + 1)

    print("Total Validation loss: {}".format(total_loss))

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-src_data', required=True)
    # parser.add_argument('-trg_data', required=True)
    # parser.add_argument('-src_lang', required=True)
    # parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=500)
    parser.add_argument('-d_model', type=int, default=500)
    parser.add_argument('-n_layers', type=int, default=8)
    parser.add_argument('-heads', type=int, default=10)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-printevery', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0002)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=90)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-vid_feat_size', type=int, default=500)
    parser.add_argument('-save_freq', type=int, default=5)

    # DataLoader
    parser.add_argument('-num_train_set', type=int, default=8300)

    # MSVD
    #parser.add_argument('-video_features_file', default='../data/features_video_pca.npz')
    #parser.add_argument('-video_descriptions_file', default='../data/video_descriptions.pickle')
    #parser.add_argument('-vocab_file', default='../data/vocab.pickle')

    # ActivityNet
    parser.add_argument('-video_features_file', default='../activitynet/activitynet_features.hdf5')
    parser.add_argument('-video_descriptions_file', default='../activitynet_descriptions.pkl')
    parser.add_argument('-vocab_file', default='../activitynet_vocab.pkl')
    parser.add_argument('-video_descriptions_csv', default='../data/video_description.csv')
    parser.add_argument('-gpu_id', type=int, default=0)

    opt = parser.parse_args()

    opt.device = torch.device('cuda:' + str(opt.gpu_id))# if torch.cuda.is_available() else 'cpu')

    # Create Data Loader
    trainloader = DataLoader(opt=opt, train=True)
    evalloader = DataLoader(opt, train=False)

    # Create Model
    model = UTransformer(num_vocab = trainloader.vocab.idx, embedding_size = opt.vid_feat_size, hidden_size = opt.d_model, num_layers = opt.n_layers, num_heads = opt.heads, total_key_depth = opt.d_model, total_value_depth = opt.d_model, filter_size = 2048).to(opt.device)

    # Use DataParallel
    model = nn.DataParallel(model)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-6)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    train_model(model, opt, trainloader, evalloader)

if __name__ == "__main__":
    main()
