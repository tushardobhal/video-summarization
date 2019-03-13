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

from DataLoader import DataLoader
from Vocabulary import Vocabulary

import pdb

def train_model(model, opt, trainloader):
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
    
    for epoch in range(opt.epochs):
        print("epoch: ", epoch)

        total_loss = 0
        for i, (src, trg, vid_names) in enumerate(trainloader.batch_data_generator()):
          trg_input = trg[:, :-1] # not include the end of sentence
          src_mask, trg_mask = create_masks(src, trg_input, opt)

          preds = model(src, trg_input, src_mask, trg_mask)
          ys = trg[:, 1:].contiguous().view(-1)
          opt.optimizer.zero_grad()
          loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
          loss.backward()
          opt.optimizer.step()
          if opt.SGDR == True: 
              opt.sched.step()

          total_loss += loss.item()

        avg_loss = total_loss / (i + 1)
        epoch_time = (time.time() - start)
        print("%dm %ds: loss = %.3f\n" %(epoch_time//60, epoch_time%60, avg_loss))

        if epoch % opt.save_freq == 0:
          torch.save(model.state_dict(), 'weights/model_weights')
          cptime = time.time()
          print("model saved at epoch ", epoch)

    # save final weights
    torch.save(model.state_dict(), 'weights/model_weights')

def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('-src_data', required=True)
    # parser.add_argument('-trg_data', required=True)
    # parser.add_argument('-src_lang', required=True)
    # parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-printevery', type=int, default=1)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-vid_feat_size', type=int, default=512)
    parser.add_argument('-save_freq', type=int, default=5)
    # DataLoader
    parser.add_argument('-num_train_set', type=int, default=1300)
    parser.add_argument('-video_features_file', default='../data/features_video_rgb_pca_i3d.npz')
    parser.add_argument('-video_descriptions_file', default='../data/video_descriptions.pickle')
    parser.add_argument('-vocab_file', default='../data/vocab.pickle')
    parser.add_argument('-video_descriptions_csv', default='../data/video_description.csv')
    parser.add_argument('-gpu_id', type=int, default=0)
    
    opt = parser.parse_args()

    opt.device = torch.device('cuda:' + str(opt.gpu_id))# if torch.cuda.is_available() else 'cpu')
    
    # read data and create model
    # read_data(opt)
    # SRC, TRG = create_fields(opt)
    # opt.train = create_dataset(opt, SRC, TRG)
    # model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
#     opt.trainX, opt.trainY, opt.group_keys = load_data()
    trainloader = DataLoader(opt=opt, train=True)
    model = get_model(opt, opt.vid_feat_size, trainloader.vocab.idx)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    # if opt.load_weights is not None and opt.floyd is not None:
    #     os.mkdir('weights')
    #     pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
    #     pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt, trainloader)

    # if opt.floyd is False:
    #     promptNextAction(model, opt, SRC, TRG)

# load 2 video pieces with same lengths
# load text descriptions
def load_data():
    ''' 
    X (video features) is built dynamically.
    This function requires features_video.npz, raw_text.pkl, Y.pkl
    '''
    print("loading data ...")
    fs = np.load("features_video.npz")
    print(len(fs.files))
    with open('raw_text.pkl', 'rb') as f:
        raw_text = pickle.load(f)
    print(len(raw_text))

    X = dict()
    not_in_count = 0
    for i, row in tqdm(enumerate(raw_text)):
      try:
        feat = fs[row[0] + "_" + row[1] + "_" + row[2]]
      except: # text description has no corresponding video feature
        not_in_count += 1
        continue

      feat_len = feat.shape[0]
      if feat_len not in X.keys():
          X[feat_len] = []
      X[feat_len].append(feat)

    for key in X.keys():
      X[key] = np.array(X[key])
    #   np.save("X_" + str(key), X[key])

    for fname in os.listdir("."):
      if fname[:2] == "X_" and fname[-3:] == "npy":
        X[int(fname[2:-4])] = np.load(fname)

    #print(X.keys())
    with open('Y.pkl', 'rb') as f:
      Y = pickle.load(f)

    group_keys = np.array([*X])

    print("data loaded.")
    return X, Y, group_keys

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
