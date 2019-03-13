import argparse
import time
import torch
import torch.nn as nn
from Models import get_model
from tqdm import tqdm
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import pickle
import argparse
from Models import get_model
from Beam import modified_beam
from torch.autograd import Variable
import re

from DataLoader import DataLoader
from Vocabulary import Vocabulary
from utransformer import UTransformer

from nltk.translate import bleu_score

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', default='weights')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=27) # <start> + 25 tokens + <end>
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')

    # DataLoader
    parser.add_argument('-num_train_set', type=int, default=1300)
    parser.add_argument('-video_features_file', default='../data/features_video_rgb_pca_i3d.npz')
    parser.add_argument('-video_descriptions_file', default='../data/video_descriptions.pickle')
    parser.add_argument('-vocab_file', default='../data/vocab.pickle')
    parser.add_argument('-video_descriptions_csv', default='../data/video_description.csv')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-vid_feat_size', type=int, default=512)

    opt = parser.parse_args()

    opt.device = torch.device('cuda:0')# if torch.cuda.is_available() else 'cpu')
   
    trainloader = DataLoader(opt=opt, train=True) 
    evalloader = DataLoader(opt=opt, train=False)
    #model = get_model(opt, opt.vid_feat_size, evalloader.vocab.idx)

    # Create Model
    model = UTransformer(num_vocab = trainloader.vocab.idx, embedding_size = opt.vid_feat_size, hidden_size = opt.d_model, num_layers = opt.n_layers, num_heads = opt.heads, total_key_depth = opt.d_model, total_value_depth = opt.d_model, filter_size = 2048).to(opt.device)

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.load_weights + '/ut_msvd'))

    model.vocab = trainloader.vocab
    model.eval()
    
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_3 = 0.0
    bleu_4 = 0.0
    count_bleu = 0
    for i, (src, trg, vid_names) in enumerate(evalloader.batch_data_generator()):
        opt.batch_size = src.shape[0] # actual batch size might be smaller in last batch iter
        #print("GT:- ", evalloader.get_words_from_index(trg[0]))
        sentences = modified_beam(model, src, trg, opt)
        #print("GT:- ", trainloader.get_sentence_from_tensor(trg[0]))
        #print("Pred:- ", evalloader.get_words_from_index(sentences[0]))
        for sent_id, sentence in enumerate(sentences):
            word_sent = evalloader.get_words_from_index(sentence)
            word_sent = " ".join(word_sent[1:word_sent.index("<end>")])
            ground_truth = [gt_sent.lower()[:-1] for gt_sent in evalloader.video_descriptions[vid_names[sent_id]]] # lower case, delete "." at the end
            
            print("Sent:- ", word_sent)
            bleu_1 += bleu_score.sentence_bleu(ground_truth,word_sent, weights = (1,0,0,0))
            bleu_2 += bleu_score.sentence_bleu(ground_truth,word_sent, weights = (1/2,1/2,0,0))
            bleu_3 += bleu_score.sentence_bleu(ground_truth,word_sent, weights = (1/3,1/3,1/3,0))
            bleu_4 += bleu_score.sentence_bleu(ground_truth,word_sent, weights = (1/4,1/4,1/4,1/4))
            count_bleu += 1
    print("BLEU-1: ", bleu_1 / count_bleu)
    print("BLEU-2: ", bleu_2 / count_bleu)
    print("BLEU-3: ", bleu_3 / count_bleu)
    print("BLEU-4: ", bleu_4 / count_bleu)

if __name__ == '__main__':
    main()
