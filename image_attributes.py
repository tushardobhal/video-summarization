#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision
import torchvision.models as models

import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os
from os.path import join
import glob
from glob import glob

import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# video data directory
data_dir = "data/"
# ResNet coco pre-trained weights
resnet_coco_weights = "pre_trained_weights/resnet_152_coco.pkl"
# coco data dir
coco_vocab_dir = "coco_data/vocab.pkl"
video_descriptions_file = "video_description.csv"
attr_net_weights = "pre_trained_weights/attr_net.pth"

# load every <image_skip_parameter> image
max_num_images_per_clip = 32
# use n most frequent words
num_most_common_words = 10
# square image size
image_size = 224
# Number of training videos
num_train_set = 1300

num_epochs = 100

print(device)


# In[ ]:


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


# In[ ]:


class DataLoader:
    def __init__(self, train=True):
        print("Loading Data Loader instance for train = {}".format(train))
        
        self.device = device
        self.data_path = data_dir
        self.video_descriptions_file = video_descriptions_file
        self.image_size = image_size
        self.max_num_images_per_clip = max_num_images_per_clip
        self.num_most_common_words = num_most_common_words
        
        self.num_train_set = num_train_set
        # load names of all the data directories
        train_test = sorted(glob(join(self.data_path, "*")))
        
        if train:
            self.names = train_test[:self.num_train_set]
        else:
            self.names = train_test[self.num_train_set:]
        
        # load all sentences in the dictionary with key as video_id
        print("Loading sentences for each video into dictionary...")
        self.video_descriptions = self.load_csv()
        
        # dictionary containing n most frequent words for each video
        self.most_freq_words = {}
        
        # load COCO dictionary and add absent words to vocabulary
        print("Loading words into vocabulary...")
        self.vocab = self.load_full_vocab()
        
        self.index = 0

        self.data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        print("Data Loader initialized")
    
    def load_csv(self):
        desc = pd.read_csv(video_descriptions_file)
        desc = desc[(desc['Language'] == 'English')]
        desc = desc[['VideoID', 'Description']]
        desc_dict = {}
        for row in desc.iterrows():
            if row[1][0] in desc_dict:
                desc_dict[row[1][0]].append(str(row[1][1]))
            else:
                desc_dict[row[1][0]] = [str(row[1][1])]
        return desc_dict
    
    def load_full_vocab(self):
        stop_words = set(nltk.corpus.stopwords.words("english"))
        stop_words.add('-')
        vocab = pickle.load(open(coco_vocab_dir, 'rb'))
        
        for key in self.video_descriptions:
            sentences = ' '.join(self.video_descriptions[key])
            most_freq_words_with_count = nltk.FreqDist(word.lower().split('.')[0] 
                                    for word in sentences.split(' ') if word.lower() not in stop_words) 
            
            most_freq_words = []
            for word in most_freq_words_with_count.most_common(self.num_most_common_words):
                most_freq_words.append(word[0])
                if word[0] not in vocab.word2idx:
                    vocab.add_word(word[0])
            
            self.most_freq_words[key] = most_freq_words
        return vocab
    
    def image_loader(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = image.convert("RGB")
        image = self.data_transforms(image).float()
        image = image.unsqueeze(0)
        return image[0]

    def show(self, img):
        npimg = img.cpu().detach().numpy()
        npimg = np.transpose(npimg, (1,2,0))
        if npimg.shape[2] == 3:
            plt.imshow(npimg)
        else:
            plt.imshow(npimg[:,:,0], cmap='gray')

    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().detach().numpy()
        plt.figure(figsize=(10,2))
        plt.imshow(np.transpose(npimg, (1, 2, 0)), aspect='auto')
    
    def get_video_id(self, clip_name):
        names = clip_name.split('/')[1].split('.')[0].split('_')
        if len(names) > 3:
            for j in range(1, len(names) - 2):
                names[0] += "_" + names[j]
        return names[0]    
            
    def get_video_descriptions(self, clip_name):
        return self.video_descriptions[self.get_video_id(clip_name)]
    
    def get_vocab(self):
        return self.vocab
    
    def get_words_from_index(self, tensor):
        words_list = []
        for idx in tensor.data.cpu().numpy():
            words_list.append(self.vocab.idx2word[idx])
        return words_list
    
    def data_generator(self):

        while True:
            x = []
            unsorted_clip = glob(join(self.names[self.index], '*.png'))
            clip = sorted(unsorted_clip, key=lambda x: float(x.split('/')[-1].split('.')[0]))
            
            image_skip_parameter = int(np.floor(len(clip)/self.max_num_images_per_clip))
            for i in range(0,len(clip),image_skip_parameter):
                x.append(self.image_loader(clip[i]))
                if len(x) == self.max_num_images_per_clip:
                    break
            
            target = torch.zeros(len(self.vocab), dtype=torch.float32)
            
            video_id = self.get_video_id(clip[0])
            for word in self.most_freq_words[video_id]:
                target[self.vocab.word2idx[word]] = 1
              
            self.index += 1
            if self.index >= len(self.names):
                self.index = 0
            yield video_id, torch.stack(x).to(device), target.to(device)


# In[ ]:


train_loader = DataLoader()
test_loader = DataLoader(train=False)


# In[ ]:


# # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py

# class ResNetPreTrained(nn.Module):
#     def __init__(self):
#         """Load the pretrained ResNet-152 and replace top fc layer."""
#         super(ResNetPreTrained, self).__init__()
        
#         resnet = models.resnet152(pretrained=False)
#         modules = list(resnet.children())[:-1]  
#         self.resnet = nn.Sequential(*modules)
#         self.linear = nn.Linear(resnet.fc.in_features, 256)
#         self.bn = nn.BatchNorm1d(256, momentum=0.01)
        
#     def forward(self, images):
#         """Extract feature vectors from input images."""
#         with torch.no_grad():
#             features = self.resnet(images)
#         return features


# In[ ]:


# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

class WordAttributeGenerator(nn.Module):
    def __init__(self, vocab_size, pre_trained):
        super(WordAttributeGenerator, self).__init__()
        
        attr_net = models.resnet50(pretrained=pre_trained)
        modules = list(attr_net.children())[:-1]      
        self.attr_net = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 5000)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(5000, vocab_size)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.attr_net(images)
        features = features.view(features.size()[0], -1)
        weights = self.softmax(features)
        features = torch.sum(weights * features, dim=0).view(1, -1)
        features = self.tanh(self.fc1(features))
        features = self.fc2(features)
        return features.view(-1)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)


# In[ ]:


# # https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

# class WordAttributeGenerator(nn.Module):
#     def __init__(self, vocab_size, pre_trained):
#         super(WordAttributeGenerator, self).__init__()
        
#         resnet_pretrained = ResNetPreTrained()
#         resnet_pretrained.load_state_dict(torch.load(resnet_coco_weights))
#         modules = list(resnet_pretrained.children())[:-2]
#         self.resnet_pretrained = nn.Sequential(*modules)
#         self.fc1 = nn.Linear(2048, 5000)
#         self.tanh = nn.Tanh()
#         self.fc2 = nn.Linear(5000, vocab_size)
#         self.softmax = nn.Softmax(dim=0)
        
#     def forward(self, images):
#         """Extract feature vectors from input images."""
#         features = self.resnet_pretrained(images)
#         features = features.view(features.size()[0], -1)
#         weights = self.softmax(features)
#         features = torch.sum(weights * features, dim=0).view(1, -1)
#         features = self.tanh(self.fc1(features))
#         features = self.fc2(features)
#         return features.view(-1)
    
# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
#     elif classname.find('BatchNorm2d') != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
#         torch.nn.init.constant_(m.bias.data, 0.0)


# In[ ]:


attr_net = WordAttributeGenerator(len(train_loader.get_vocab()), pre_trained=True).to(device)
attr_net.train()
attr_net.apply(weights_init_normal)


# In[ ]:


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(attr_net.parameters())


# In[ ]:


def validate():
    with torch.no_grad():
        vid, x, target = next(test_loader.data_generator())
        features = attr_net(x)
        loss = criterion(features, target)
        
    predicted = test_loader.get_words_from_index(torch.topk(features, num_most_common_words)[1])
    actual = test_loader.get_words_from_index(torch.topk(target, num_most_common_words)[1])
    print("Val Loss - {}, Video ID - {}, Output - {}, Target - {}".format(loss.item(), vid, predicted, actual))
    
# attr_net.load_state_dict(torch.load(attr_net_weights))
# attr_net.eval()
# validate()


# In[ ]:


sample_interval = 25

try:
    attr_net.load_state_dict(torch.load(attr_net_weights))
except:
    print("The saved attr_net model file does not exist")
    
for epoch in range(0, num_epochs):
    for i in range(num_train_set):
            
        _, x, target = next(train_loader.data_generator())

        optimizer.zero_grad();
        features = attr_net(x)
        loss = criterion(features, target)
        loss.backward()
        optimizer.step()

        # Print statistics and save checkpoints
        print("\r[Epoch %d/%d] [Batch %d/%d] [Train loss: %f]" %
                                                        (epoch, num_epochs, i, num_train_set, loss.item()))

        if i % sample_interval == 0:
            torch.save(attr_net.state_dict(), attr_net_weights)
            validate()

