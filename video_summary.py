#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

import os
from os.path import join
from glob import glob

from sklearn.decomposition import PCA
import skimage.io as io
from skimage.transform import resize
from matplotlib.pyplot import figure

import numpy as np
import cv2


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
c3d_pre_trained = 'pre_trained_weights/c3d_ucf101.pth'
data_dir = "data"
video_features_file = "features_video.npz"
max_frames_per_clip = 500
print(device)

use_pca = True


# In[ ]:


# http://www.cs.utexas.edu/users/ml/clamp/videoDescription/

class DataLoader:
    
    def __init__(self):
        super(DataLoader, self).__init__()
        self.pca = None
        self.n_components = 512
        self.pca = PCA(n_components=self.n_components)
        
    def create_data_dirs(self):
        clips = sorted(glob(join("youtube_clips", "*.avi")))
        
        num_clip = 1
        for clip in clips:
            print("Saving clip - {}/{} - {}".format(num_clip, len(clips), clip))
            
            cap = cv2.VideoCapture(clip)
            names = clip.split('/')[1].split('.')[0].split('_')
            if len(names) > 3:
                for j in range(1, len(names) - 2):
                    names[0] += "_" + names[j]
                names[1] = names[-2]
                names[2] = names[-1]
            
            dir_name = "{}/{}_{}_{}".format(data_dir, names[0], names[1], names[2])
            try:
                os.mkdir(dir_name)
            except OSError:  
                print ("Creation of the directory {} failed".format(names))
            
            i = 1
            while(True):
                ret, frame = cap.read()
                if ret == False:
                    break
                    
                file_name = "{}/{}.png".format(dir_name, i)
                cv2.imwrite(file_name, frame)
                i += 1
                
            cap.release()
            num_clip += 1
            
    def get_video_clip(self, clip_name, verbose=True):
        """
        Loads a clip to be fed to C3D for classification.
        
        Parameters
        ----------
        clip_name: str
            the name of the clip (subfolder in 'data').
        verbose: bool
            if True, shows all the frames (default=True)
            
        Returns
        -------
        Tensor
            a pytorch batch (num_batch, channels, frames, height, width)
        """

        clip = sorted(glob(join('data', clip_name, '*.png')))
        num_frames = len(clip)
        if len(clip) > 500:
            num_frames = max_frames_per_clip
        
        clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True, anti_aliasing=True, 
                                mode='reflect') for frame in clip])
        clip = clip[:num_frames, :, 44:44+112, :]  # crop centrally

        if verbose:
            clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, num_frames * 112, 3))
            figure(figsize = (20,2))
            io.imshow(clip_img.astype(np.uint8), interpolation='nearest')
            io.show()
    
        clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
        clip = np.expand_dims(clip, axis=0)  # batch axis
        clip = np.float32(clip)
        
        return torch.from_numpy(clip).to(device)
    
    def extract_video_features(self):
        clips = sorted(glob(join(data_dir, '*')))
        feature_dict = {}
        
        i = 1
        for clip in clips:
            clip_name = clip.split('/')[-1]
            print("{}/{} - {}".format(i, len(clips), clip_name)) 
            
            x = self.get_video_clip(clip_name, False)
            features = c3d(x).data.cpu().numpy()
                
            del x
            torch.cuda.empty_cache()
            
            if clip_name in feature_dict:
                feature_dict[clip_name].append(features)
            else:
                feature_dict[clip_name] = [features]
            
            i += 1
        
        self.write_features(feature_dict)
    
    def pca_transform(self, features):
        return self.pca.fit_transform(features)
        
    def write_features(self, features):
        np.savez(video_features_file, **features)


# In[ ]:


# https://github.com/DavideA/c3d-pytorch

class C3D(nn.Module):
    def __init__(self, pre_trained=True):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        if pre_trained:
            self.__load_pretrained_weights()

    def forward(self, x):
        
        with torch.no_grad():
            h = self.relu(self.conv1(x))
            h = self.pool1(h)
    
            h = self.relu(self.conv2(h))
            h = self.pool2(h)
    
            h = self.relu(self.conv3a(h))
            h = self.relu(self.conv3b(h))
            h = self.pool3(h)
    
            h = self.relu(self.conv4a(h))
            h = self.relu(self.conv4b(h))
            h = self.pool4(h)
    
            h = self.relu(self.conv5a(h))
            h = self.relu(self.conv5b(h))
            h = self.pool5(h)
    
            h = h.view(-1, 8192)
            h = self.fc6(h)

        return h
    
    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(c3d_pre_trained)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# In[ ]:


c3d = C3D(pre_trained=True)
c3d.to(device)
c3d.eval()

data_loader = DataLoader()


# In[ ]:


data_loader.extract_video_features()


# In[ ]:


# features = np.load(video_features_file)
# tot = 0
# for feature in features:
#     print(features[feature].shape)

