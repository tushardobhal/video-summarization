import torch
from torchtext import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    # if opt.device == 0:
    #   np_mask = np_mask.cuda()
    return np_mask
