import torch
from Batch import *
import torch.nn.functional as F
import math
import pdb

def modified_beam(model, src, trg, opt):
    outputs, e_outputs, log_scores = init_vars(model, src, trg, opt)
    eos_tok = model.vocab.word2idx['<end>']
    src_mask = torch.ones(src.shape[:-1], dtype = torch.uint8).unsqueeze(-2).to(opt.device).repeat(opt.k,1,1)
    ind = None
    for i in range(2, opt.max_len):
        # print(i)
        trg_mask = nopeak_mask(i, opt)

        # pdb.set_trace()
        out = model.out(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask)).detach()

        out = F.softmax(out, dim=-1).detach()
    
        if i < opt.max_len - 1:
            outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k, opt)
        else: # last iter: get the best sentence instead of k best
            outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, 1, opt)

        # if (outputs==eos_tok).nonzero().size(0) == opt.k:
        #     alpha = 0.7
        #     div = 1/((outputs==eos_tok).nonzero()[:,1].type_as(log_scores)**alpha)
        #     _, ind = torch.max(log_scores * div, 1)
        #     ind = ind.data[0]
        #     break
    
    # if ind is None:
    #     length = (outputs[0]==eos_tok).nonzero()[0]
    #     return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    # else:
    #     length = (outputs[ind]==eos_tok).nonzero()[0]
    #     return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

    return outputs

def init_vars(model, src, trg, opt):
    # init_tok = TRG.vocab.stoi['<sos>']
    init_tok = model.vocab('<start>')
    # src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    src_mask = torch.ones(src.shape[:-1], dtype = torch.uint8).unsqueeze(-2).to(opt.device)
    e_output = model.encoder(src, src_mask).detach()
    
    outputs = torch.LongTensor([[init_tok]] * opt.batch_size)
    outputs = outputs.to(opt.device)
    
    trg_mask = nopeak_mask(1, opt)
    # trg_input = trg[:, :-1] # not include the end of sentence
    # src_mask, trg_mask = create_masks(src, trg_input, opt)

    # pdb.set_trace()
    out = model.out(model.decoder(outputs,
        e_output, src_mask, trg_mask)).detach()
    out = F.softmax(out, dim=-1)
    
    # pdb.set_trace()
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.log(probs).view(opt.batch_size * opt.k, -1)
    
    outputs = torch.zeros(opt.batch_size, opt.k, opt.max_len).long().to(opt.device)
    outputs[:, :, 0] = init_tok
    outputs[:, :, 1] = ix
    outputs = outputs.view(opt.batch_size * opt.k, -1) # batch_size = opt.batch_size * opt.k in k-beam search
    # pdb.set_trace()

    # copy e_outputs over k times 
    e_outputs = e_output.repeat(opt.k, 1,1)

    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k, opt):

    probs, ix = out[:, -1].data.topk(k)
    # log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    log_probs = torch.log(probs) + log_scores
    k_probs, k_ix = log_probs.view(opt.batch_size, -1).topk(k)
    
    # reshape outputs for select-k indexing
    prev_k = int(outputs.shape[0] / opt.batch_size)
    outputs = outputs.view(opt.batch_size, prev_k, -1)

    # create and reformat indexing arrays
    row = (k_ix // k).unsqueeze(-1)
    batch_idx = torch.tensor(np.arange(opt.batch_size), dtype = torch.long).view(-1,1,1)
    s_idx = torch.tensor(np.arange(i), dtype = torch.long).view(1,1,-1)
    batch_idx_i = batch_idx.squeeze(-1)

    if k == 1:
        last_outputs = torch.zeros(opt.batch_size, 1, opt.max_len).long().to(opt.device)
        last_outputs[:,:,:i] = outputs[batch_idx, row, s_idx]
        last_outputs[:,:,i] = ix.view(opt.batch_size, -1)[batch_idx_i, k_ix]
        last_outputs = last_outputs.view(opt.batch_size, -1)
        return last_outputs, log_scores

    outputs[:, :, :i] = outputs[batch_idx, row, s_idx]
    outputs[:, :, i] = ix.view(opt.batch_size, -1)[batch_idx_i, k_ix]

    

    # reshape batch_size back to opt.batch_size * k
    log_scores = k_probs.view(opt.batch_size * k,1)
    outputs = outputs.view(opt.batch_size * k, -1)

    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)

        out = model.out(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        if (outputs==eos_tok).nonzero().size(0) == opt.k:
            alpha = 0.7
            div = 1/((outputs==eos_tok).nonzero()[:,1].type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

