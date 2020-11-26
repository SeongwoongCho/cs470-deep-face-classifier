import random
import os
import numpy as np
import torch
import torch.nn as nn

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x,y,beta = 1.0, use_cuda=True):
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size()[0]).cuda()
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def cross_entropy(weights = None):
    def _cross_entropy(input, target, size_average=True, weights = weights):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean
        Examples::
            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)
            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        if weights is None:
            weights = torch.Tensor([1]*len(target.shape[1]))
        weights = torch.Tensor(weights)
        weights = weights / torch.sum(weights,dim = 0, keepdim=True)
        weights = weights.cuda()
        logsoftmax = nn.LogSoftmax()
        if size_average:
            return torch.mean(torch.sum(-weights*target * logsoftmax(input), dim=1))
        else:
            return torch.sum(torch.sum(-weights*target * logsoftmax(input), dim=1))
    return _cross_entropy

def to_onehot(label,num_classes=2):
    return np.eye(num_classes)[label]

def label_smoothing(onehot,eps = 0.01):
    assert onehot.ndim == 1
    d = len(onehot)
    return (1-eps)*onehot + (eps/(d-1))*(1-onehot)

def eval_metric(preds,trues,n_classes = 3):
    acc = 0
    Precisions = []
    Recalls = []
    
    for i in range(n_classes):
        tp,fp,tn,fn = 0,0,0,0
        for p,t in zip(preds,trues):
            if p == t:
                if t==i:
                    tp +=1
                else:
                    tn +=1
                acc +=1
            else:
                if t==i:
                    fn +=1
                else:
                    fp +=1
        Recalls.append(tp/(tp+fn))
        Precisions.append(tp/(tp+fp))
    
    acc = acc / (len(preds)*n_classes)
    precision = np.mean(Precisions)
    recall = np.mean(Recalls)
    f1 = 2*(precision*recall)/(precision+recall)
    return acc,precision,recall,f1