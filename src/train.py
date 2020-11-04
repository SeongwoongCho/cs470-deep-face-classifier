import torch
import torch.nn as nn
import numpy as np
import os

from torch.optim import AdamW,SGD
from torch.utils import data
from utils import *
from warmup_scheduler import GradualWarmupScheduler
from models import get_cls_model
from dataloader import get_datas,Dataset
from tqdm import tqdm
from importify import Serializable

seed_everything(42)

class Config(Serializable):
    def __init__(self):
        super(Config, self).__init__()
        ## training mode
        self.exp_name = 'baseline'
        self.coeff = 0 ## efficientnet model parameters
        
        ## training parameters
        self.learning_rate = 4e-3
        self.batch_size = 32
        self.n_epoch = 100
        self.optim = 'adamw'
        self.weight_decay = 1e-5
        self.num_workers = 16
        self.warmup = 1

        ## other hyper parameters
        self.mixup_prob = 0.
        self.mixup_alpha = 0.
        self.cutmix_prob = 0.
        self.cutmix_beta = 0.
        self.label_smoothing = 0.01
        
        ## training_options
        self.amp = False
        
config = Config()
config.parse()

if config.amp:
    from apex import amp

save_path = './logs/{}'.format(config.exp_name)
os.makedirs(save_path,exist_ok=True)
saved_status = config.export_json(path=os.path.join(save_path,'saved_status.json'))

train_x,train_y,valid_x,valid_y = get_datas()
train_dataset = Dataset(train_x,train_y, is_train=True, ls_eps = config.label_smoothing)
valid_dataset = Dataset(valid_x,valid_y, is_train=False, ls_eps = 0)
train_loader=data.DataLoader(dataset=train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True)
valid_loader=data.DataLoader(dataset=valid_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=False)

model = get_cls_model(config.coeff)
model.cuda()

if not config.amp:
    model= nn.DataParallel(model)
        
if config.optim == 'adamw':
    optimizer = AdamW(model.parameters(),lr=config.learning_rate,weight_decay = config.weight_decay)
elif config.optim == 'sgd':
    optimizer = SGD(model.parameters(),lr=config.learning_rate,weight_decay = config.weight_decay, momentum = 0.9, nesterov = True)

if config.amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model= nn.DataParallel(model)
    
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = config.n_epoch*len(train_loader))
if config.warmup:
    scheduler = GradualWarmupScheduler(optimizer, multiplier = 1, total_epoch = config.warmup*len(train_loader), after_scheduler = scheduler)
criterion = cross_entropy()

best_acc=np.inf
step = 0
for epoch in range(config.n_epoch):
    train_loss=0
    optimizer.zero_grad()
    model.train()

    progress_bar = tqdm(train_loader)
    for idx,data in enumerate(progress_bar):
        x = data['x'].cuda()
        y = data['y'].cuda()
        
        if np.random.uniform(0,1) < config.mixup_prob:
            x,y_a,y_b,lam = mixup_data(x,y,config.mixup_alpha)
            pred = model(x)
            loss = criterion(pred,y_a*lam + y_b*(1-lam))
        elif np.random.uniform(0,1) < config.cutmix_prob:
            x,y_a,y_b,lam = cutmix_data(x,y,config.cutmix_beta)
            pred = model(x)
            loss = criterion(pred,y_a*lam + y_b*(1-lam))
        else:
            pred = model(speech)
            loss = criterion(pred,speech_label)
        if config.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss+=loss.item()/len(train_loader)
        scheduler.step(step)
        step +=1
        progress_bar.set_description(
            'Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. current loss: {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, config.n_epoch, idx + 1, len(train_loader), loss.item()))

    valid_loss=0
    valid_acc=0
    model.eval()
    for idx,data in enumerate(tqdm(valid_loader)):
        x = data['x'].cuda()
        y = data['y'].cuda()
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred,y)
        valid_loss+=loss.item()/len(valid_loader)
        pred=pred.detach().max(1)[1]
        y = y.detach().max(1)[1]
        acc = pred.eq(y.view_as(pred)).sum().item() / len(pred)
        valid_acc+=acc/len(valid_loader)

    torch.save(model.module.state_dict(),os.path.join(save_path,'%d_best_%.4f.pth'%(epoch,valid_loss)))
    print("Epoch [%d]/[%d] train_loss: %.6f valid_loss: %.6f valid_acc:%.6f"%(
    epoch,config.n_epoch,train_loss,valid_loss,valid_acc))
