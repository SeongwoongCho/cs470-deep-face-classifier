import torch
from dataloader import *
from torch.utils import data
from models import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from utils import eval_metric
import numpy as np

torch.backends.cudnn.benchmark = True
x,y = get_test_datas()
test_dataset = Dataset(x,y, is_train=False, ls_eps = 0)
test_loader=data.DataLoader(dataset=test_dataset,batch_size=128,num_workers=10,shuffle=False)

coeffs = [50,0,0,0,4,4]
weight_paths = ['./logs/resnet50_non/10_best_0.0862_0.6842.pth',
                './logs/effb0_non/16_best_0.0782_0.7220.pth',
                './logs/effb0_cutmix/47_best_0.0664_0.7858.pth',
                './logs/effb0_cutmix_ls/40_best_0.0747_0.7766.pth',
                './logs/effb4_cutmix/59_best_0.0672_0.7730.pth',
                './logs/effb4_cutmix_ls/58_best_0.0778_0.7665.pth']
exp_names = ['resnet50',
            'efficientnet-b0',
            'efficientnet-b0 + cutmix',
            'efficientnet-b0 + cutmix + ls',
            'efficientnet-b4 + cutmix',
            'efficientnet-b4 + cutmix + ls']

models = []

for coeff,weight_path in zip(coeffs,weight_paths):
    model = get_cls_model(coeff)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model = model.eval()
    models.append(model)
    
## This result is checkpoint with best loss

for model,exp_name in zip(models,exp_names):
    trues = []
    preds = []
    for idx,data in enumerate(test_loader):
        x = data['x'].cuda()
        y = data['y']
        y = torch.argmax(y,dim = -1).detach().numpy()
        with torch.no_grad():
            pred = F.softmax(model(x))
            pred = torch.argmax(pred,dim = -1).cpu().detach().numpy()

        trues += y.tolist()
        preds += pred.tolist()
        
    acc, precision, recall, f1 = eval_metric(preds,trues)
    print("="*30)
    print(exp_name)
    print("accuracy: {} precision: {} recall: {} f1: {}".format(acc,precision,recall,f1))
    print()
    print("confusion matrix")
    print('    bald     bear     non-bear')
    print(confusion_matrix(trues,preds,normalize = "true"))
    print()

for temperature in [0.5,1,2]:
    trues = []
    preds = []
    for idx,data in enumerate(test_loader):
        x = data['x'].cuda()
        y = data['y']
        y = torch.argmax(y,dim = -1).detach().numpy()
        with torch.no_grad():
            pred1 = F.softmax(models[2](x)).pow(temperature)
            pred2 = F.softmax(models[4](x)).pow(temperature)
            pred1 = pred1/torch.sum(pred1,dim = -1, keepdim=True)
            pred2 = pred2/torch.sum(pred2,dim = -1, keepdim=True)

            pred = torch.argmax(pred1 + pred2,dim = -1).cpu().detach().numpy()

        trues += y.tolist()
        preds += pred.tolist()

    acc, precision, recall, f1 = eval_metric(preds,trues)
    print("="*30)
    print("Ensemble 2 best model with temperature value {}".format(temperature))
    print("accuracy: {} precision: {} recall: {} f1: {}".format(acc,precision,recall,f1))
    print()
    print("confusion matrix")
    print('    bald     bear     non-bear')
    print(confusion_matrix(trues,preds,normalize = "true"))
    print()