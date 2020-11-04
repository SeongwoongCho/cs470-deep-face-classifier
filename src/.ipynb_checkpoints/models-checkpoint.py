from efficientnet_pytorch import EfficientNet
import torch.nn as nn

num_classes = 6
def get_cls_model(coeff):
    model = EfficientNet.from_pretrained('efficientnet-b%d'%coeff)
    model._fc = nn.Linear(model._fc.in_features,num_classes)
    return model