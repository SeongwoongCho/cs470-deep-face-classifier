from efftorch.efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18,resnet34,resnet50
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

num_classes = 3
def get_cls_model(coeff):
    if coeff in [0,1,2,3,4,5,6,7]:
        model = EfficientNet.from_pretrained('efficientnet-b%d'%coeff)
        model._fc = nn.Linear(model._fc.in_features,num_classes)
    if coeff == 18:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,num_classes)
    if coeff == 34:
        model = resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,num_classes)
    if coeff == 50:
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features,num_classes)
    return model