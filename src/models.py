from efficientnet_pytorch import EfficientNet

def get_cls_model(coeff):
    model = EfficientNet.from_pretrained('efficientnet-b%d'%coeff)
    return model