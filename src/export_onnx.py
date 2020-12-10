import torch 
import torch.nn as nn
import torch.nn.functional as F
from models import get_cls_model

class Transposing(nn.Module):
    def __init__(self):
        super(Transposing,self).__init__()
    def forward(self,input):
        input = input/255
        input = input.transpose(0,3,1,2)
        return input

class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape(1,3,1,1))
        self.register_buffer('std', torch.Tensor(std).reshape(1,3,1,1))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean
        std = self.std
        return (input - mean) / std

class EnsembleModel(nn.Module):
    def __init__(self,models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
#         self.Transposer = Transposing()
        self.normalizer = Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #self.temperature = 0.5
    def forward(self,x):
#         x = self.Transposer(x)
        x = self.normalizer(x)
        preds = []
        for model in self.models:
            #pred = F.softmax(model(x)).pow(self.temperature)
            pred = F.softmax(model(x)) ## do not apply temperature sharpening due to onnxjs issue
            pred = pred / torch.sum(pred,dim = -1, keepdim = True)
            preds.append(pred.unsqueeze(0))
        preds = torch.mean(torch.cat(preds, dim = 0),dim = 0)
        return preds
        
        
print("exporting...")
with torch.no_grad():
    model1 = get_cls_model(0)
#     model2 = get_cls_model(4)
    model1.set_swish(False)
#     model2.set_swish(False)
    model1.onnx = True
#     model2.onnx = True
    model1.load_state_dict(torch.load('./logs/effb0_cutmix/47_best_0.0664_0.7858.pth'))
#     model2.load_state_dict(torch.load('./logs/effb4_cutmix/41_best_0.0662_0.7735.pth'))
    #model = EnsembleModel([model1,model2])
    
    model = EnsembleModel([model1])
    model.eval()
    # Dummy input for ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
#     dummy_input = torch.randn(1,224,224,3)
    # Export with ONNX
    torch.onnx.export(model, dummy_input, "./logs/onnx/export_model_light.onnx", verbose=True)

print("complete exporting!")
print()
# Test export
print("check export file...")
import onnx

model = onnx.load("./logs/onnx/export_model_light.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

print("complete checking!")
print()