from torch.utils import data
from transforms import get_transform
from utils import seed_everything, to_onehot, label_smoothing
import cv2

seed_everything(42)

num_classes = 6

def get_datas():
    """
    x : list of img file path
    y : list of label(integer)
    """
    
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    
    return train_x,train_y,valid_x,valid_y

class Dataset(data.Dataset):
    def __init__(self,X,y,is_train = True, ls_eps = 0):
        self.X = X
        self.y = y
        self.ls_eps = 0
        self.is_train = is_train
        self.transform = get_transform(is_train)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        x = self.X[idx]
        y = self.y[idx]
        x = cv2.imread(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = self.transform(image = x)["image"]
        x = np.rollaxis(x,-1,0) # H,W,C -> C,H,W

        y = to_onehot(y,num_classes)
        y = label_smoothing(y,self.ls_eps)

        data = {}
        data['x'] = torch.from_numpy(x.astype('float32'))
        data['y'] = torch.from_numpy(y.astype('float32'))
        return data