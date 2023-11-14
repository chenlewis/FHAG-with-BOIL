from torchvision.models import vit_b_16, ViT_B_16_Weights, swin_b, Swin_B_Weights
import sys
sys.path.append('../')
from models.basic_module import  BasicModule
from torch import nn
import torch
from torch.optim import Adam
from transformers import BeitForImageClassification
        

class BeiTB(BasicModule):
    def __init__(self, model_name='BeiTB'):
        super(BeiTB, self).__init__()
        self.model_name = model_name
        self.model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
        self.model.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x).logits

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('BeiTB model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('BeiTB model freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    
class ViTB16(BasicModule):
    def __init__(self, model_name='ViTB16'):
        super(ViTB16, self).__init__()
        self.model_name = model_name
        self.model = vit_b_16(weights=ViT_B_16_Weights)
        self.model.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('ViTB16 model freeze cnn weights')
            return Adam(self.model.heads.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('ViTB16 model freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        
class SwinB(BasicModule):
    def __init__(self, model_name='SwinB'):
        super(SwinB, self).__init__()
        self.model_name = model_name
        self.model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.model.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('SwinB model freeze cnn weights')
            return Adam(self.model.head.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('SwinB model freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        

        
        
        
if __name__ == '__main__':
    pass