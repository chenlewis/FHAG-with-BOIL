from torchvision.models import resnet18, resnet50, densenet121, resnext101_32x8d, efficientnet_b0, efficientnet_b4, convnext_tiny
import sys
sys.path.append('../')
from models.basic_module import  BasicModule
from torch import nn
import torch
from torch.optim import Adam




class EfficientNetB0(BasicModule):
    def __init__(self, model_name='EfficientNetB0'):
        super(EfficientNetB0, self).__init__()
        self.model_name = model_name
        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('resnet50 model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('resnet50 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        

class EfficientNetB4(BasicModule):
    def __init__(self, model_name='EfficientNetB4'):
        super(EfficientNetB4, self).__init__()
        self.model_name = model_name
        self.model = efficientnet_b4(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1792, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('resnet50 model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('resnet50 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        

class ConvNeXtTiny(BasicModule):
    def __init__(self, model_name='ConvNeXtTiny'):
        super(ConvNeXtTiny, self).__init__()
        self.model_name = model_name
        self.model = convnext_tiny(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.LayerNorm((768,1,1), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('resnet50 model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('resnet50 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        


class ResNet18(BasicModule):
    def __init__(self, model_name='ResNet18'):
        super(ResNet18, self).__init__()
        self.model_name = model_name
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('resnet18 model freeze cnn weights')
            return Adam(self.model.fc.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('resnet18 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        

        
class ResNet50(BasicModule):
    def __init__(self, model_name='ResNet50'):
        super(ResNet50, self).__init__()
        self.model_name = model_name
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        
    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('resnet50 model freeze cnn weights')
            return Adam(self.model.fc.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('resnet50 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        

        

class ResNeXt101(BasicModule):
    def __init__(self, model_name='ResNeXt101'):
        super(ResNeXt101, self).__init__()
        self.model_name = model_name
        self.model = resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('resnext101 model freeze cnn weights')
            return Adam(self.model.fc.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('resnext101 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        


class DenseNet121(BasicModule):
    def __init__(self, model_name='DenseNet121'):
        super(DenseNet121, self).__init__()
        self.model_name = model_name
        self.model = densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
    def get_optimizer(self, lr, weight_decay, freeze=True):
        if freeze:
            print ('densenet121 model freeze cnn weights')
            return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
        else:
            print ('densenet121 model do not freeze cnn weights')
            return Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        
        
if __name__ == '__main__':
    model  = ConvNeXtTiny()
    data = torch.randn((5, 3, 224, 224))
    result = model(data)