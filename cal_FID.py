import os
from tqdm.notebook import tqdm
import torch as t
from torch.utils.data import DataLoader
import numpy as np
import csv
from PIL import Image
import random
from torchvision import transforms as T
from torch.utils import data
from scipy import linalg
from torchvision import models
import torch.nn as nn


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)
    
def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)



class DefaultDataset(data.Dataset):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.ReadCsv()
        
        height, width = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transforms = T.Compose([
            T.Resize([height, width]),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
            
    def __getitem__(self, index):
        img_path = self.datapath[index]
        img = Image.open(img_path)
        img_t = self.transforms(img)
        return img_t
          
    def __len__(self):
        return len(self.datapath)
    
    
    def ReadCsv(self):
        self.datapath = []
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            for i, data in enumerate(reader):
                if 'D2_patch/' in data[0] or 'D2_patch_fag2/' in data[0]:
                    img_path = os.path.join('/home/data/lbk/dataset', data[0])
                else:
                    img_path = os.path.join('/home/data1/lbk/dataset', data[0])
                self.datapath.append(img_path)


def get_eval_loader(csv_path):
    test_data = DefaultDataset(csv_path)
    test_dataloader = DataLoader(test_data,batch_size=128,shuffle=False,\
                                 num_workers=4,pin_memory=True,drop_last=False)
    return test_dataloader



def calculate_fid_given_csv(csv1, csv2, device_index=0):
    csv_list = [csv1, csv2]
    device = t.device(device_index)
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(csv_path) for csv_path in csv_list]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            actv = inception(x.to(device))
            actvs.append(actv.cpu().detach())
        actvs = t.cat(actvs, dim=0).numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


csv1 = './data/D2_recap_ink.csv'
csv2 = './data/Cycle_D2Generate_ink.csv'

fid_value = calculate_fid_given_csv(csv1, csv2, device_index=1)
print('FID: ', fid_value)



csv1 = './data/D2_recap_las.csv'
csv2 = './data/Cycle_D2Generate_las.csv'

fid_value = calculate_fid_given_csv(csv1, csv2, device_index=1)
print('FID: ', fid_value)