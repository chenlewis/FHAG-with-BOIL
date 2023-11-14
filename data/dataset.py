# coding:utf8
import os
from PIL import Image, ImageFilter
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import csv
import random
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from io import BytesIO

class D2CNN(data.Dataset):
    def __init__(self, csv_root_list, data_root='/YOUR_DATA_PATH/', transforms=None, train=False, val=False, test=False, whole=False, 
                 frequency=False, gen=None):
        self.train = train
        self.val   = val
        self.test  = test
        self.frequency = frequency
        self.gen = gen
        if train:
            self.split = 'train'
        elif val:
            self.split = 'val'
        elif test:
            self.split = 'test'
        elif whole:
            self.split = 'whole'
        else:
            print ('False dataset Type input!')
        print ('dataset Type: ', self.split)
        self.csv_root_list = csv_root_list
        self.data_root = data_root
        self.ReadCsv()
        
        if transforms is None:
            print ('transform is None, use default transform!')
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])
        else:
            print ('transfom is not None, use setting!')
            self.transforms = transforms
            
        self.totensor = T.ToTensor()
            
    def __getitem__(self, index):
        img_path = self.datapath[index]
        first = self.first[index]
        printer = self.printer[index]
        second = self.second[index]
        label = self.label[index]
        
        img = Image.open(img_path)
        
        if self.frequency:
            img = self.RGB_fft(img)
            img_t = self.totensor(img).float()
#             print (246346347)
        else:
            img_t = self.transforms(img)
        
        return img_t, img_path, first, printer, second, label
          
    def __len__(self):
        return len(self.datapath)
    
    
    def RGB_fft(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img_b = img[:, :, 0]
        f_b = np.fft.fft2(img_b)
        fshift_b = np.fft.fftshift(f_b)

        img_g = img[:, :, 1]
        f_g = np.fft.fft2(img_g)
        fshift_g = np.fft.fftshift(f_g)

        img_r = img[:, :, 2]
        f_r = np.fft.fft2(img_r)
        fshift_r = np.fft.fftshift(f_r)

        fshift_origin_abs = np.stack((np.abs(fshift_b), np.abs(fshift_g), np.abs(fshift_r)), -1)
#         img_f = Image.fromarray(fshift_origin_abs)
        return fshift_origin_abs

    
    def ReadCsv(self):
        self.datapath = []
        self.first = []
        self.printer = []
        self.second = []
        self.label   = []
        for csv_path in self.csv_root_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for i, data in enumerate(reader):
                    if data[5] == self.split or self.split == 'whole':
                        img_path = os.path.join(self.data_root, data[0])
                        self.datapath.append(img_path)
                        img_label = 0 if data[1] == 'capture' else 1
                        self.label.append(img_label)

                        self.first.append(data[2])
                        self.printer.append(data[3])
                        self.second.append(data[4])


        if not len(self.datapath) == len(self.label):
            print ('false length corresponding!')
        else:
            self.data_len = len(self.datapath)
            print ('data length: ', self.data_len)

class D1CNN(data.Dataset):
    def __init__(self, csv_root_list, data_root='/YOUR_DATA_PATH/', transforms=None, train=False, val=False, whole=False):
        self.train = train
        self.val   = val
        if train:
            self.split = 'train'
        elif val:
            self.split = 'val'
        elif whole:
            self.split = 'whole'
        else:
            print ('False dataset Type input!')
        print ('dataset Type: ', self.split)
        self.csv_root_list = csv_root_list
        self.data_root = data_root
        self.ReadCsv()
        
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])
            
    def __getitem__(self, index):
        img_path = self.datapath[index]
        label = self.label[index]
        
        img = Image.open(img_path)
        img_t = self.transforms(img)
        return img_t, img_path, label
          
    def __len__(self):
        return len(self.datapath)
    
    
    def ReadCsv(self):
        self.datapath = []
        self.label   = []
        for csv_path in self.csv_root_list:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for i, data in enumerate(reader):
                    if data[3] == self.split or self.split == 'whole':
                        img_path = os.path.join(self.data_root, data[0])
                        self.datapath.append(img_path)
                        img_label = 0 if data[1] == 'legal' else 1
                        self.label.append(img_label)

        if not len(self.datapath) == len(self.label):
            print ('false length corresponding!')
        else:
            self.data_len = len(self.datapath)
            print ('data length: ', self.data_len)
            


class D2_certificate_test(data.Dataset):
    def __init__(self, csv_list, data_root='/home/data1/lbk/certificate', transforms=None, deteriorate=None, frequency=False):
        self.split = 'test all'
        print ('dataset Type: ', self.split)
        self.csv_list = csv_list
        self.data_root = data_root
        self.deteriorate = deteriorate
        self.frequency = frequency
        self.ReadCsv()
        
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])
        self.totensor = T.ToTensor()
            
    def __getitem__(self, index):
        img_path = self.datapath[index]
        label = self.label[index]
        
        img = Image.open(img_path)
        
        if not self.deteriorate is None:
            deteriorate_type = self.deteriorate.split('_')[0]
            if deteriorate_type == 'noise':
                img = self.gaussian_noise(img, 0, float('0.0' + self.deteriorate.split('_')[1]))
            elif deteriorate_type == 'blur':
                img = self.gaussian_blur(img, int(self.deteriorate.split('_')[1]))
            elif deteriorate_type == 'compress':
#                 print ('deter!!')
                img = self.compress_JPEG(img, int(self.deteriorate.split('_')[1]))
            else:
                print ('deteriorate false')
                
        if self.frequency:
            img = self.RGB_fft(img)
            img_t = self.totensor(img).float()
        else:
            img_t = self.transforms(img)
        return img_t, img_path, label
          
    def __len__(self):
        return len(self.datapath)
    
    def RGB_fft(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img_b = img[:, :, 0]
        f_b = np.fft.fft2(img_b)
        fshift_b = np.fft.fftshift(f_b)

        img_g = img[:, :, 1]
        f_g = np.fft.fft2(img_g)
        fshift_g = np.fft.fftshift(f_g)

        img_r = img[:, :, 2]
        f_r = np.fft.fft2(img_r)
        fshift_r = np.fft.fftshift(f_r)

        fshift_origin_abs = np.stack((np.abs(fshift_b), np.abs(fshift_g), np.abs(fshift_r)), -1)
#         img_f = Image.fromarray(fshift_origin_abs)
        return fshift_origin_abs
    
    
    def gaussian_blur(self, img, radius=2):
        img_blur = img.filter(ImageFilter.GaussianBlur(radius))
        return img_blur

    def gaussian_noise(self, img, mean=0, sigma=0.01):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = img / 255
        noise = np.random.normal(mean, sigma, img.shape)
        gaussian_out = img + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        gaussian_out = np.uint8(gaussian_out*255)
        img_noise = Image.fromarray(gaussian_out)
        return img_noise
    
    
    def compress_JPEG(self, img, quality):
        output_buffer = BytesIO()
        options = {'quality': quality}
        img.save(output_buffer, format='JPEG', **options)
        compressed_img = Image.open(output_buffer)
        return compressed_img

    def ReadCsv(self):
        self.datapath = []
        self.first = []
        self.printer = []
        self.second = []
        self.label   = []
        for csv_root in self.csv_list:
            with open(csv_root, 'r') as f:
                reader = csv.reader(f)
                for i, data in enumerate(reader):
                    img_path = os.path.join(self.data_root, data[0])
                    self.datapath.append(img_path)
                    img_label = 0 if data[1] == 'legal' else 1
                    self.label.append(img_label)
                
        if not len(self.datapath) == len(self.label):
            print ('false length corresponding!')
        else:
            self.data_len = len(self.datapath)
            print ('data length: ', self.data_len)              

    
if __name__ == '__main__':
    csv_root = ['']
    data_root = ''
    dataset = D2CNN(csv_root, data_root, whole=True)
    train_loader = DataLoader(dataset, batch_size=192,
                                  shuffle=True, num_workers=4)
    for ii, data in enumerate(tqdm(train_loader)):
        pass
