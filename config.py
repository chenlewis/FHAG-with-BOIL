# coding:utf8
import warnings
import torch as t
import torchvision.transforms as T
import os

class DefaultConfig(object):
    env = 'ResNet18'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'ResNet18'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    model_name = None
    
    
    use_seed = False
    seed = 15150213

    # put your csv path here
    csv_list = ['']
    
    
    csv_index = 0

    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    result_file = 'results/result1.csv'
    fig_path = None

    max_epoch = 20
    lr = 0.0001  # initial learning rate
    lr_decay = 1  # when val_loss increase, lr = lr*lr_decay  default 0.5
    device = t.device('cpu')
    device_index = 0
    loss_weight = t.Tensor([1, 1]).to(device)
    weights = 1,1
    weight_decay = 0e-5  # 损失函数
    isplot = False
    freeze = True
    transform = False
    transforms = None
    frequency = False

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device(opt.device_index) if opt.use_gpu else t.device('cpu')
        opt.result_file = os.path.join('results', opt.env+'.csv')
        if opt.model_name is None:
            opt.model_name = opt.env
        opt.env = opt.env.replace('_', '-')
        opt.loss_weight = t.Tensor([ int(opt.weights[0]), int(opt.weights[1]) ]).to(opt.device)
        
        if type(opt.csv_index) is int:
            opt.csv_root = [opt.csv_list[int(opt.csv_index)]]
        else:
            opt.csv_root = [opt.csv_list[int(i)] for i in opt.csv_index]
            
        if opt.transform:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            opt.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.RandomAutocontrast(),
                T.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5)),
                T.ToTensor(),
                normalize
            ])


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
