from config import opt
import os
import torch as t
import models
from data.dataset import D2CNN
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from utils.metrics import ComputeMetric
from tqdm import tqdm
import numpy as np
import csv
import random


t.backends.cudnn.benchmark = True
t.multiprocessing.set_sharing_strategy('file_system')


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@t.no_grad() # pytorch>=0.5
def test(opt, model_name):
    # configure model
    print ('begin test, best_model_name: ', model_name)
    model = getattr(models, opt.model)(opt.model_name).eval()
    model.load(model_name)
    model.to(opt.device)

    confusion_matrix = meter.ConfusionMeter(2)
    
    # data
    if opt.frequency:
        test_data = D2CNN('YOUR_FREQUENCY_CSV_PATH', test=True, frequency=True)
    else:
        test_data = D2CNN(opt.csv_root, test=True)
        
    test_data = D2CNN(opt.csv_root, test=True, frequency=opt.frequency)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data, path, first, printer, second, label) in \
    enumerate(tqdm(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        confusion_matrix.add(score.detach(), label.type(t.LongTensor))
        probability = t.nn.functional.softmax(score,dim=1)[:,1].detach().tolist()   
        
        # write your own csv detail
        batch_results = [(label_.item(), probability_, path_, first_, printer_, second_) 
                         for label_, probability_, path_, first_, printer_, second_
                         in zip(label, probability, path, first, printer, second)]
        # -------------------------------------------------

        results += batch_results
    write_csv(results, opt.result_file)
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print ('test accurancy: ', accuracy)

    return results


def write_csv(results, save_path='results/result.csv'):
    file = open(save_path, 'w')
    writer = csv.writer(file)
    # write your own csv head
    writer.writerow(['label', 'probability', 'path', 'first', 'printer', 'second'])
    # -------------------------------------------------
    writer.writerows(results)
    file.close()

    
    
def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    if opt.use_seed:
        setup_seed(opt.seed)
        print ('set seed suceess, seed: ', opt.seed)
    
    # step1: configure model
    model = getattr(models, opt.model)(opt.model_name)
    if opt.load_model_path:
        model.load(opt.load_model_path)
        print ('load model weights success: ', opt.load_model_path)
    model.to(opt.device)

    # step2: data
    train_data = D2CNN(opt.csv_root, train=True, transforms=opt.transforms, frequency=opt.frequency)
    if opt.frequency:
        val_data = D2CNN(['YOUR_FREQUENCY_CSV_PATH'], val=True, frequency=True)
    else:
        val_data = D2CNN(opt.csv_root, val=True)
        
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss(weight=opt.loss_weight)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay, opt.freeze)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    best_Valacc = 0

    # train
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data, path, first, printer, second, label) in \
        enumerate(tqdm(train_dataloader)):

            # train model 
            input = data.to(opt.device)
            target = label.to(opt.device)
            
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
#             loss = model(input, target)
            loss.backward()
            optimizer.step()
            
            
            # meters update and visualize
            loss_meter.add(loss.item())
#             detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach()) 

            if (ii + 1)%opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)
        
        prefix = os.path.join('checkpoint', model.model_name)
        if not os.path.exists(prefix):
                os.makedirs(prefix)
        model_name = os.path.join(prefix , opt.env+'_Epoch_'+str(epoch)+'_Valacc_'+str(val_accuracy)+'.pth')            
        model.save(name=model_name)
        
        if val_accuracy > best_Valacc:
            best_model_name = model_name
            best_Valacc = val_accuracy
        
        vis.plot('val_acc',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                            epoch = epoch,loss = loss_meter.value()[0],
            val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        
        # update learning rate
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]
        
    test(opt, best_model_name)
    
    

    
    

@t.no_grad()
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, path, first, printer, second, label) \
    in enumerate(tqdm(dataloader)):
        val_in = val_input.to(opt.device)
        score = model(val_in)
        confusion_matrix.add(score.detach(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy




if __name__=='__main__':
    import fire
    fire.Fire()