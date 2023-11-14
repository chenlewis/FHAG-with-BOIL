import sys
import os
import csv
from utils.metrics import ComputeMetric
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def get_img_dict(csv_path):
    print (csv_path)
    img_dict = defaultdict(dict)
    model_name = csv_path.split('/')[-1].split('.')[0]
    file = open(csv_path, 'r')
    reader = csv.reader(file)
    for i, data in enumerate(reader):
        if i == 0:
            continue
        # try to parse your own label, score, img_name 
        label = int(data[0])
        score = float(data[1])
        img_name = data[2].split('/')[-2]
        # -------------------------------------------------
        if not img_name in img_dict.keys():
            img_dict[img_name] = {'label': label, 'num': 1, 'score':[score]}
        else:
            if not img_dict[img_name]['label'] == label:
                print ('false 1')
            img_dict[img_name]['num'] += 1
            img_dict[img_name]['score'] += [score]
    file.close()
    print (len(img_dict))
    return img_dict, model_name

def get_thresh(img_dict, model_name):
    y_ture = np.array([])
    y_score = np.array([])
    for k, v in img_dict.items():
        if not len(v['score']) == v['num']:
            print ('false 2')
        score_averagy = sum(v['score']) / v['num']
        y_ture = np.append(y_ture, v['label']) 
        y_score = np.append(y_score, score_averagy)
    auc, eer, best_thresh = ComputeMetric(y_ture, y_score, isPlot=True,\
                                          model_name=model_name, fig_path='../results/')
    print ('model name: ', model_name)
    print ('auc: ', auc, ' eer: ', eer, ' best thresh: ', best_thresh)
    return best_thresh


def evaluate(csv_path, root = '../results'):
    csv_path = os.path.join(root, csv_path)
    img_dict, model_name = get_img_dict(csv_path)
    best_thresh = get_thresh(img_dict, model_name)
    

    fn_list = []
    fp_list = []
    for k, v in img_dict.items():
        score_average = sum(v['score']) / v['num']
        if v['label'] == 0:
            if score_average >= best_thresh:
                fn_list.append(k)
        if v['label'] == 1:
            if score_average <= best_thresh:
                fp_list.append(k)
    return fn_list, fp_list



def count_template(name_list):
    num_dict = defaultdict(int)
    for i in name_list:
        num_dict[int(i.split('.')[0].split('_')[0])] += 1
    plt.figure(figsize=(10,10))
    plt.bar(list(num_dict.keys()), list(num_dict.values()))
    plt.show()
    num_tuple = sorted(num_dict.items(), key = lambda k:-k[1])
    print ('most false in :')
    print (num_tuple[:7])

def count_printer(name_list):
    num_dict = defaultdict(int)
    for i in name_list:
        num_dict[i.split('.')[0].split('_')[1]] += 1
    num_dict = sorted(num_dict.items())
    plt.figure(figsize=(10,10))
    plt.bar([i[0] for i in num_dict], [i[1] for i in num_dict])
    plt.show()

def count_capture(name_list):
    num_dict = defaultdict(int)
    for i in name_list:
        num_dict[i.split('.')[0].split('_')[2]] += 1
    num_dict = sorted(num_dict.items())
    plt.figure(figsize=(10,10))
    plt.bar([i[0] for i in num_dict], [i[1] for i in num_dict])
    plt.show()

def false_analyze(name_list):
    name_list = sorted(name_list)
    name_len = len(name_list[0].split('/')[-1].split('_'))
    if name_len > 2:
        count_template(name_list)
        count_printer(name_list)
        count_capture(name_list)
    else:
        count_template(name_list)

        
def pair_analyze(origin_path, fag_path):
    _, origin_fp_list = evaluate(origin_path)
    _, fag_fp_list = evaluate(fag_path)
    
    origin_num_dict = defaultdict(int)
    for i in origin_fp_list:
        origin_num_dict[i.split('.')[0].split('_')[1]] += 1
    origin_num_dict = sorted(origin_num_dict.items())
    fag_num_dict = defaultdict(int)
    for i in fag_fp_list:
        fag_num_dict[i.split('.')[0].split('_')[1]] += 1
    fag_num_dict = sorted(fag_num_dict.items())
    
    if not [i[0] for i in origin_num_dict] == [i[0] for i in fag_num_dict]:
        print ('1111111111')
    
    key_list = [i[0] for i in origin_num_dict]
    origin_x = [i for i in range(len(key_list))]
    fag_x = [i+0.2 for i in origin_x]
    plt.figure(figsize=(10,10))
    plt.bar(origin_x, [i[1] for i in origin_num_dict], width=0.2, label='origin')
    plt.bar(fag_x, [i[1] for i in fag_num_dict], width=0.2, label='fag')
    plt.xticks([i+0.1 for i in origin_x] , key_list)
    plt.legend()
    plt.show()
    
    origin_num_dict = defaultdict(int)
    for i in origin_fp_list:
        origin_num_dict[i.split('.')[0].split('_')[2]] += 1
    origin_num_dict = sorted(origin_num_dict.items())
    fag_num_dict = defaultdict(int)
    for i in fag_fp_list:
        fag_num_dict[i.split('.')[0].split('_')[2]] += 1
    fag_num_dict = sorted(fag_num_dict.items())
    
    if not [i[0] for i in origin_num_dict] == [i[0] for i in fag_num_dict]:
        print ('1111111111')
    
    key_list = [i[0] for i in origin_num_dict]
    origin_x = [i for i in range(len(key_list))]
    fag_x = [i+0.2 for i in origin_x]
    plt.figure(figsize=(10,10))
    plt.bar(origin_x, [i[1] for i in origin_num_dict], width=0.2, label='origin')
    plt.bar(fag_x, [i[1] for i in fag_num_dict], width=0.2, label='fag')
    plt.xticks([i+0.1 for i in origin_x] , key_list)
    plt.legend()
    plt.show()



for csv_path in ['']:
    fn_list, fp_list = evaluate(csv_path)