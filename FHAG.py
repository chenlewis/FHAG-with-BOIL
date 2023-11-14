import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm


file = open('YOUR_TRAINSET_PATCH_CSV_PATH', 'r')
reader = csv.reader(file)

scanner_list = ['HP_M176n', 'HP-M176n', 'EPSON_V850', 'EPSONV850_L805', 'EPSON_V850_L805']
phone_list = ['iphone6s', 'iPhone6s', 'oppo', 'opporeno', 'OPPO', 'oppo_reno']

fag_dict = defaultdict(list)
for i, data in enumerate(reader):
    label = data[1]
    mode = data[5]
    if label == 'recapture' and mode == 'train':
        path = data[0]
        printer = data[3]
        second = data[4]
        if printer == 'HP-M176n':
            if second in scanner_list:
                fag_dict['laser-scanner'].append([path, second])
            elif second in phone_list:
                fag_dict['laser-phone'].append([path, second])
            else:
                print ('false 1')
                print (second)
        elif printer == 'EPSON_L805':
            radius = data[7]
            if second in scanner_list:
                fag_dict['injek-scanner'].append([path, second, radius])
            elif second in phone_list:
                fag_dict['injek-phone'].append([path, second, radius])
            else:
                print ('false 2')
                print (second)
        else:
            print ('false 3')
            
file.close()
for k,v in fag_dict.items():
    print (k, ': ', len(v))



def RGB_fft(img):
    img_b = img[:, :, 0]
    f_b = np.fft.fft2(img_b)
    fshift_b = np.fft.fftshift(f_b)
    fshift_b_angel = np.angle(fshift_b)

    img_g = img[:, :, 1]
    f_g = np.fft.fft2(img_g)
    fshift_g = np.fft.fftshift(f_g)
    fshift_g_angel = np.angle(fshift_g)

    img_r = img[:, :, 2]
    f_r = np.fft.fft2(img_r)
    fshift_r = np.fft.fftshift(f_r)
    fshift_r_angel = np.angle(fshift_r)

    fshift_angel = np.stack((fshift_b_angel, fshift_g_angel, fshift_r_angel), -1)
    fshift_origin_abs = np.stack((np.abs(fshift_b), np.abs(fshift_g), np.abs(fshift_r)), -1)
    
    return fshift_origin_abs, fshift_angel

def RGB_ifft(img_fft):
    img_fft_b = img_fft[:, :, 0]
    img_fft_g = img_fft[:, :, 1]
    img_fft_r = img_fft[:, :, 2]
    img_aug_b = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_b)))
    img_aug_g = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_g)))
    img_aug_r = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_r)))
    img_aug = np.stack((img_aug_b, img_aug_g, img_aug_r), -1).astype(np.uint8)
#     print (type(img_aug[0][0][0]))
    return img_aug

def resize_img(img, second):
    if '176' in second:
        divide = [1,2]
        d = random.choice(divide)
        img = cv2.resize(img, (int(img.shape[1]/d), int(img.shape[0]/d)))
        return img, d
    elif '850' in second:
        divide = [2, 3, 4]
        d = random.choice(divide)
        img = cv2.resize(img, (int(img.shape[1]/d), int(img.shape[0]/d)))
        return img, d
    else:
        return img, 1
    
def show(img):
    plt.figure(figsize=(12,12))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    fshift_abs = np.log(np.abs(fshift))
    plt.imshow(fshift_abs, 'gray')
    plt.show()


def add_noise_laser(spectral, mask):
    height, width, depth = spectral.shape
    mask1 = 1 - mask
    
    alpha = np.random.uniform(0.5, 0.8, height * width * depth)
    alpha = alpha.reshape((height, width, depth))
    
    beta = np.random.normal(0, 1, height * width * depth)
    beta = beta.reshape((height, width, depth))
    
    spectral_mask_noise = (alpha * spectral + beta) * mask + spectral * mask1
    return spectral_mask_noise


def Laser_fag(patch_path_list, save_root, r_min_BOIL, r_max_BOIL, mode=1, aug_tail='fag.tif'):
    print ('laser patch len: ', len(patch_path_list))
    perm = [i for i in range(len(patch_path_list))]
    random.shuffle(perm)
    patch_path_list_perm = [patch_path_list[i] for i in perm]
    
    # define a mask with size same as your patch
    mask = np.zeros((224, 224, 3), np.uint8)
    center = (112, 112)
    
    print ('start laser img fft and mask')
    for i in tqdm(range(len(patch_path_list))):
        mask_temp = mask.copy()
        if mode == 1:
            r_min, r_max = r_min_BOIL[i], r_max_BOIL[i]
        elif mode == 2:
            r_min, r_max = r_min_BOIL[i], r_max_BOIL[i]
        else:
            print ('false mode input')
            
        cv2.circle(mask_temp, center, r_max, (1,1,1), -1)
        cv2.circle(mask_temp, center, r_min, (0,0,0), -1)
        mask1 = 1 - mask_temp

        img1 = cv2.imread(patch_path_list[i][0])
        img2 = cv2.imread(patch_path_list_perm[i][0])
#         img2, d = resize_img(img2, patch_path_list_perm[i][1])
        
        img1_fft_abs, img1_fft_angle = RGB_fft(img1)
        img2_fft_abs, img2_fft_angle = RGB_fft(img2)
#         img2_fft_abs = cv2.resize(img2_fft_abs, (224, 224))
        
        img1_fft_abs_mask = img1_fft_abs * mask_temp
        img1_fft_abs_mask1 = img1_fft_abs * mask1
        img2_fft_abs_mask = img2_fft_abs * mask_temp
        img2_fft_abs_mask1 = img2_fft_abs * mask1
        img1_fft_abs_mask1_noise = add_noise_laser(img1_fft_abs_mask1, mask1)
        
        max1 = np.max(img1_fft_abs_mask)
        min1 = np.min(img1_fft_abs_mask, where=mask!=0, initial=max1)
        max2 = np.max(img2_fft_abs_mask)
        min2 = np.min(img2_fft_abs_mask, where=mask!=0, initial=max2)
        
        img2_fft_abs_mask = ((img2_fft_abs_mask - min2) / (max2 - min2)) * (max1 - min1) + min1
        lmda = np.random.beta(0.1, 0.1, size=(1, 1, 1))
        img1_fft_abs_mask = lmda *  img1_fft_abs_mask + (1 - lmda) * img2_fft_abs_mask
        img1_fft_abs_new = img1_fft_abs_mask + img1_fft_abs_mask1_noise
        img1_fft = img1_fft_abs_new * np.e ** (1j * img1_fft_angle)
        img1_aug = RGB_ifft(img1_fft)
        
        data = patch_path_list[i][0]
        save_dir = os.path.join(save_root, '/'.join(data.split('/')[-3:-1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_name = data.split('/')[-1].split('.')[0] + aug_tail
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img1_aug)


# augment for laserjket patch---------------------------------------------------------------------------
read_root = 'YOUR_READ_ROOT'
save_root = 'YOUR_SAVE_ROOT'

# how many time of sample you want to augment
aug_num = 3
for aug_index in range(aug_num):
    print ('--------------------------round_' + str(aug_index) + '-----------------------')
    aug_tail = '_fag_' + str(aug_index) + '.tif'
    
    path_list = fag_dict['laser-scanner']
    patch_path_list = []
    r_min_list = []
    r_max_list = []
    print ('laser-scanner strat read data')
    for i, data in tqdm(enumerate(path_list)):
        path = os.path.join(read_root, data[0])
        patch_path_list.append([path, data[1]])
        r_min_list.append('') # cal by BOIL
        r_max_list.append('') # cal by BOIL
    print ('laser-scanner read success')
    Laser_fag(patch_path_list, save_root, r_min_list, r_max_list, mode=1, aug_tail=aug_tail)
    print ('laser-scanner fag success')
    
    print('---------------------------------------------------------------------')
    
    path_list = fag_dict['laser-phone']
    patch_path_list = []
    print ('laser-phone strat read data')
    for i, data in tqdm(enumerate(path_list)):
        path = os.path.join(read_root, data[0])
        patch_path_list.append([path, data[1]])
    print ('laser-phone read success')
    Laser_fag(patch_path_list, save_root, r_min_list, r_max_list, mode=2, aug_tail=aug_tail)
    print ('laser-phone fag success')
# augment for laserjket patch---------------------------------------------------------------------------




def add_noise_inkjet(spectral, fb):
    height, width, depth = spectral.shape
    center = (int(spectral.shape[1]/2), int(spectral.shape[0]/2))
    mask = np.zeros((height, width), np.uint8)

    cv2.circle(mask, center, int(fb), (1,1,1), -1)
    mask1 = 1 - mask

    alpha_b = np.random.uniform(0.5, 0.8, height * width)
    alpha_b = alpha_b.reshape((height, width))
    alpha_g = np.random.uniform(0.5, 0.8, height * width)
    alpha_g = alpha_g.reshape((height, width))
    alpha_r = np.random.uniform(0.5, 0.8, height * width)
    alpha_r = alpha_r.reshape((height, width))
    
    beta_b = np.random.normal(0, 1, height * width)
    beta_b = beta_b.reshape((height, width))
    beta_g = np.random.normal(0, 1, height * width)
    beta_g = beta_g.reshape((height, width))
    beta_r = np.random.normal(0, 1, height * width)
    beta_r = beta_r.reshape((height, width))
    
    spectral_mask_noise_b = (alpha_b * spectral[:, :, 0] + beta_b) * mask + spectral[:, :, 0] * mask1
    spectral_mask_noise_g = (alpha_g * spectral[:, :, 1] + beta_g) * mask + spectral[:, :, 1] * mask1
    spectral_mask_noise_r = (alpha_r * spectral[:, :, 2] + beta_r) * mask + spectral[:, :, 2] * mask1
    spectral_mask_noise = np.stack((spectral_mask_noise_b, spectral_mask_noise_g, spectral_mask_noise_r), -1)
    return spectral_mask_noise



def Inkjet_fag(patch_path_list, save_root, aug_tail='fag.tif'):
    print ('inkjet patch len: ', len(patch_path_list))
    for i in tqdm(range(len(patch_path_list))):
        img = cv2.imread(patch_path_list[i][0])
        img_fft_abs_origin, img_fft_angle = RGB_fft(img)
        second = patch_path_list[i][1]

        fb = int(float(patch_path_list[i][2]))
            
        mask = np.zeros((224, 224, 3), np.uint8)
        center = (112, 112)
        cv2.circle(mask, center, int(fb), (1,1,1), -1)
        mask1 = 1 - mask
#         img_fft_abs = img_fft_abs * mask1 * d + img_fft_abs_origin * mask
        
        spectral_aug = add_noise_inkjet(img_fft_abs_origin, fb)

        img_fft = spectral_aug * np.e ** (1j * img_fft_angle)
        img_aug = RGB_ifft(img_fft)
#         show(img_aug)
        data = path_list[i][0]
        save_dir = os.path.join(save_root, '/'.join(data.split('/')[-3:-1]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_name = data.split('/')[-1].split('.')[0] + aug_tail
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img_aug)


# augment for inkjet patch---------------------------------------------------------------------------
read_root = 'YOUR_READ_ROOT'
save_root = 'YOUR_SAVE_ROOT'

aug_num = 3
for aug_index in range(aug_num):
    print ('--------------------------round_' + str(aug_index) + '-----------------------')
    aug_tail = '_fag2_' + str(aug_index) + '.tif'
    
    path_list = fag_dict['injek-scanner']
    patch_path_list = []
    print ('injek-scanner strat read data')
    for i, data in tqdm(enumerate(path_list)):
        path = os.path.join(read_root, data[0])
        patch_path_list.append([path, data[1], data[2]])
    print ('injek-scanner read success')
    Inkjet_fag(patch_path_list, save_root, aug_tail=aug_tail)
    print ('injek-scanner fag success')
    
    print('---------------------------------------------------------------------')
    path_list = fag_dict['injek-phone']
    patch_path_list = []
    for i, data in tqdm(enumerate(path_list)):
        path = os.path.join(read_root, data[0])
        patch_path_list.append([path, data[1], data[2]])
    print ('injek-phone read success')
    Inkjet_fag(patch_path_list, save_root, aug_tail=aug_tail)
    print ('injek-phone fag success')
# augment for inkjet patch---------------------------------------------------------------------------

