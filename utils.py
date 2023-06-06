import os
import sys
import time
import random
import pathlib
import math

import numpy as np

import torch
import torch.nn.functional as F
from torchvision.io import read_image
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.ops import Conv2dNormActivation
from torchsummary import summary

import torch_geometric
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data as geo_Data
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv



'''****************************************************** paths'''
WS_PATH = os.getcwd()
EXPERIMENT_OUTPUT_DIR = os.path.join(WS_PATH , 'experiment_output')

Night_val_txt = os.path.join(WS_PATH, 'dataset_txt/Night_val_stereo_centre.txt')
Autumn_val_txt = os.path.join(WS_PATH, 'dataset_txt/Autumn_val_stereo_centre.txt')
Suncloud_val_txt = os.path.join(WS_PATH, 'dataset_txt/Suncloud_val_stereo_centre.txt')
Night_val_vo = os.path.join(WS_PATH, 'ee5346_dataset/Night_val/vo/vo.csv')
Autumn_val_vo = os.path.join(WS_PATH, 'ee5346_dataset/Autumn_val/vo/vo.csv')
Suncloud_val_vo = os.path.join(WS_PATH, 'ee5346_dataset/Suncloud_val/vo/vo.csv')
Night_val_imgfolder = os.path.join(WS_PATH , 'ee5346_dataset/Night_val/stereo/centre/')
Autumn_val_imgfolder = os.path.join(WS_PATH , 'ee5346_dataset/Autumn_val/stereo/centre/')
Suncloud_val_imgfolder = os.path.join(WS_PATH , 'ee5346_dataset/Suncloud_val/stereo/centre/')

#### TODO: after finish copying dataset, create these path
DIY1_night_txt = os.path.join(WS_PATH, 'dataset_txt/DIY1_night.txt')
DIY1_day_txt = os.path.join(WS_PATH, 'dataset_txt/DIY1_day.txt')
DIY1_night_vo = os.path.join(WS_PATH, 'LCD_DIY_1/vo_data/DIY1_night_vo.csv')
DIY1_day_vo = os.path.join(WS_PATH, 'LCD_DIY_1/vo_data/DIY1_day_vo.csv')
DIY1_night_imgfolder = os.path.join(WS_PATH , 'LCD_DIY_1/img_data/query/')
DIY1_day_imgfolder = os.path.join(WS_PATH , 'LCD_DIY_1/img_data/ref/')


### loop closure data
Loop_dataset_path = WS_PATH + '/loop_closure_dataset/'
lcd_autumn_night_easy_path = Loop_dataset_path + 'robotcar_qAutumn_dbNight_easy_final.txt'
lcd_autumn_night_diff_path = Loop_dataset_path + 'robotcar_qAutumn_dbNight_diff_final.txt'
lcd_autumn_suncloud_easy_path = Loop_dataset_path + 'robotcar_qAutumn_dbSunCloud_easy_final.txt'
lcd_autumn_suncloud_diff_path = Loop_dataset_path + 'robotcar_qAutumn_dbSunCloud_diff_final.txt'

#### TODO: 
# DIY1_night_day_loopclosure = Loop_dataset_path + 'DIY1_groundtruth_20.txt'  ## the gt threshold is 20m
DIY1_night_day_loopclosure = Loop_dataset_path + 'DIY1_groundtruth_25.txt'  ## the gt threshold is 25m



lcd_autumn_night_val_path = Loop_dataset_path + 'robotcar_qAutumn_dbNight_val_final.txt'
lcd_autumn_suncloud_val_path = Loop_dataset_path + 'robotcar_qAutumn_dbSunCloud_val_final.txt'



'''****************************************************** params for model'''
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GNN_VARIANT = 0  # (0:GCN, 1:GATv2, 2:GIN)

BACKBONE_OUTPUT_DIM = 512
# GNN_IN_DIM is calculated by the staticmethod in LCDModel
GNN_HID_DIM = 256
GNN_OUT_DIM = 256
CONTEXT_VECTOR_DIM = 256
DIST_ENC_DIM = 32
TIME_ENC_DIM = 32

# GNN_LAYERS = 3
# DROPOUT_RAT = 0.3

# LR = 0.001
# EPOCHES = 30
MATCH_THREASH = 0.5


'''****************************************************** params for data'''

# TRAIN_RATIO = 0.8
# VAL_RATIO = 0.1
# TEST_RATIO = 0.1


DIST_THREASH = 100 # meter. general dist threash.
IS_ONLINE = False # online case: no right seq for query. offline: normal.
# (up) meter. 0 or n, adjust for online / offline loop closure. However, here we can ignore online case here.
# SELECT_INTVAL = 7  # meter.
COUNT_IMG_INTVAL = 10 # select interval while counting vo dist
TIME_NORM = 10 * 1000000
DIST_NORM = DIST_THREASH





'''****************************************************** util functions'''

def find_seq(query_img, query_vo, query_img_idx, query_vo_idx, dist_threash, select_intv, count_img_intv, is_online):
    '''
    query_img: numpy array, load from txt. shape (n,1). dtype=string.
    query_vo: numpy array, load from txt. shape (n,8). dtype=float/double.
    '''
    
    left_img_idx, right_img_idx = None, None
    left_img_idx_seq, right_img_idx_seq = [], []
    left_relat_dist_seq, right_relat_dist_seq = [], []
    
    left_dist_count, right_dist_count = 0, 0
    left_dist_count_temp, right_dist_count_temp = 0, 0


    #### count left
    left_vo_idx = query_vo_idx  
    left_img_idx = query_img_idx  
    terminate_left = False   
    while not terminate_left:
        
        left_last_vo_idx, left_last_img_idx = left_vo_idx, left_img_idx   
             
        left_img_idx = left_last_img_idx - count_img_intv  
        left_vo_idx = left_last_vo_idx - count_img_intv 
              
        if(left_img_idx>=0 and left_vo_idx>=0):
            vo_data = query_vo[left_vo_idx+1:left_last_vo_idx+1] 
            # print(vo_data.shape)  # (n,8)
            vo_x, vo_y, vo_z = vo_data[:,2], vo_data[:,3], vo_data[:,4]
            # print(vo_x.shape) # (n,)
            dist = np.sum(np.sqrt(vo_x**2 + vo_y**2 + vo_z**2))
            # print(img_strarr[left_img_idx])
            # print(dist)            
            left_dist_count += dist
            left_dist_count_temp += dist
            
            if(left_dist_count_temp >= select_intv):
                left_img_idx_seq.append(left_img_idx)
                left_relat_dist_seq.append(left_dist_count_temp)
                left_dist_count_temp = 0

            if(left_dist_count > dist_threash):
                terminate_left = True
        else:
            terminate_left = True

    left_relat_dist_seq = (- np.cumsum(left_relat_dist_seq)).tolist()
    left_img_idx_seq.reverse()   # reverse so that it rank from small to large
    left_relat_dist_seq.reverse()

    #### count right
    if(not is_online):
        right_vo_idx = query_vo_idx  
        right_img_idx = query_img_idx  
        terminate_right = False   
        while not terminate_right:
            
            right_last_vo_idx, right_last_img_idx = right_vo_idx, right_img_idx   
                
            right_img_idx = right_last_img_idx + count_img_intv  
            right_vo_idx = right_last_vo_idx + count_img_intv 
                
            if(right_img_idx < query_img.shape[0] and right_vo_idx < query_vo.shape[0]):
                vo_data = query_vo[right_last_vo_idx+1:right_vo_idx+1] 
                vo_x, vo_y, vo_z = vo_data[:,2], vo_data[:,3], vo_data[:,4]
                dist = np.sum(np.sqrt(vo_x**2 + vo_y**2 + vo_z**2))
            
                right_dist_count += dist
                right_dist_count_temp += dist
                
                if(right_dist_count_temp >= select_intv):
                    right_img_idx_seq.append(right_img_idx)
                    right_relat_dist_seq.append(right_dist_count_temp)
                    right_dist_count_temp = 0

                if(right_dist_count > dist_threash):
                    terminate_right = True
            else:
                terminate_right = True            
    
    right_relat_dist_seq = np.cumsum(right_relat_dist_seq).tolist()
    
    return left_img_idx_seq, right_img_idx_seq, \
        left_relat_dist_seq, right_relat_dist_seq, \
            left_dist_count, right_dist_count


def imgstr2timestamp(img_seq):
    timestamp_seq = []
    for img_str in img_seq:
        time_str, _ = img_str.split('.')
        timestamp_seq.append(int(time_str))
    return timestamp_seq
    

def imgstr2idx(imgstrs, img_strarr, vo_strarr):
    img_idxs, vo_idxs = [], []  
    
    for i in range(len(imgstrs)):
        imgstr = imgstrs[i]
        img_idx = np.where(img_strarr == imgstr)[0][0]
        query_timestamp, _ = imgstr.split('.')
        query_timestamp = int(query_timestamp)
        
        ### handle special case: target not find in the first column
        # print(query_timestamp)
        # print(np.where(vo_strarr[:,0].flatten() == np.double(query_timestamp))[0]) 
        
        not_find1 = len(np.where(vo_strarr[:,0].flatten() == np.double(query_timestamp))[0]) == 0
        not_find2 = len( np.where(vo_strarr[:,1].flatten() == np.double(query_timestamp))[0]) == 0
        while(not_find1 and not_find2):
            img_idx -= 1
            imgstr = img_strarr[img_idx]
            query_timestamp, _ = imgstr.split('.')
            query_timestamp = int(query_timestamp)    
            not_find1 = len(np.where(vo_strarr[:,0].flatten() == np.double(query_timestamp))[0]) == 0
            not_find2 = len( np.where(vo_strarr[:,1].flatten() == np.double(query_timestamp))[0]) == 0

        if( len(np.where(vo_strarr[:,0].flatten() == np.double(query_timestamp))[0]) == 0):
            ## can't find this timestamp in the first colomn
            vo_idx = np.where(vo_strarr[:,1].flatten() == np.double(query_timestamp))[0][0]
        else:
            vo_idx = np.where(vo_strarr[:,0].flatten() == np.double(query_timestamp))[0][0]       
      
        img_idxs.append(img_idx)        
        vo_idxs.append(vo_idx)
        
    return img_idxs, vo_idxs
