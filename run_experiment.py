
from utils import *
from model import *
import nni


'''TODO: should be modified'''
CURRENT_EXPERIMENT_DIR_NAME = 'experiment8'     ### TODO: should be modified for each experiment!!!!
CURRENT_EXPERIMENT_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, CURRENT_EXPERIMENT_DIR_NAME)
assert os.path.isdir(CURRENT_EXPERIMENT_DIR)    # make sure that the dir exist


ENABLE_TINY_TEST = True ## experiment 6: use more data as well as reversed data. then reduce data after random shuffled.
# TINY_TEST_RATIO = 0.4 # see nni params

IS_REDUCE_DIY1 = False
REDUCE_DIY1_RAT = 0.4

DIY1_DATA_NUM = 3000

# BATCH_LOADS = 10
BATCH_LOADS = 8
BATCH_UPDATES = 2
EPOCHES = 5

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# random.seed(0)
np.random.seed(0)

### experiment 3: 
BACKBONE_OUTPUT_DIM = 512
GNN_HID_DIM = 512
GNN_OUT_DIM = 512



### fixed nni params
# GNN_LAYERS = 2
SELECT_INTVAL = 7  # meter.
CONTEXT_VECTOR_DIM = 256


'''****************************************************** nni get params'''
nni_params = {
    'tiny_test_ratio': 0.4,
    # 'gnn_variant': 0,  
    'dist_thresh': 60,  
    # 'select_intval': 10,
    # 'context_vec_dim': 256,   
    'lr': 0.0005,
    'layers': 3,
    'dropout_rat': 0.3,
}
optimized_params = nni.get_next_parameter()
nni_params.update(optimized_params)

DIST_THREASH = nni_params['dist_thresh']
LR = nni_params['lr']
GNN_LAYERS = nni_params['layers']
# GNN_VARIANT = nni_params['gnn_variant']
DROPOUT_RAT = nni_params['dropout_rat']
TINY_TEST_RATIO = nni_params['tiny_test_ratio']
# CONTEXT_VECTOR_DIM = nni_params['context_vec_dim']
# SELECT_INTVAL = nni_params['select_intval']



'''****************************************************** make dir for this trial'''
time_str = time.strftime( "Date_%Y_%m_%d_Time_%H_%M_%S", time.gmtime() )
nni_exp_id = nni.get_experiment_id()
nni_trial_id = nni.get_trial_id()
info_str = time_str + '_expID_' + nni_exp_id + '_trialID_' + nni_trial_id

trial_path = os.path.join(CURRENT_EXPERIMENT_DIR, info_str)
while(os.path.isdir(trial_path)):
    trial_path += '_'   # to prevent possible conflict
os.mkdir(trial_path)
print(f'trial_path: {trial_path}')




'''****************************************************** load autumn night data'''
np.random.seed(0)

#### imgs txt file to array, shape (n,)
autumn_img_strarr = np.loadtxt(Autumn_val_txt, dtype=str)
night_img_strarr = np.loadtxt(Night_val_txt, dtype=str)

#### vo txt file, shape (n,8)
Autumn_vo_strarr = np.loadtxt(Autumn_val_vo, delimiter=',', skiprows=1, dtype=np.double) 
night_vo_strarr = np.loadtxt(Night_val_vo, delimiter=',', skiprows=1, dtype=np.double) 


#### load the example loop closure files
lcd_autumn_night_easy_strarr = np.loadtxt(lcd_autumn_night_easy_path, dtype=str, delimiter=' ')   
lcd_autumn_night_diff_strarr = np.loadtxt(lcd_autumn_night_diff_path, dtype=str, delimiter=' ')  


lcd_autumn_night_strarr = np.concatenate((lcd_autumn_night_easy_strarr,
                                          lcd_autumn_night_diff_strarr), axis=0) 

#### random shuffle and pick a portion
np.random.shuffle(lcd_autumn_night_strarr)


#### remove the ',' in numpy_str_
for i in range(len(lcd_autumn_night_strarr)):
    lcd_autumn_night_strarr[i][0] = np.char.strip(lcd_autumn_night_strarr[i][0], ',')
    lcd_autumn_night_strarr[i][1] = np.char.strip(lcd_autumn_night_strarr[i][1], ',')
for i in range(len(lcd_autumn_night_strarr)):
    _, lcd_autumn_night_strarr[i][0] = str(lcd_autumn_night_strarr[i][0]).split('/')
    _, lcd_autumn_night_strarr[i][1] = str(lcd_autumn_night_strarr[i][1]).split('/') 
    
#### find sequence for each line and build LCDdata list
autumn_img_idxs, autumn_vo_idxs = \
    imgstr2idx(lcd_autumn_night_strarr[:,0], autumn_img_strarr, Autumn_vo_strarr)
night_img_idxs, night_vo_idxs = \
    imgstr2idx(lcd_autumn_night_strarr[:,1], night_img_strarr, night_vo_strarr)
autumn_night_gt = [float(g) for g in lcd_autumn_night_strarr[:,2]]

assert len(autumn_img_idxs) == len(night_img_idxs)

#### build the LCDdata list
LCDdata_list = []

for i in range(len(autumn_img_idxs)):
    ## find query
    q1_img_idx, q1_vo_idx = autumn_img_idxs[i], autumn_vo_idxs[i]
    q2_img_idx, q2_vo_idx = night_img_idxs[i], night_vo_idxs[i] 
    q1 = autumn_img_strarr[q1_img_idx]
    q2 = night_img_strarr[q2_img_idx]  
    
    ## find sequence
    q1_left_idx_seq, q1_right_idx_seq, q1_left_relat_dist, q1_right_relat_dist, _, _ = \
        find_seq(autumn_img_strarr, Autumn_vo_strarr, q1_img_idx, q1_vo_idx, DIST_THREASH, SELECT_INTVAL, COUNT_IMG_INTVAL, False)
    q2_left_idx_seq, q2_right_idx_seq, q2_left_relat_dist, q2_right_relat_dist, _, _ = \
        find_seq(night_img_strarr, night_vo_strarr, q2_img_idx, q2_vo_idx, DIST_THREASH, SELECT_INTVAL, COUNT_IMG_INTVAL, False)
 
    ## info for each query
    q1_pos = len(q1_left_idx_seq)
    q1_imgfolder = Autumn_val_imgfolder
    q1_idx_seq = q1_left_idx_seq + [q1_img_idx] + q1_right_idx_seq
    q1_seq = autumn_img_strarr.flatten()[q1_idx_seq]
    q1_relat_dist = q1_left_relat_dist + [0.0] + q1_right_relat_dist
    q1_relat_dist = [(d / DIST_NORM) for d in q1_relat_dist]     
    
    q2_pos = len(q2_left_idx_seq)
    q2_imgfolder = Night_val_imgfolder  
    q2_idx_seq = q2_left_idx_seq + [q2_img_idx] + q2_right_idx_seq
    q2_seq = night_img_strarr.flatten()[q2_idx_seq]
    q2_relat_dist = q2_left_relat_dist + [0.0] + q2_right_relat_dist
    q2_relat_dist = [(d / DIST_NORM) for d in q2_relat_dist]
    
    q1_timestamp, q2_timestamp = imgstr2timestamp(q1_seq), imgstr2timestamp(q2_seq)
    q1_relat_timestamp = [( (t - q1_timestamp[q1_pos]) / TIME_NORM) for t in q1_timestamp]
    q2_relat_timestamp = [( (t - q2_timestamp[q2_pos]) / TIME_NORM) for t in q2_timestamp]  
        
    lcd_data = LCDdata(q1_imgfolder, q2_imgfolder,
                       q1_seq, q2_seq,
                       q1_pos, q2_pos,
                       q1_relat_dist, q2_relat_dist,
                       q1_relat_timestamp, q2_relat_timestamp,
                       autumn_night_gt[i])

    LCDdata_list.append(lcd_data)




'''****************************************************** load DIY data'''
np.random.seed(0)

DIY1_night_img_strarr = np.loadtxt(DIY1_night_txt, dtype=str)
DIY1_day_img_strarr = np.loadtxt(DIY1_day_txt, dtype=str)

DIY1_night_vo_strarr = np.loadtxt(DIY1_night_vo, delimiter=',', skiprows=1, dtype=np.double) 
DIY1_day_vo_strarr = np.loadtxt(DIY1_day_vo, delimiter=',', skiprows=1, dtype=np.double) 

DIY1_night_day_loopclosure_strarr = \
    np.loadtxt(DIY1_night_day_loopclosure, dtype=str, delimiter=' ')
    
    
for i in range(len(DIY1_night_day_loopclosure_strarr)):
    DIY1_night_day_loopclosure_strarr[i][0] = np.char.strip(DIY1_night_day_loopclosure_strarr[i][0], ',')
    DIY1_night_day_loopclosure_strarr[i][0] = np.char.strip(DIY1_night_day_loopclosure_strarr[i][0], '.')
    DIY1_night_day_loopclosure_strarr[i][0] = np.char.strip(DIY1_night_day_loopclosure_strarr[i][0], '/')

    DIY1_night_day_loopclosure_strarr[i][1] = np.char.strip(DIY1_night_day_loopclosure_strarr[i][1], ',')
    DIY1_night_day_loopclosure_strarr[i][1] = np.char.strip(DIY1_night_day_loopclosure_strarr[i][1], '.')
    DIY1_night_day_loopclosure_strarr[i][1] = np.char.strip(DIY1_night_day_loopclosure_strarr[i][1], '/')
    
for i in range(len(DIY1_night_day_loopclosure_strarr)):
    _, DIY1_night_day_loopclosure_strarr[i][0] = str(DIY1_night_day_loopclosure_strarr[i][0]).split('/')
    _, DIY1_night_day_loopclosure_strarr[i][1] = str(DIY1_night_day_loopclosure_strarr[i][1]).split('/') 
# print(DIY1_night_day_loopclosure_strarr[:5])


DIY1_night_img_idx, DIY1_night_vo_idx = \
    imgstr2idx(DIY1_night_day_loopclosure_strarr[:,0], 
               DIY1_night_img_strarr,
               DIY1_night_vo_strarr
               )
DIY1_day_img_idx, DIY1_day_vo_idx = \
    imgstr2idx(DIY1_night_day_loopclosure_strarr[:,1],
               DIY1_day_img_strarr,
               DIY1_day_vo_strarr)
DIY1_night_day_gt = [float(g) for g in DIY1_night_day_loopclosure_strarr[:,2]]

assert len(DIY1_night_img_idx) == len(DIY1_day_img_idx)


LCDdata_list_2 = []
for i in range(len(DIY1_night_img_idx)):
    '''experiment5: reverse q1/q2, night/day'''
    q2_img_idx, q2_vo_idx = DIY1_night_img_idx[i], DIY1_night_vo_idx[i]
    q1_img_idx, q1_vo_idx = DIY1_day_img_idx[i], DIY1_day_vo_idx[i]
    q2 = DIY1_night_img_strarr[q2_img_idx]
    q1 = DIY1_day_img_strarr[q1_img_idx]
    
    ## find sequence
    '''experiment5: reverse q1/q2, night/day'''    
    q2_left_idx_seq, q2_right_idx_seq, q2_left_relat_dist, q2_right_relat_dist, _, _ = \
        find_seq(DIY1_night_img_strarr, DIY1_night_vo_strarr, q2_img_idx, q2_vo_idx, DIST_THREASH, SELECT_INTVAL, COUNT_IMG_INTVAL, False)
    q1_left_idx_seq, q1_right_idx_seq, q1_left_relat_dist, q1_right_relat_dist, _, _ = \
        find_seq(DIY1_day_img_strarr, DIY1_day_vo_strarr, q1_img_idx, q1_vo_idx, DIST_THREASH, SELECT_INTVAL, COUNT_IMG_INTVAL, False)
 
    ## info for each query
    q1_pos = len(q1_left_idx_seq)
    
    '''experiment5: reverse the night and day'''
    q1_imgfolder = DIY1_day_imgfolder
    q1_idx_seq = q1_left_idx_seq + [q1_img_idx] + q1_right_idx_seq
    q1_seq = DIY1_day_img_strarr.flatten()[q1_idx_seq]
    q1_relat_dist = q1_left_relat_dist + [0.0] + q1_right_relat_dist
    q1_relat_dist = [(d / DIST_NORM) for d in q1_relat_dist]     
    
    q2_pos = len(q2_left_idx_seq)
    '''experiment5: reverse the night and day'''
    q2_imgfolder = DIY1_night_imgfolder  
    q2_idx_seq = q2_left_idx_seq + [q2_img_idx] + q2_right_idx_seq
    q2_seq = DIY1_night_img_strarr.flatten()[q2_idx_seq]
    q2_relat_dist = q2_left_relat_dist + [0.0] + q2_right_relat_dist
    q2_relat_dist = [(d / DIST_NORM) for d in q2_relat_dist] 
    
    q1_timestamp, q2_timestamp = imgstr2timestamp(q1_seq), imgstr2timestamp(q2_seq)
    q1_relat_timestamp = [( (t - q1_timestamp[q1_pos]) / TIME_NORM) for t in q1_timestamp]
    q2_relat_timestamp = [( (t - q2_timestamp[q2_pos]) / TIME_NORM) for t in q2_timestamp] 
    
    '''experiment5: now, q1 is day and q2 is night'''
    lcd_data = LCDdata(q1_imgfolder, q2_imgfolder,
                       q1_seq, q2_seq,
                       q1_pos, q2_pos,
                       q1_relat_dist, q2_relat_dist,
                       q1_relat_timestamp, q2_relat_timestamp,
                       DIY1_night_day_gt[i])
    lcd_data_reverse = LCDdata(q2_imgfolder, q1_imgfolder,
                            q2_seq, q1_seq,
                            q2_pos, q1_pos,
                            q2_relat_dist, q1_relat_dist,
                            q2_relat_timestamp, q1_relat_timestamp,
                            DIY1_night_day_gt[i])    
    LCDdata_list_2.append(lcd_data)    
    LCDdata_list_2.append(lcd_data_reverse)  


np.random.shuffle(LCDdata_list_2)
LCDdata_list_2 = LCDdata_list_2[:DIY1_DATA_NUM]  
   


'''****************************************************** suncloud datasets'''

np.random.seed(0)

#### imgs txt file to array, shape (n,)
autumn_img_strarr = np.loadtxt(Autumn_val_txt, dtype=str)
suncloud_img_strarr = np.loadtxt(Suncloud_val_txt, dtype=str)

#### vo txt file, shape (n,8)
Autumn_vo_strarr = np.loadtxt(Autumn_val_vo, delimiter=',', skiprows=1, dtype=np.double) 
suncloud_vo_strarr = np.loadtxt(Suncloud_val_vo, delimiter=',', skiprows=1, dtype=np.double) 

#### load the example loop closure files
lcd_autumn_suncloud_easy_strarr = np.loadtxt(lcd_autumn_suncloud_easy_path, dtype=str, delimiter=' ')   
lcd_autumn_suncloud_diff_strarr = np.loadtxt(lcd_autumn_suncloud_diff_path, dtype=str, delimiter=' ')  


lcd_autumn_suncloud_strarr = np.concatenate((lcd_autumn_suncloud_easy_strarr,
                                          lcd_autumn_suncloud_diff_strarr), axis=0) 



#### random shuffle and pick a portion

np.random.shuffle(lcd_autumn_suncloud_strarr)


#### remove the ',' in numpy_str_
for i in range(len(lcd_autumn_suncloud_strarr)):
    lcd_autumn_suncloud_strarr[i][0] = np.char.strip(lcd_autumn_suncloud_strarr[i][0], ',')
    lcd_autumn_suncloud_strarr[i][1] = np.char.strip(lcd_autumn_suncloud_strarr[i][1], ',')
for i in range(len(lcd_autumn_suncloud_strarr)):
    _, lcd_autumn_suncloud_strarr[i][0] = str(lcd_autumn_suncloud_strarr[i][0]).split('/')
    _, lcd_autumn_suncloud_strarr[i][1] = str(lcd_autumn_suncloud_strarr[i][1]).split('/') 
    
#### find sequence for each line and build LCDdata list
autumn_img_idxs, autumn_vo_idxs = \
    imgstr2idx(lcd_autumn_suncloud_strarr[:,0], autumn_img_strarr, Autumn_vo_strarr)
suncloud_img_idxs, suncloud_vo_idxs = \
    imgstr2idx(lcd_autumn_suncloud_strarr[:,1], suncloud_img_strarr, suncloud_vo_strarr)
autumn_suncloud_gt = [float(g) for g in lcd_autumn_suncloud_strarr[:,2]]

assert len(autumn_img_idxs) == len(suncloud_img_idxs)

#### build the LCDdata list
LCDdata_list_3 = []

for i in range(len(autumn_img_idxs)):
    ## find query
    q1_img_idx, q1_vo_idx = autumn_img_idxs[i], autumn_vo_idxs[i]
    q2_img_idx, q2_vo_idx = suncloud_img_idxs[i], suncloud_vo_idxs[i] 
    q1 = autumn_img_strarr[q1_img_idx]
    q2 = suncloud_img_strarr[q2_img_idx]  
    
    ## find sequence
    q1_left_idx_seq, q1_right_idx_seq, q1_left_relat_dist, q1_right_relat_dist, _, _ = \
        find_seq(autumn_img_strarr, Autumn_vo_strarr, q1_img_idx, q1_vo_idx, DIST_THREASH, SELECT_INTVAL, COUNT_IMG_INTVAL, False)
    q2_left_idx_seq, q2_right_idx_seq, q2_left_relat_dist, q2_right_relat_dist, _, _ = \
        find_seq(suncloud_img_strarr, suncloud_vo_strarr, q2_img_idx, q2_vo_idx, DIST_THREASH, SELECT_INTVAL, COUNT_IMG_INTVAL, False)
 
    ## info for each query
    q1_pos = len(q1_left_idx_seq)
    q1_imgfolder = Autumn_val_imgfolder
    q1_idx_seq = q1_left_idx_seq + [q1_img_idx] + q1_right_idx_seq
    q1_seq = autumn_img_strarr.flatten()[q1_idx_seq]
    q1_relat_dist = q1_left_relat_dist + [0.0] + q1_right_relat_dist
    q1_relat_dist = [(d / DIST_NORM) for d in q1_relat_dist]     
    
    q2_pos = len(q2_left_idx_seq)
    q2_imgfolder = Suncloud_val_imgfolder  
    q2_idx_seq = q2_left_idx_seq + [q2_img_idx] + q2_right_idx_seq
    q2_seq = suncloud_img_strarr.flatten()[q2_idx_seq]
    q2_relat_dist = q2_left_relat_dist + [0.0] + q2_right_relat_dist
    q2_relat_dist = [(d / DIST_NORM) for d in q2_relat_dist]
    
    q1_timestamp, q2_timestamp = imgstr2timestamp(q1_seq), imgstr2timestamp(q2_seq)
    q1_relat_timestamp = [( (t - q1_timestamp[q1_pos]) / TIME_NORM) for t in q1_timestamp]
    q2_relat_timestamp = [( (t - q2_timestamp[q2_pos]) / TIME_NORM) for t in q2_timestamp]  
        
    lcd_data = LCDdata(q1_imgfolder, q2_imgfolder,
                       q1_seq, q2_seq,
                       q1_pos, q2_pos,
                       q1_relat_dist, q2_relat_dist,
                       q1_relat_timestamp, q2_relat_timestamp,
                       autumn_suncloud_gt[i])
    lcd_data_reverse = LCDdata(q2_imgfolder, q1_imgfolder,
                       q2_seq, q1_seq,
                       q2_pos, q1_pos,
                       q2_relat_dist, q1_relat_dist,
                       q2_relat_timestamp, q1_relat_timestamp,
                       autumn_suncloud_gt[i])    
    
    LCDdata_list_3.append(lcd_data)
    LCDdata_list_3.append(lcd_data_reverse)


# print(f'LCDdata_list_3[:10]: {LCDdata_list_3[:10]}') 
    

# train_num_3 = int(len(LCDdata_list_3)*TRAIN_RATIO)
# val_num_3 = train_num_3 + int(len(LCDdata_list_3)*VAL_RATIO)

# train_data_list_3 = LCDdata_list_3[:train_num_3]
# val_data_list_3 = LCDdata_list_3[train_num_3:val_num_3]
# test_data_list_3 = LCDdata_list_3[val_num_3:]



'''****************************************************** combine datasets'''
random.seed(0)


'''experiment5: use DIY1 as train, autumn-night as val/test'''
train_data_list = LCDdata_list_2
train_data_list.extend(LCDdata_list_3)
val_data_list = LCDdata_list.copy()
test_data_list = LCDdata_list.copy()

# print(train_data_list)

'''experimen5: first try train & val & test on DIY1'''
# train_data_list = train_data_list_2
# val_data_list = val_data_list_2
# test_data_list = test_data_list_2


# train_data_list.extend(train_data_list_2)
# train_data_list.extend(train_data_list_3)
### python random to shuffle train list
random.shuffle(train_data_list)


# val_data_list.extend(val_data_list_2)
# # val_data_list.extend(val_data_list_3)
random.shuffle(val_data_list)

# test_data_list.extend(test_data_list_2)
# # test_data_list.extend(test_data_list_3)
random.shuffle(test_data_list)

if(ENABLE_TINY_TEST):
    train_len = len(train_data_list)
    new_train_len = int(train_len * TINY_TEST_RATIO)
    train_data_list = train_data_list[:new_train_len]


print(f'train data num: {len(train_data_list)}')
print(f'val data num: {len(val_data_list)}')
print(f'test data num: {len(test_data_list)}')


    
'''****************************************************** load model'''

#### load backbone
effi_weights = EfficientNet_V2_M_Weights.DEFAULT
effi_model = efficientnet_v2_m(weights=effi_weights).to(DEVICE)
effi_model.eval()
for param in effi_model.parameters():
    param.requires_grad = False

effi_model.avgpool = \
    torch.nn.Sequential(
        Conv2dNormActivation(
            in_channels=1280,
            out_channels=512,
            kernel_size=3,
            stride=3,
            padding=0,
            activation_layer=torch.nn.LeakyReLU,
        ),
        Conv2dNormActivation(
            in_channels=512,
            out_channels=512,
            kernel_size=5,
            stride=1,
            padding=0,
            activation_layer=torch.nn.LeakyReLU,
        )
    )


#### replace the classifier
effi_model.classifier = \
    torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=BACKBONE_OUTPUT_DIM, bias=True).to(DEVICE),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(DROPOUT_RAT)
    )
    

for m in effi_model.avgpool.modules():
    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
        # print(f'test init norm')        
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  
    else:
        pass    
for m in effi_model.classifier.modules():
    if isinstance(m, torch.nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        torch.nn.init.uniform_(m.weight, -init_range, init_range)
        torch.nn.init.zeros_(m.bias)
        
    
backbone_train_params = sum(p.numel() for p in effi_model.parameters() if p.requires_grad)         
print(f'backbone_train_params:{backbone_train_params}')

#### init gnn
gnn_in_dim = LCDModel.count_gnn_in_dim(BACKBONE_OUTPUT_DIM, DIST_ENC_DIM, TIME_ENC_DIM)

if(GNN_VARIANT == 0):
    ## GCN
    gnn_model = GCN(GNN_LAYERS, gnn_in_dim, GNN_HID_DIM, GNN_OUT_DIM, DROPOUT_RAT).to(DEVICE)
elif(GNN_VARIANT == 1):
    ## GATv2
    gnn_model = GATv2(GNN_LAYERS, gnn_in_dim, GNN_HID_DIM, GNN_OUT_DIM, DROPOUT_RAT).to(DEVICE)
else:
    ## GIN
    gnn_model = GIN(GNN_LAYERS, gnn_in_dim, GNN_HID_DIM, GNN_OUT_DIM, DROPOUT_RAT).to(DEVICE)
    
for m in gnn_model.modules():
    if isinstance(m, torch.nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        torch.nn.init.uniform_(m.weight, -init_range, init_range)
        torch.nn.init.zeros_(m.bias) 

# gnn_train_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
gnn_train_params = torch_geometric.profile.count_parameters(gnn_model)
print(f'gnn_train_params:{gnn_train_params}')

lcd_model = LCDModel(effi_model, effi_weights.transforms(),
                     gnn_model, GNN_OUT_DIM, CONTEXT_VECTOR_DIM, DEVICE,
                     DROPOUT_RAT, DIST_ENC_DIM, TIME_ENC_DIM)



total_train_params = sum(p.numel() for p in lcd_model.parameters() if p.requires_grad)
print(f'total_train_params:{total_train_params}')
total_params = sum(p.numel() for p in lcd_model.parameters())
print(f'total_params:{total_params}')



'''****************************************************** train val eval'''
optimizer = torch.optim.AdamW(lcd_model.parameters(), lr=LR)
loss_fn = torch.nn.BCELoss()



def train(lcd_model:LCDModel, 
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.BCELoss, 
          lcd_data_list_train:list, 
          ):
    
    lcd_model.train()
    total_loss = 0.0
    train_data_len = len(lcd_data_list_train)
    
    ### build train data idx, and use batch loads and updates
    batch_load_idx = []
    current_idx = 0
    while(current_idx < train_data_len):
        batch_load_idx.append(current_idx)
        current_idx += BATCH_LOADS
        
    for idx in batch_load_idx:
        print(f'processing batch loading start with {idx} of total {train_data_len}')
        next_idx = min(idx+BATCH_LOADS, train_data_len)
        batch_data = train_data_list[idx:next_idx]  
        for data in batch_data:
            data.to_torch_seq()
            data.to_device(DEVICE)  # process once, accelerate?

        
        ## batch updates
        for update in range(BATCH_UPDATES):
            for data in batch_data:
                out = lcd_model(data)
                loss = loss_fn(out, data.gt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  
                total_loss += loss.item()                
        
        for data in batch_data:
            data.release_images()
        

    avg_loss = total_loss / (len(lcd_data_list_train) * BATCH_UPDATES)

    return total_loss, avg_loss
    

def val(lcd_model:LCDModel, 
          lcd_data_list_val:list, 
          ):   
    lcd_model.eval()
    total_loss = 0.0
    for data in lcd_data_list_val:
        data.to_torch_seq()
        data.to_device(DEVICE)
                
        out = lcd_model(data)
        total_loss += loss_fn(out, data.gt).item()
        
        data.release_images()
    
    avg_loss = total_loss / len(lcd_data_list_val)
    
    return total_loss, avg_loss

def evaluate(lcd_model:LCDModel, 
          lcd_data_list_test:list, 
          ):   
    lcd_model.eval()
    success_num = 0
  
    for data in lcd_data_list_test:
        data.to_torch_seq()
        data.to_device(DEVICE)
        
        out = lcd_model(data)
        predicted_label = 1.0 if out.item() >= MATCH_THREASH else 0.0
        if(predicted_label == data.gt.item()):
            success_num += 1
        
        data.release_images()
            
    success_rate = success_num / len(lcd_data_list_test)

    return success_num, success_rate


print(f'start training ...')  
train_avg_loss_list, val_avg_loss_list, success_rate_list = [], [], []
last_success_rate = 0
for epoch in range(EPOCHES):
    print(f'start epoch: {epoch}')
    start_time = time.time()
    
    print(f'training ...')
    train_total_loss, train_avg_loss = \
        train(lcd_model, optimizer, loss_fn, train_data_list, )
        
    print(f'validating ...')
    val_total_loss, val_avg_loss = \
        val(lcd_model, val_data_list)
        
    print(f'evaluating ...')
    success_num, success_rate = \
        evaluate(lcd_model, test_data_list)    

    if(success_rate >= last_success_rate):
        model_info_str = 'bestmodel' + '_epoch_' + str(epoch) + '_sr_' + str(success_rate) + ".pth"
        model_save_path = os.path.join(trial_path, model_info_str)
        torch.save(lcd_model, model_save_path)
        last_success_rate = success_rate

    train_avg_loss_list.append(train_avg_loss)
    val_avg_loss_list.append(val_avg_loss)
    success_rate_list.append(success_rate)
    
    end_time = time.time()
    time_span = end_time - start_time    
    
    nni_metric = {}
    nni_metric['default'] = success_rate
    nni_metric['train_avg_loss'] = train_avg_loss
    nni_metric['val_avg_loss'] = val_avg_loss
    nni_metric['epoch'] = epoch
    nni.report_intermediate_result(nni_metric)
    
    print(f'finish epoch {epoch}. training time: {time_span}.')
    print(f'avg train loss: {train_avg_loss}. avg val loss: {val_avg_loss}.')
    # print(f'avg train loss: {train_avg_loss}.')
    print(f'success_rate: {success_rate}.')
    print('\n')


np_train_avg_loss = np.array(train_avg_loss_list)
np_val_avg_loss = np.array(val_avg_loss_list)
np_sr = np.array(success_rate_list) 

max_sr = np.max(np_sr)
max_sr_epoch = np.argmax(np_sr)
relat_train_avg_loss = np_train_avg_loss[max_sr_epoch]
relat_val_avg_loss = np_val_avg_loss[max_sr_epoch]

nni_metric = {}
nni_metric['default'] = max_sr
nni_metric['train_avg_loss'] = relat_train_avg_loss
nni_metric['val_avg_loss'] = relat_val_avg_loss
nni_metric['epoch'] = max_sr_epoch

nni.report_final_result(nni_metric)


#### save experiment data to txt
train_avg_loss_path = os.path.join(trial_path, 'train_avg_loss.txt')
val_avg_loss_path = os.path.join(trial_path, 'val_avg_loss.txt')
sr_path = os.path.join(trial_path, 'success_rate.txt')

np.savetxt(train_avg_loss_path, np_train_avg_loss, delimiter=' ')
np.savetxt(val_avg_loss_path, np_val_avg_loss, delimiter=' ')
np.savetxt(sr_path, np_sr, delimiter=' ')
