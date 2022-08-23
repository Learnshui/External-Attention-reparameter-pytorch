import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
import cv2
import os
import numpy as np

from time import time
from PIL import Image
import glob

import warnings

warnings.filterwarnings('ignore')

from networks.tanet import TA_Net_
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
#堆叠的都是同一个切片的相应变换(448,448,3),然后再将结果进行一系列变换，再与金标准比较，不是很懂
BATCHSIZE_PER_CARD = 8


def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()


    label_1D = label_1D / 255

    auc = metrics.roc_auc_score(label_1D, result_1D)

    # print("AUC={0:.4f}".format(auc))

    return auc

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)#转变类型
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    dice = (2*TP + 0.01)/(FP + FN + 2*TP + 0.01)
    return acc,sen, dice

def read_lits_validset(root_path):
    root_path = root_path + '/img'
    # fold = os.listdir(root_path)
    img_list = []
    label_list = []
    
    # val_list = root_path.replace('trainingset', 'validset')
    fold_s = os.listdir(root_path)

    for item in fold_s:
        img_list += (glob.glob(os.path.join(root_path, item) + '/*.npy'))
        img_list = sorted(img_list, key=lambda x: (int(x.split('/')[-2].split('-')[-1]), int(x.split('-')[-1].split('.')[0])))
    label_list = [x.replace('img', 'gt').replace('volume', 'segmentation') for x in img_list]
    # print(img_list)
    # print(label_list)
    return img_list, label_list


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        # print('******batchsize********', batchsize)

        return self.test_one_img_lits(path)

    def test_one_img_lits(self, path): ###
        # img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = np.load(path)
        img  = cv2.resize(img, (448,448), cv2.INTER_LINEAR)
        img = np.stack((img,img,img), axis=2) #堆叠的都是同一个切片的相应变换(448,448,3)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])#(2,448,448,3)
        img2 = np.array(img1)[:, ::-1] #(2,448,448,3)
        img3 = np.concatenate([img1, img2])#(4,448,448,3)
        img4 = np.array(img3)[:, :, ::-1]#(4,448,448,3)
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)##(8,3,448,448)
        img5 = np.array(img5, np.float32) / 100.0 ##(8,3,448,448)
        img5 = V(torch.Tensor(img5).cuda())#(8,3,448,448)

        mask= self.net.forward(img5)#mask,_ = self.net.forward(img5) (8,1,448,448)
        mask = mask.squeeze().cpu().data.numpy()  # .squeeze(1)（8，448，448）
        #mask = self.net.forward(img5)
        mask1 = mask[:4] + mask[4:, :, ::-1] #（4，448，448）
        mask2 = mask1[:2] + mask1[2:, ::-1]#（2，448，448）
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1] #（448，448）

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model,strict=False)


def test_tanet_lits():
    source = '/home/datacenter/hdd/data_dadong/Lits_dataset_crop/validset'
    relog = open('/home/shuiyuanyuan/TANet_for_Lits/test_results_lits/Evaluation_metrics.txt','a+')
    img_list, label_list = read_lits_validset(source)
    # val = os.listdir(source)
    
    solver = TTAFrame(TA_Net_)
    #solver.load('weights/CE-NetDRIVE_normal.th')
    solver.load('/home/shuiyuanyuan/weights/TA-Net-Lits_plus_spatial_multi.th')
    #solver.load('weights/TA-Net_noAugDRIVE_plus_spatial_multi.th')
    tic = time()
    #target = 'results_Colon_test_cenet/'
    #target = 'results_Colon_test_channel_spatial_multi/'
    target = '/home/shuiyuanyuan/TANet_for_Lits/test_results_lits/TANet/'

    if not os.path.exists(target):
        os.mkdir(target)
    # gt_root = '/home/datacenter/ssd1/data_dadong/New_Lits_dataset_tumor/validaset'
    total_m1 = 0

    hausdorff = 0
    total_acc = []
    total_sen = []
    total_dice = []
    threshold = 0.05
    disc = 0.2
    total_auc = []

    for i, slice in enumerate(img_list[30:50]):  #10个切片

        mask = solver.test_one_img_from_path(slice) #（448，448）
        #mask = mask * 255.0 #pang
        # new_mask = mask.copy()
        mask = mask / 8.0
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        ground_truth_path = slice.replace('img','gt').replace('volume','segmentation')
        # ground_truth = np.array(Image.open(ground_truth_path))
        ground_truth = np.load(ground_truth_path) #（512，512）
        # ground_truth = cv2.resize(ground_truth, (448,448), cv2.INTER_NEAREST)
        ground_truth[ground_truth<1]=0
        ground_truth[ground_truth>=1]=1

        mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

        acc,sen,dice = accuracy(mask, ground_truth)
        total_acc.append(acc)
        total_sen.append(sen)
        total_dice.append(dice)

        print('The {} image--Accuracy:{},Dice coeff:{}'.format(i + 1, acc, dice))

        cv2.imwrite(target + slice.split('/')[-1].split('.')[0] + '-mask.png', mask*255.0)
        cv2.imwrite(target + slice.split('/')[-1].split('.')[0] + '-label.png', ground_truth*255.0)
    print(np.mean(total_acc), np.std(total_acc))
    print(np.mean(total_sen), np.std(total_sen))
    print(np.mean(total_dice), np.std(total_dice))
    # print(np.mean(total_auc), np.std(total_auc))
    relog.write('TANet metrics: \n')
    relog.write('Mean accuracy: {}--Mean Std: {} \n'.format(np.mean(total_acc), np.std(total_acc)))
    relog.write('Mean sencitivity: {}--Mean Std: {} \n'.format(np.mean(total_sen), np.std(total_sen)))
    relog.write('Mean dice: {}--Mean Std: {} \n'.format(np.mean(total_dice), np.std(total_dice)))



if __name__ == '__main__':
    # test_ce_net_vessel() 
    test_tanet_lits()
