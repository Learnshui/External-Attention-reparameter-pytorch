import glob
import numpy as np
import torch
import os
import cv2
import csv
import tqdm
from models.models import EANet
from matplotlib.image import imsave
from sklearn.metrics import accuracy_score
from train.evaluation import *

#可以
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

if __name__ == "__main__":

    fold=1
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    
    net =EANet(n_channels=3, n_classes=1)
    net.to(device=device)
    # 加载训练模型参数
    # net.load_state_dict(torch.load('/home/shuiyuanyuan/EANet/results/eanet_best.pth'))  
    # solution 2
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/home/shuiyuanyuan/EANet/results/eanet_best.pth').items()})
  
    # 测试模式
    net.eval()
    # 读取所有图片路径
   
    tests_path = '/home/datacenter/hdd/data_dadong/Lits_dataset_crop/validset'

    acc = 0.	# Accuracy
    count = 0
    # f = open('/home/shuiyuanyuan/EANet/test_results_lits/CEnet.csv', 'w')
    # f.write('name,JS,DC'+'\n')
    img_list, label_list = read_lits_validset(tests_path)
    AUC_score = 0
    sen = 0
    spe = 0
    jac = 0
    dice = 0
    acc_score = 0
    infer_time = []
    k=0
    for i, slice in enumerate(img_list):#[20:80]
        k=k+1
        img=np.load(slice)
        img_cur  = cv2.resize(img, (448,448), cv2.INTER_LINEAR)
        img = np.stack((img_cur,img_cur,img_cur), axis=2) #堆叠的都是同一个切片的相应变换(448,448,3)
        # img = img.reshape(1,3,448,448)
        img = np.expand_dims(img, axis=0)
        img = img.transpose((0,3,1,2))
        img = torch.Tensor(img).cuda()
        pred, p1,p2,p3,p4,e= net(img)
        mask = slice.replace('img','gt').replace('volume','segmentation')
        mask = np.load(mask)#array
        mask = torch.from_numpy(mask).cuda()#tensor
        mask[mask<1]=0
        mask[mask>=1]=1#(448, 448)
        mask1 = np.array(mask.data.cpu())
        sig = torch.nn.Sigmoid()
        pred = sig(pred)
        
        #保存结果
        # 提取结果
        pred1 = np.array(pred.data.cpu()[0])[0]#(448, 448)
        pred = pred.squeeze()
        pred1[pred1 >= 0.5] = 1
        pred1[pred1 < 0.5] = 0

        img_cur = torch.from_numpy(img_cur).cuda()
        img_cur = np.array(img_cur.data.cpu())
        # img_cur=img_cur.squeeze().data.cpu().numpy().astype(np.uint8)
        # img = pred1#(448, 448)
        # predict_save = img.squeeze().data.cpu().numpy().astype(np.uint8)
        # label_save = mask.squeeze().data.cpu().numpy().astype(np.uint8)
        if not os.path.exists('/home/shuiyuanyuan/EANet/test_results_lits/Lits'):
            os.makedirs('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict1')
        #     os.makedirs('./result/NewMLFPNet2/predict')
            # os.makedirs('/home/shuiyuanyuan/EANet/test_results_lits/Lits/label')
        # matplotlib.image.imsave('./result/NewMLFPNet2/color/color_{}.png'.format(step), color_label)
        if 300<=k<=400:
            imsave('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict1/{}_predict.png'.format(i), pred1*255.0, cmap='gray')
            imsave('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict1/{}_label.png'.format(i), mask1*255.0, cmap='gray')
            imsave('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict1/{}_image.png'.format(i), img_cur, cmap='gray')
        acc_score += accuracy_score(mask1.flatten(), pred1.flatten())
        TP = ((pred1==1)*(mask1==1)).sum()
        FP = (pred1==1).sum() - TP
        FN = (mask1==1).sum() - TP
        TN = ((pred1==0)*(mask1==0)).sum()
        smooth = 0.01
        sen += (TP + smooth)/(TP + FN + smooth)
        spe += (TN + smooth)/(TN + FP + smooth)
        jac += (TP + smooth) / (TP + FP + FN + smooth)

        # AUC = roc_auc_score(label_save.flatten(), predict_save.flatten(), average='macro')


        intersect = (pred1 * mask1).sum()
        dice += (2*intersect + smooth)/(smooth + pred1.sum() + mask1.sum())
      
        count+=1
    acc = acc_score/count
    SE = sen/count
    SP = spe/count
    # PC = PC/count
    # F1 = F1/count
    JS = jac/count
    DC = dice/count

    print('ACC:%.4f' % acc)
    print('SE:%.4f' % SE)
    print('SP:%.4f' % SP)
    # print('PC:%.4f' % PC)
    # print('F1:%.4f' % F1)
    print('JS:%.4f' % JS)
    print('DC:%.4f' % DC)
  
# ACC:0.9894
# SE:0.8549
# SP:0.9947
# JS:0.8172
# DC:0.8552




# python<predict.py>sce2.txts
#by kun wang
