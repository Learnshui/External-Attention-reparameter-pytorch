#在laddernet里面，Fork过来了已经，还有ROC曲线的画法
visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()
#visualize results comparing mask and prediction:
assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    total_img = np.concatenate((orig_stripe,masks_stripe,pred_stripe),axis=0)
    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(i))#.show()
    
    
    
    
    
#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img


#ROC曲线画，test中
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
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from lib.TO_ROC import pred_only_FOV
#可以
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def pred_only_FOV(data_imgs,data_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                new_pred_imgs.append(data_imgs[i,:,y,x])
                new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks
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
    preds = []
    masks = []
    for i, slice in enumerate(img_list[0:200]):#[20:80]
        name =img_list[i].split('/')[-1].split('.')[-2]
        k=k+1
        img=np.load(slice)
        img_cur  = cv2.resize(img, (448,448), cv2.INTER_LINEAR)
        img = np.stack((img_cur,img_cur,img_cur), axis=2) #堆叠的都是同一个切片的相应变换(448,448,3)
        # img = img.reshape(1,3,448,448)
        img = np.expand_dims(img, axis=0)
        img = img.transpose((0,3,1,2))
        img = torch.Tensor(img).cuda()
        pred, p1,p2,p3,p4,e= net(img)
        pred_show = pred
        mask = slice.replace('img','gt').replace('volume','segmentation')
        mask = np.load(mask)#array
        mask = torch.from_numpy(mask).cuda()#tensor
        mask[mask<1]=0
        mask[mask>=1]=1#(448, 448)
        mask_show = mask.reshape([1,1,mask.shape[0],mask.shape[1]])
  
        mask1 = np.array(mask.data.cpu())
        sig = torch.nn.Sigmoid()
        pred = sig(pred)
        pred_show = sig(pred_show)
   
        pred_show =pred_show.data.cpu().numpy()
        preds.append(pred_show)
        mask_show =mask_show.data.cpu().numpy()
        masks.append(mask_show)

        #保存结果
        # 提取结果
        pred1 = np.array(pred.data.cpu()[0])[0]#(448, 448)
        pred = pred.squeeze()
        pred1[pred1 >= 0.4] = 1
        pred1[pred1 < 0.4] = 0
   
        img_cur = torch.from_numpy(img_cur).cuda()
        img_cur = np.array(img_cur.data.cpu())
        # img_cur=img_cur.squeeze().data.cpu().numpy().astype(np.uint8)
        # img = pred1#(448, 448)
        # predict_save = img.squeeze().data.cpu().numpy().astype(np.uint8)
        # label_save = mask.squeeze().data.cpu().numpy().astype(np.uint8)
        if not os.path.exists('/home/shuiyuanyuan/EANet/test_results_lits/Lits'):
            os.makedirs('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict')
        #     os.makedirs('./result/NewMLFPNet2/predict')
            # os.makedirs('/home/shuiyuanyuan/EANet/test_results_lits/Lits/label')
        # matplotlib.image.imsave('./result/NewMLFPNet2/color/color_{}.png'.format(step), color_label)
        # if 70<=k<=80:
        #     imsave('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict/{}_predict.png'.format(name), pred1*255.0, cmap='gray')
        #     imsave('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict/{}_label.png'.format(name), mask1*255.0, cmap='gray')
        #     imsave('/home/shuiyuanyuan/EANet/test_results_lits/Lits/predict/{}_image.png'.format(name), img_cur, cmap='gray')
   
        acc_score += accuracy_score(mask1.flatten(), pred1.flatten())
        TP = ((pred1==1)*(mask1==1)).sum()
        FP = (pred1==1).sum() - TP
        FN = (mask1==1).sum() - TP
        TN = ((pred1==0)*(mask1==0)).sum()
        smooth = 0.01
        sen += (TP + smooth)/(TP + FN + smooth)
        spe += (TN + smooth)/(TN + FP + smooth)
        jac += (TP + smooth) / (TP + FP + FN + smooth)
        intersect = (pred1 * mask1).sum()
        dice += (2*intersect + smooth)/(smooth + pred1.sum() + mask1.sum())
      
        count+=1
    predictions = np.concatenate(preds,axis=0)
    masks = np.concatenate(masks,axis=0)
    y_scores, y_true = pred_only_FOV(predictions,masks)
    fpr, tpr, thresholds = metrics.roc_curve((y_true), y_scores)
    AUC_ROC = metrics.roc_auc_score(y_true, y_scores)
    print("\nArea under the ROC curve: " +str(AUC_ROC))
    roc_curve =plt.figure()
    plt.plot(fpr,tpr,'--',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig("ROC.png")

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
    print('JS:%.4f' % JS)
    print('DC:%.4f' % DC)

