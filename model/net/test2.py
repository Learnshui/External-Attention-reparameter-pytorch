import os
import torch
import argparse
import numpy as np
# import pandas as pd
import torch.utils.data as Data
# from utils.binary import assd
from distutils.version import LooseVersion
from networks.tanet import TA_Net_
from networks.CANet import CANet
from networks.CEnet import CE_Net_
from datasets.Lits import LiTS_liver
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from time import *
import visdom
from matplotlib.image import imsave
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
Test_Model = {'CPFNet': CE_Net_}

Test_Dataset = {'Lits': LiTS_liver}

def Test_isic(test_loader, model):
    F1 = []
    ACC = []
    Sen = []
    Spe = []
    Jaccard = []
    Dice = []
    print(len(test_loader))
    model.eval()
    for step, (img, lab) in enumerate(test_loader):
        image = img.float().cuda()   #shape of image is [1,3,448,448]  torch.float32
        target = lab.float().cuda()   #shape of image is [1,1,448,448]  torch.float32
        output = model(image)  #output shape [1,2,192,256] torch.float32
        output[output > 0.1] = 1
        output[output <= 0.1] = 0
        predict_dis = output
        label_dis = target
        predict_save = predict_dis.squeeze().data.cpu().numpy().astype(np.uint8)
        label_save = label_dis.squeeze().data.cpu().numpy().astype(np.uint8)

        acc_score = accuracy_score(label_save.flatten(), predict_save.flatten())
        TP = ((predict_save==1)*(label_save==1)).sum()
        FP = (predict_save==1).sum() - TP
        FN = (label_save==1).sum() - TP
        TN = ((predict_save==0)*(label_save==0)).sum()
        smooth = 0.01
        sen = (TP + smooth)/(TP + FN + smooth)
        spe = (TN + smooth)/(TN + FP + smooth)
        jac = (TP + smooth) / (TP + FP + FN + smooth)
        intersect = (predict_save * label_save).sum()
        dice = (2*intersect + smooth)/(smooth + predict_save.sum() + label_save.sum())
        if not os.path.exists('/home/shuiyuanyuan/try_segmentation/result/CPFNet/Lits'):
            os.makedirs('/home/shuiyuanyuan/try_segmentation/result/CPFNet/Lits/predict')
            os.makedirs('/home/shuiyuanyuan/try_segmentation/result/CPFNet/Lits/label')
        imsave('/home/shuiyuanyuan/try_segmentation/result/CPFNet/Lits/predict/predict_{}.png'.format(step), predict_save, cmap='gray')
        imsave('/home/shuiyuanyuan/try_segmentation/result/CPFNet/Lits/label/label_{}.png'.format(step), label_save, cmap='gray')
        print('Accuracy:{}----Sensitivity score:{}----Jaccard score:{}----Dice score:{}'.format(acc_score, sen, jac, dice))
        ACC.append(acc_score)
        Sen.append(sen)
        Spe.append(spe)
        Jaccard.append(jac)
        Dice.append(dice)
    return F1, ACC, Sen, Spe, Jaccard, Dice



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='CPFNet',
                        help='a name for identitying the model. Choose from the following options: Unet')

    # Path related arguments
    parser.add_argument('--root_path', default='/home/datacenter/hdd/data_dadong/Lits_dataset_crop/validset',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='/home/shuiyuanyuan/try_segmentation/run/lits',
                        help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=50, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--mode', default='test', help='training mode')
    parser.add_argument('--data', default='Lits', help='choose the dataset')

    args = parser.parse_args()

    # loading the dataset
    print('loading the {0} dataset ...'.format('test'))
    testset = Test_Dataset[args.data](dataset_path=args.root_path, mode='val')
    testloader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True)
    print('Loading is done\n')

    # Define model
    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        model = Test_Model[args.id](args.num_input,args.num_classes).cuda()
        # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Load the trained best model
    modelname = args.ckpt + '/' + 'min_loss_Lits_checkpoint.pth' #+ '_' + args.data + '_checkpoint.pth.tar'
    # modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        # checkpoint = torch.load(modelname)
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(torch.load(modelname), strict=False)
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        # print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    F1, ACC, Sen, Spe, Jaccard, Dice = Test_isic(testloader, model)

    file = open('/home/shuiyuanyuan/try_segmentation/test_results_lits/Evaluation_CPFnet_metrics.txt', 'a+')
    file.write('CPFNet Metrix:({})-{}'.format(0, datetime.datetime.now()) + '\n')
    str2 = 'The ACC score:{}'.format(np.sum(ACC) / len(ACC))
    # str3 = 'The AUC score:{}'.format(np.sum(AUC) / len(AUC))
    str4 = 'The Sencitivity score:{}'.format(np.sum(Sen) / len(Sen))
    str5 = 'The Specificity score:{}'.format(np.sum(Spe) / len(Spe))
    str6 = 'The Jaccord score:{}'.format(np.sum(Jaccard) / len(Jaccard))
    str7 = 'The Dice score:{}'.format(np.sum(Dice) / len(Dice))
    file.write(str2 + '\n' + str4 + '\n' + str5 + '\n' + str6 + '\n' + str7 + '\n')
    file.close()
    print(str2)
    print(str4)
    print(str5)
    print(str6)
    print(str7)
