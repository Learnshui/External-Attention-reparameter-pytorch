import re
import cv2
import os
import time

from sklearn.metrics import accuracy_score
import losses
import argparse
import torch.utils.data as Data
from datasets.Lits import LiTS_liver
import pickle
import torch
import torch.nn as nn
from tqdm import TqdmExperimentalWarning, tqdm
import torch.utils.data as data
from torch.autograd import Variable as V
import sys
from networks.tanet import TA_Net_
from networks.CANet import CANet
from networks.HHnet import HHnet
from networks.CEnet import CE_Net_
# from networks.CANet import CANet
from utils.evalue import evalue
from loss import dice_bce_loss
#from loss import dice_loss
from data import ImageFolder
from utils.logger import Logger
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
#import image_utils
savepath='/home/shuiyuanyuan/try_segmentation/intermediate_results/'
import numpy as np
from models.sync_batchnorm.replicate import patch_replication_callback

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
def train_Dice_function(input_, target):
        # pre_sum = torch.sum(predict)
        # mask_sum = torch.sum(mask)
        # inter = torch.sum(predict*mask)
        # Dice = 2*inter/(pre_sum + mask_sum)
        smooth = 1.
        input_flat = input_.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = input_flat * target_flat
        dice = (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return dice
class Trainer(object):
    def __init__(self, args):
        self.args = args
        model=HHnet()
        self.model=model
        # Define Saver
        # Recoder the running processing
        self.saver = Saver(args)
        sys.stdout = Logger(
            os.path.join(self.saver.experiment_dir, 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S"))) 
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.criterion = nn.BCELoss()
        # self.criterion = losses.DiceLoss()
        trainset = LiTS_liver(dataset_path=args.ROOT, mode='train')
        validset = LiTS_liver(dataset_path=args.ROOTVAL, mode='val')

        train_data_loader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_data_loader = Data.DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        # Define Evaluator
        NAME = 'HHnet_-Lits'
        mylog = open('/home/shuiyuanyuan/try_segmentation/logs/' + NAME + '.log', 'w')
        self.mylog=mylog
        optimizer =torch.optim.Adam(self.model.parameters(), weight_decay=args.weight_decay)
        self.model, self.optimizer ,self.train_data_loader,self.val_data_loader= model, optimizer,train_data_loader,val_data_loader
        self.evaluator = Evaluator(args.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_data_loader))
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint  
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self,epoch):
        train_loss = 0.0
        self.model.train()
        # print the total number of parameters in the network
        # solver.paraNum() 
        # load model
        #solver.load('./weights/' + NAME + '_plus_spatial_multi.th')
        # start the logging files
        tic = time.time()
        F1 = []
        ACC = []
        AUC_score = []
        Sen = []
        Spe = []
        Jaccard = []
        Dice = []
        infer_time = []
        # data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        train_dice = 0
        index = 0
        tbar = tqdm(self.train_data_loader)
        num_img_tr = len(self.train_data_loader)
        for i,(img, mask) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output= self.model(img)#
            output=output.float().cuda()
            mask=mask.float().cuda()
            loss = self.criterion(output, mask)#loss里面是否加softmax
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_dice = train_dice + train_Dice_function(output, mask).item()
            # train_dice=train_dice+1-loss
            index = index + 1
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))#loss的呈现在不停的变
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            if i % (len(self.train_data_loader) // 10) == 0:
                global_step = i + len(self.train_data_loader) * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, img, mask, output, global_step)
            
            acc_score,sen,spe,jac,dice=evalue(output,mask)

            ACC.append(acc_score)
            Sen.append(sen)
            Spe.append(spe)
            Jaccard.append(jac)
            Dice.append(dice)
                # str1 = 'The f1 score:{}'.format(np.sum(F1) / len(F1))
        
        # file  = open('./valid_metrix_Lits.txt', 'a+')
        # self.mylog.write('valid:({})-{}'.format(0, datetime.datetime.now()) + '\n')
        train_epoch_loss = train_loss/len(self.train_data_loader)
        train_dice_avg = train_dice /len(self.train_data_loader)
        str0=train_epoch_loss

        str2 = np.sum(ACC) / len(ACC)
        # str3 = 'The AUC score:{}'.format(np.sum(AUC) / len(AUC))
        str4 = np.sum(Sen) / len(Sen)
        str5 =np.sum(Spe) / len(Spe)
        str6 = np.sum(Jaccard) / len(Jaccard)
        str7 = np.sum(Dice) / len(Dice)
        print('train  Accuracy:{}----Sensitivity score:{}----Specificity score:{}----Jaccard score:{}----Dice score:{}'.format(str2, str4,str5, str6, str7))
        if self.args.no_val:
        # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
              'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
         }, is_best)

        print('epoch:', epoch, '    time before imwrite:', int(time.time() - tic))
        
       
        print('train_dice_avg', train_dice_avg)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/total_dice_epoch', str7, epoch)
        self.writer.add_scalar('train/total_jaccard_epoch', str6, epoch)
        self.writer.add_scalar('train/train_acc_avg', str2, epoch)
        print('********')
        print('epoch:', epoch, '    time:', int(time.time() - tic))
        print('totalNum in an epoch:',index)
        print('train_loss:', train_epoch_loss)
   
        return train_epoch_loss,str0,str7,str2 ,  str4 , str5 ,str6

    def validation(self, epoch,minloss,max_dice,args):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_data_loader, desc='\r')
        length=len(self.val_data_loader)
        test_loss = 0.0
        val_time = 0
        ACC = []
        Sen = []
        Spe = []
        Jaccard = []
        Dice = []
        for i,(img, mask) in enumerate(tbar):
            if self.args.cuda:
                img, mask = img.float().cuda(), mask.float().cuda()#【8，1，512，512】
            with torch.no_grad():
                start = time.time()
                outputs  = self.model(img)
                end = time.time()
                val_time += end-start
            
            loss = self.criterion(outputs, mask)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            acc_score,sen,spe,jac,dice=evalue(outputs,mask)

            ACC.append(acc_score)
        # AUC_score.append(AUC)
            Sen.append(sen)
            Spe.append(spe)
            Jaccard.append(jac)
            Dice.append(dice)
                # str1 = 'The f1 score:{}'.format(np.sum(F1) / len(F1))
        
        # file  = open('./valid_metrix_Lits.txt', 'a+')
        # self.mylog.write('valid:({})-{}'.format(0, datetime.datetime.now()) + '\n')
        str0=test_loss/length
        str2 = np.sum(ACC) / len(ACC)
        # str3 = 'The AUC score:{}'.format(np.sum(AUC) / len(AUC))
        str4 = np.sum(Sen) / len(Sen)
        str5 =np.sum(Spe) / len(Spe)
        str6 = np.sum(Jaccard) / len(Jaccard)
        str7 = np.sum(Dice) / len(Dice)
        print('valid  Accuracy:{}----Sensitivity score:{}----Specificity score:{}----Jaccard score:{}----Dice score:{}'.format(str2, str4,str5, str6, str7))
        self.writer.add_scalar('valid/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('valid/vaild_dice_epoch', str7, epoch)
        self.writer.add_scalar('valid/total_jaccard_epoch', str6, epoch)
        self.writer.add_scalar('valid/valid_acc_avg', str2, epoch)
        if str0 < min(minloss):
            minloss.append(str0)
            print(minloss)
            modelname = self.saver.experiment_dir + '/' + 'min_loss' + '_' + args.checkname + '_checkpoint.pth'
            print('the best model will be saved at {}'.format(modelname))
            # state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
            torch.save(self.model.state_dict(), modelname)
        if str7 > max(max_dice):
            max_dice.append(str7)
            print(max_dice)
            modelname = self.saver.experiment_dir+ '/' + 'max_dice' + '_' + args.checkname + '_checkpoint.pth'
            print('the best model will be saved at {}'.format(modelname))
            # state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
            torch.save(self.model.state_dict(), modelname)
        new_pred = str7
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        return str0,str7 , str2 ,  str4 , str5 ,str6


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--Image-size', type=int, default=(448, 448),
                        help='')
    parser.add_argument('--nclass', type=int, default=2,
                        help='')
    parser.add_argument('--NUM-EARLY-STOP', type=int, default=20,
                        help='')
    parser.add_argument('--NUM-UPDATE-LR', type=int, default=10,
                        help='')
    parser.add_argument('--ckpt', default='./run/lits/saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--INITAL-EPOCH-LOSS', type=int, default=1000,
                        help='')
    parser.add_argument('--ROOT', type=str, default='/home/datacenter/hdd/data_dadong/Lits_dataset_crop/trainingset',
                        help='')
    parser.add_argument('--ROOTVAL', type=str, default='/home/datacenter/hdd/data_dadong/Lits_dataset_crop/validset',
                        help='')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='lits',
                        choices=['lits', 'pairwise_chaos'],
                        help='dataset name (default: pairwise_lits)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
  
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--backbone', type=str, default='tanet',
                        metavar='N', help='input batch size for \
                                testing (default: auto)')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='whether to use pretrained backbone (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='HHnet',
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--only-val', action='store_true', default=False,
                        help='just evaluation but not training')
    parser.add_argument('--save-predict', action='store_true', default=False,
                        help='save predicted mask in disk')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    
    if args.only_val:
        if args.resume is not None:
            trainer.validation(trainer.args.start_epoch)
            return
        else:
            raise NotImplementedError
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    no_optim = 0
    train_epoch_best_loss = 100000
    old_lr = args.lr
    minloss = [1.0]
    max_dice = [0]
    start = time.time()
    trainer.mylog.write('epoch,loss,dice,acc,sen,spe,jac \n')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_epoch_loss,str0,str7 , str2 ,  str4 , str5 ,str6=trainer.training(epoch)
        trainer.mylog.write('training:{}' .format(epoch))
        trainer.mylog.write(' {} '.format(str0))
        trainer.mylog.write(' {} '.format(str7))
        trainer.mylog.write(' {} '.format(str2))
        trainer.mylog.write(' {} '.format(str4))
        trainer.mylog.write(' {} '.format(str5))
        trainer.mylog.write(' {}  \n'.format(str6))
        
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            torch.save(trainer.model.state_dict(), args.ckpt + '/' + trainer.NAME + '_plus_spatial_multi.th')
        if no_optim > 100:
            print(trainer.mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 50:
            if old_lr < 1e-8:
                break
            trainer.model.load_state_dict(torch.load(args.ckpt + '/' + trainer.NAME + '_plus_spatial_multi.th'), strict=False)
            new_lr = old_lr * 0.1
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            trainer.mylog.write('update learning rate: %f -> %f\n' % (old_lr, new_lr))
            print('update learning rate: %f -> %f' % (old_lr, new_lr))
            old_lr = new_lr
            no_optim=0
        if not trainer.args.no_val :
            str0,str7 , str2 ,  str4 , str5 ,str6=trainer.validation(epoch,minloss,max_dice,args)
            
            trainer.mylog.write('validing:{}' .format(epoch))
            trainer.mylog.write(' {} '.format(str0))
            trainer.mylog.write(' {} '.format(str7))
            trainer.mylog.write(' {} '.format(str2))
            trainer.mylog.write(' {} '.format(str4))
            trainer.mylog.write(' {} '.format(str5))
            trainer.mylog.write(' {}  \n'.format(str6))
            
        trainer.mylog.flush()
    
    end = time.time()
    print('Training time is: ', end-start)
    trainer.writer.close()

if __name__ == '__main__':
    print(torch.__version__)
    main()


