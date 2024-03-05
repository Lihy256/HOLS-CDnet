# coding=utf-8
import argparse
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import *
from loss.BCL import BCL
from loss.DiceLoss import DiceLoss, make_one_hot
from HOLSCDnet import HOLSCDnet
# from decoder_rgb import decoder_rgb
import numpy as np
import random
from train_options import parser
from dataset import HRLSCDdataset
import logging
import cv2
import time

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() and len(opt.gpu_ids) > 0
                      else "cpu")
print(device)
# set seeds
seed_torch(2023)


# load train dataset
train_Set = HRLSCDdataset(root_dir=opt.root_dir,
                          split_dir=opt.split_dir,
                          set='train',  # target load City train dataset,
                          rgb_size=256,
                          sar_size=64)
train_loader = DataLoader(train_Set,
                          batch_size=opt.batch_size,
                          num_workers=opt.num_workers,
                          shuffle=True, drop_last=True)
# load val dataset
val_Set = HRLSCDdataset(root_dir=opt.root_dir,
                          split_dir=opt.split_dir,
                          set='val',  # target load City train dataset,
                          rgb_size=256,
                          sar_size=64)
val_loader = DataLoader(val_Set,
                          batch_size=opt.batch_size,
                          num_workers=opt.num_workers,
                          shuffle=True, drop_last=True)
# load val dataset
test_Set = HRLSCDdataset(root_dir=opt.root_dir,
                          split_dir=opt.split_dir,
                          set='test',  # target load City train dataset,
                          rgb_size=256,
                          sar_size=64)
test_loader = DataLoader(test_Set,
                          batch_size=1,
                          num_workers=opt.num_workers,
                          shuffle=True, drop_last=True)
#define the model
netCD = HOLSCDnet().cuda()

if opt.pretrained_path is not None:
    print("=> loading model '{}'".format(opt.pretrained_path))
    netCD.load_state_dict(torch.load(opt.pretrained_path))
    dir,pth = os.path.split(opt.pretrained_path)
    start_epoch = int(pth[pth.rfind('.')-1:pth.rfind('.')])+1
else:
    start_epoch = 1

optimizerCD = optim.Adam(netCD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerCD, T_max=opt.num_epochs)
BCELoss = torch.nn.BCEWithLogitsLoss().cuda()
DiceLoss = DiceLoss().cuda()


def trainVal():
    # training
    best_val_iou = 0
    for epoch in range(start_epoch, opt.num_epochs+1):
        all_batchsize = 0
        all_CD_loss = 0
        train_bar = tqdm(train_loader)
        netCD.train()
        logging.info('----------------train!-----------------')
        for name, image1, image2, sar, label in train_bar:
            image1 = image1.cuda()
            image2 = image2.cuda()
            sar = sar.cuda()
            label = label.long().cuda()
            label = make_one_hot(label.long(), 2)

            optimizerCD.zero_grad()
            outs = netCD(image1, image2, sar)
            out1 = outs['outrgb']
            out2 = outs['outfuse']
            Dice_loss1 = DiceLoss(out1, label)
            Dice_loss2 = DiceLoss(out2, label)
            Dice_loss = opt.wOut1*Dice_loss1+(1-opt.wOut1)*Dice_loss2

            BCE_loss1 = BCELoss(out1, label)
            BCE_loss2 = BCELoss(out2, label)
            BCE_loss = opt.wOut1*BCE_loss1+(1-opt.wOut1)*BCE_loss2

            CD_loss = opt.wBCE * BCE_loss + opt.wDice * Dice_loss
            CD_loss.backward()

            optimizerCD.step()
            scheduler.step()

            # loss for current batch before optimization
            all_batchsize += opt.batch_size
            all_CD_loss += CD_loss.item() * opt.batch_size

            train_bar.set_description(
                desc='[%d/%d] CD: %.4f  ' % (
                    epoch, opt.num_epochs, all_CD_loss/ all_batchsize,
                ))
        logging.info('----------------val!-----------------')
        netCD.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0, 0

            for name,image1, image2, sar, label in val_bar:
                image1 = image1.cuda()
                image2 = image2.cuda()
                sar = sar.cuda()
                label = label.long().cuda()

                outs = netCD(image1, image2, sar)
                out1 = outs['outrgb']
                out2 = outs['outfuse']
                out = opt.wOut1 * out1 + (1 - opt.wOut1) * out2

                cd_map = torch.argmax(out, 1)
                label = torch.squeeze(label, 1)

                pre = cd_map.cpu().detach().numpy()
                gt_value = label.cpu().detach().numpy()

                intr, unn = calMetric_iou(pre, gt_value)
                inter = inter + intr
                unin = unin + unn

                iou = (inter * 1.0 / unin)

                val_bar.set_description(
                    desc='IoU: %.4f' %iou)
        # save parameters of better model
        if iou>= best_val_iou:
            best_val_iou = iou
            pth_name = 'HOLSCDnet_epoch_{}.pth'.format(epoch)
            torch.save(netCD.state_dict(), os.path.join(opt.model_dir,pth_name))
            opt.best_model_path = os.path.join(opt.model_dir,pth_name)
            print('the best model is: '+pth_name)

########################################test#############################################
def test():
    print('---------------------------begin testing--------------------------')
    print('loading the best model',opt.best_model_path)
    netCD.load_state_dict(torch.load(opt.best_model_path))
    netCD.eval()
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        inter, unin = 0, 0
        tps, fps, fns,tns = 0, 0, 0,0
        all_time = 0
        image_num = 0

        for name,image1, image2, sar, label in test_bar:
            image1 = image1.cuda()
            image2 = image2.cuda()
            sar = sar.cuda()
            label = label.long().cuda()

            start_time = time.time()
            outs = netCD(image1, image2, sar)
            out1 = outs['outrgb']
            out2 = outs['outfuse']
            out = opt.wOut1 * out1 + (1 - opt.wOut1) * out2
            cd_map = torch.argmax(out, 1)
            all_time+=time.time()-start_time
            image_num+=1
            label = torch.squeeze(label, 1)

            pre = cd_map.cpu().detach().numpy()
            gt_value = label.cpu().detach().numpy()

            intr, unn = calMetric_iou(pre, gt_value)
            inter = inter + intr
            unin = unin + unn
            iou = (inter * 1.0 / unin)

            tp, fp, fn, tn = cal_F1(pre, gt_value)
            tps = tps + tp
            tns = tns + tn
            fps = fps + fp
            fns = fns + fn
            prec = tps / ((tps + fps) + 0.00000001)
            recal = tps / ((tps + fns) + 0.00000001)
            F1 = 2 * (prec * recal) / ((prec + recal) + 0.00000001)
            OA = (tps+tns)/(tps+tns+fps+fns)

            pre[pre == 1] = 255
            pre = np.transpose(pre,(1,2,0))
            predict_img_dir = os.path.join(opt.predict_img_dir,'HOLSCDnet test')
            if not os.path.exists(predict_img_dir):
                os.makedirs(predict_img_dir)
            cv2.imwrite(os.path.join(predict_img_dir,str(name[0])),pre)

            test_bar.set_description(
                desc='IoU: %.4f; F1:%.4f' % (iou,F1))
    print('precision:{}'.format(prec))
    print('recall:{}'.format(recal))
    print('iou:{}'.format(iou))
    print('F1:{}'.format(F1))
    print('OA:{}'.format(OA))
    print('predict time of one image ',all_time/image_num)

    txt_name = 'HOLSCDnet test'+'.txt'
    txt_path = os.path.join(opt.precision_txt_dir,txt_name)
    with open(txt_path, 'a') as file:
        file.write('precision:{}\n'.format(prec))
        file.write('recall:{}\n'.format(recal))
        file.write('iou:{}\n'.format(iou))
        file.write('F1:{}\n'.format(F1))
        file.write('OA:{}\n'.format(OA))
        file.write('predict time of one image:{}\n'.format(all_time/image_num))
        print('write txt successfully')




if __name__ == '__main__':
    trainVal()
    test()