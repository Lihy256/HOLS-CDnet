import numpy as np
import torch
from train_options import parser
from torch.utils.data import DataLoader
from torch.utils import data
from os.path import join
from data_utils import *
import torchvision.transforms as transforms

class HRLSCDdataset(data.Dataset):
    def __init__(self, root_dir, split_dir, set, rgb_size=256, sar_size=64):
        super(HRLSCDdataset, self).__init__()
        '''
            root_dir:  '/home/lhy/jqy/HRLSCD_project/dataset'
            split_dir: '/home/lhy/jqy/HRLSCD_project/dataset/train_val_test_txt'
            set：'train'/'test'/'val'
        '''
        self.rgb_size = rgb_size
        self.sar_size = sar_size
        self.image1_path = join(root_dir,'image1')
        self.image2_path = join(root_dir,'image2')
        self.sar_path = join(root_dir,'sar')
        self.label_path = join(root_dir,'label')
        self.txt_path = join(split_dir,set+'.txt')
        self.image_list = open(self.txt_path,'r')
        self.image_list = [x.rstrip() for x in self.image_list.readlines()]
    def __getitem__(self, index):
        image_name = self.image_list[index]+'.tif'
        image1_path = join(self.image1_path,image_name)
        image2_path = join(self.image2_path,image_name)
        sar_path = join(self.sar_path,image_name)
        label_path = join(self.label_path,image_name)
        image1 = read_tif(image1_path,size=[self.rgb_size,self.rgb_size])
        image2 = read_tif(image2_path,size=[self.rgb_size,self.rgb_size])
        sar = read_tif(sar_path,size=[self.sar_size,self.sar_size])
        label = read_tif(label_path,size=[self.rgb_size,self.rgb_size])
        image1 = preprocess(image1,'optical')
        image2 = preprocess(image2,'optical')
        sar = preprocess(sar,'sar')
        label = preprocess(label, 'label')
        image1 = torch.tensor(image1,dtype = torch.float)
        image2 = torch.tensor(image2,dtype = torch.float)
        sar = torch.tensor(sar,dtype = torch.float)
        label = torch.tensor(label,dtype = torch.float)
        return image_name,image1,image2, sar, label
    def __len__(self):
        return len(self.image_list)

if __name__=='__main__':
    opt = parser.parse_args()
    train_Set = HRLSCDdataset(root_dir=opt.root_dir,
                            split_dir=opt.split_dir,
                            set='train',  # target load City train dataset,
                            rgb_size=256,
                            sar_size=64)
    train_loader = DataLoader(train_Set,
                              batch_size=2,
                              num_workers=opt.num_workers,
                              shuffle=True, drop_last=True)
    for i, batch in enumerate(train_loader,1):
        name = batch[0]
        image1 = batch[1]
        image2 = batch[2]
        sar = batch[3]
        label = batch[4]
        print(name,image1.shape,image2.shape,sar.shape,label.shape)
        break