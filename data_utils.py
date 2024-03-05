#coding=utf-8
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torchvision.transforms as transforms
import os
from osgeo import gdal
import random

def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
'''
functions for evaluate the network
'''
def cal_F1(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(np.logical_and(predict == 1, label == 0))
    fn = np.sum(np.logical_and(predict == 0, label == 1))
    tn = np.sum(np.logical_and(predict == 0, label == 0))

    return tp, fp, fn, tn
def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp
'''
functions for read tif images
'''
def read_tif(path,size):
    ds = gdal.Open(path)
    bands_num = ds.RasterCount
    shape = (bands_num,size[0],size[1])
    img = np.zeros( shape=shape,dtype=float)
    for i in range(bands_num):
        band = ds.GetRasterBand(i+1)
        img[i,:,:] = band.ReadAsArray(buf_xsize=size[1],buf_ysize=size[0])#note that the x-axis mains col
    return img


def preprocess(image,image_type):
    image[np.isnan(image)]=0
    if image_type =='optical':
        image = image/255.0
        image = torch.tensor(image, dtype=torch.float32)
    elif image_type == 'sar':
        image = image/10*(-1)
        image = torch.tensor(image, dtype=torch.float32)
    elif image_type =='label':
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.where(image > 0, torch.tensor(1.0), image)
    return image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.tif', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)



def getSampleLabel(img_path):
    img_name = img_path.split('\\')[-1]
    return torch.from_numpy(np.array([int(img_name[0] == 'i')], dtype=np.float32))

def getDataList(img_path):
    dataline = open(img_path, 'r').readlines()
    datalist =[]
    for line in dataline:
        temp = line.strip('\n')
        datalist.append(temp)
    return datalist

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result

def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, hr2_path, opt_path, s1s2_path,lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]
        self.opt_filenames = [join(opt_path, x) for x in datalist if is_image_file(x)]
        self.otp_filenames = [join(s1s2_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):
        hr1_img = self.transform(gdal.Open(self.hr1_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        hr2_img = self.transform(gdal.Open(self.hr2_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        opt_img = self.transform(gdal.Open(self.opt_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        s1s2_img = self.transform(gdal.Open(self.s1s2_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        label = self.label_transform(gdal.Open(self.lab_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, hr2_img,opt_img ,s1s2_img,label

    def __len__(self):
        return len(self.hr1_filenames)

class LoadDatasetFromFolder_test(Dataset):
    def __init__(self, args, hr1_path, hr2_path):
        super(LoadDatasetFromFolder_test, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.opt_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.s1s2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]

    def __getitem__(self, index):
        hr1_img = self.transform(gdal.Open(self.hr1_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        hr2_img = self.transform(gdal.Open(self.hr2_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        opt_img = self.transform(gdal.Open(self.opt_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        s1s2_img = self.transform(gdal.Open(self.s1s2_filenames[index]).ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32))
        return hr1_img, hr2_img,opt_img,s1s2_img

    def __len__(self):
        return len(self.hr1_filenames)


