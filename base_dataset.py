from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data
from osgeo import gdal
# import spectral
import cv2
import torch


class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
        if max_iters is not None:
            self.img_ids = self.img_ids + self.img_ids[0:1]
        self.files = []
        for name in self.img_ids:
            imgrgb1_file,imgrgb2_file,imgsar_file,label_file = self.get_metadata(name)
            self.files.append((imgrgb1_file,imgrgb2_file,imgsar_file,label_file, name))

    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image[np.isnan(image)] = 0
        image = (image-np.mean(image))/np.std(image)
        return image.transpose((2, 0, 1)) # NHWC to NCHW

    def preprocess_rgb(self, image):
        image[np.isnan(image)] = 0
        image = image/255
        return image


    def preprocess_s1s2(self, image1):
        image1[np.isnan(image1)] = 0
        image1 = image1/10*(-1)
        return image1
    def preprocess_opt(self, image):
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        image[np.isnan(image)] = 0
        image = (image-np.mean(image))/np.std(image)
        # print(image.shape)
        return image # NHWC to NCHW

    def preprocess_label(self, label):
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = (image-np.mean(image))/np.std(image)
        # print(label.shape)

        return label # NHWC to NCHW

    def get_imagergb1(self, file):
        return _load_imgrgb1(file, self.image_size, Image.BICUBIC, rgb=True)
    def get_imagergb2(self, file):
        return _load_imgrgb2(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_imageopt(self, file):
        return _load_imgopt(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_imagesar(self, file):
        return _load_imgsar(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_lab(file, self.labels_size, Image.NEAREST, rgb=False)


def _load_imgrgb1(file, size, interpolation, rgb):
    # img = spectral.open_image(file).load()
    filepath = str(file)
    #  GDAL read
    imgds = gdal.Open(filepath + '.tif')
    img = imgds.ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32)
    # print(img.shape)
    return img

def _load_imgrgb2(file, size, interpolation, rgb):
    # img = spectral.open_image(file).load()
    filepath = str(file)
    #  GDAL read
    imgds = gdal.Open(filepath + '.tif')
    img = imgds.ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32)
    # print(img.shape)
    return img
def _load_imgopt(file, size, interpolation, rgb):
    # img = spectral.open_image(file).load()
    filepath = str(file)
    #  GDAL read
    imgds = gdal.Open(filepath + '.tif')
    img = imgds.ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32)
    # print(img.shape)
    return img
def _load_imgsar(file, size, interpolation, rgb):
    # img = spectral.open_image(file).load()
    filepath = str(file)
    #  GDAL read
    imgds = gdal.Open(filepath + '.tif')
    in_band = imgds.RasterCount
    img = imgds.ReadAsArray(buf_xsize=64,buf_ysize=64).astype(np.float32)
    #img = np.transpose(img,(1,2,0))
    #img = cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)######change the size of sar, make it the same with RGB
    #img = np.transpose(img, (2, 0, 1))
    return img

def _load_lab(file, size, interpolation, rgb):
    # img = spectral.open_image(file).load()
    filepath = str(file)
    imgds = gdal.Open(filepath + '.tif')
    img = imgds.ReadAsArray(buf_xsize=256,buf_ysize=256).astype(np.float32)
    # img = Image.open((filepath+'.png'))
    # if rgb:
    #     img = img.convert('RGB')
    # img = cv2.imread(filepath,cv2.IMREAD_COLOR)
    # print(f"the shape of the spectral_load_img is {img.shape}")
    # img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    # print(img.shape)
    return img
