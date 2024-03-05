import argparse

#training options
parser = argparse.ArgumentParser(description='Train Change Detection Models')

# training parameters
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=8, type=int, help='batchsize')
parser.add_argument('--rgbinputch', default=3, type=int, help='rgbinputchannel')
parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')

parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use None for CPU')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')
parser.add_argument('--wDice', type=float, default=0.5, help='weight of dice loss')
parser.add_argument('--wBCE', type=float, default=0.5, help='weight of BCE loss')
parser.add_argument('--wOut1',type=float,default=0.8,help='weight of VHR output')

# path for loading data
parser.add_argument('--root_dir', default=r'E:\HR-LSCD\dataset', type=str, help='rootpath for image')
parser.add_argument('--split_dir', default=r'E:\HR-LSCD\dataset\train_val_test_txt', type=str, help='txt path of splitting dataset')


parser.add_argument('--pretrained_path', default=None, type=str, help='path for pretrain model')
# network saving and loading parameters
parser.add_argument('--model_dir', default=r'E:\HR-LSCD\result\checkpoint', type=str, help='save dir for CD model')
parser.add_argument('--best_model_path', default=None, type=str, help='best model in val')
parser.add_argument('--precision_txt_dir', default=r'E:\HR-LSCD\result\precision_txt', type=str, help='txt dir for precision evaluation ')
parser.add_argument('--predict_img_dir', default=r'E:\HR-LSCD\result\predict_image', type=str, help='predicted image dir')


