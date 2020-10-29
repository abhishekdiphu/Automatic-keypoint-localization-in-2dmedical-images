'''
LSP Dataset
'''
from os.path import join
import argparse

from glob import glob
import cv2
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
import json as json
# from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='handmedical/')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=.4067)
parser.add_argument('--heatmap_sigma', type=float, default=1)
parser.add_argument('--occlusion_sigma', type=float, default=3)

class HANDXRAY(Dataset):
    '''
    X-Ray dataset
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = 'handmedical'# cfg.path
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self.train_split = cfg.train_split
        self.heatmap_sigma = cfg.heatmap_sigma
        self.occlusion_sigma = cfg.occlusion_sigma
        self.out_size   = 256
        self.out_size_b = 256
        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        
        #print(self.files)
        self.annot = sorted(glob(join(self.path, 'images/*.meta')))
        print("length of images",len(self.annot))
        self.files = sorted(glob(join(self.path, 'images/*.png')))
        print("length of annotation", len(self.files))

    def __len__(self):
        # Return length
        if self.mode == 'train':
            return int(self.train_split * len(self.files))
        else:
            return len(self.files) - int(self.train_split * len(self.files))


    def __getitem__(self, idx):
        # if validation, offset index
        #idx =3
        if self.mode == 'val':
            idx += int(self.train_split * len(self.files))

       
        
        file_name = self.files[idx]
        
        # if (self.mode == 'val'):
            # print('reading ', file_name)
        # image = Image.open(file_name)
        # b, g, r = image.split()
        # image = image.merge('RGB', (r, g, b))
        # image = image.resize((self.crop_size, self.crop_size))
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        
        image = (image)/255;                                    #(image - 128.0)/ 128;
        
        crop_image = cv2.resize(image, (self.crop_size, self.crop_size))
        
        annot = self.annot[idx]
        #print(annot)
        # Read annotations
        #for l in range(0,12):
        with open(annot, 'r') as json_file:
            data = json.load(json_file)
        #print("::::::::::::::::::",data)


        annotations  =[]

        #print(str(idx[1][0]))
        #for i in range(0, 12):
        for p, j in data['objects'][0]["parts"].items():#[idx[i][0]]['position']:
                    #print("value of p:", p)
                    #print("value of j:", j)
                    #annotations[i].append(p[1]['position']) 
                    annotations.append(j['position'])
        #print("###############",annotations)     
        
        
        x = range(64)  
        xx, yy = np.meshgrid(x, x)
        #print(len(annotations))       
        heatmaps = np.zeros((12, 64, 64)) #len(annotations)
        
        
        for joint_id in range(len(annotations)):
            x_cc, y_cc= annotations[joint_id]
            x_c = x_cc * (64*1/image.shape[1])
            y_c =   y_cc * (64*1/image.shape[0])
            
            #print("vis :",x_c, y_c)
            
            heatmaps[joint_id] = np.exp(-5*((x_c - xx)**2 + (y_c - yy)**2)/(2**2))
    
        return {
            # image is in CHW format
            'image': torch.Tensor(crop_image.transpose(2, 0, 1)),#,/255.,
            'kp_2d': torch.Tensor(annotations),
            'heatmaps': torch.Tensor(heatmaps*1.8),
            'occlusions': torch.Tensor(heatmaps*1.7),
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = HANDXRAY(args)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.clf()
        print(data['image'].min())
        print(data['image'].max())
        plt.subplot(1, 3, 1)
        #plt.imshow(data['image'].numpy().transpose(1, 2, 0)/255.0)  ## originally it was  there ,noramlized images are produced
        plt.imshow((data['image'].numpy().transpose(1, 2, 0)*255).astype(np.uint8))
        #plt.scatter(data['kp_2d'][:, 0].numpy(), data['kp_2d'][:, 1].numpy(), c=data['kp_2d'][:, 1])

        plt.subplot(1, 3, 2)
        plt.imshow(data['heatmaps'].numpy().sum(0))
        print(data['heatmaps'].numpy().sum(0).min())
        print(data['heatmaps'].numpy().sum(0).max())

        plt.subplot(1, 3, 3)
        plt.imshow(data['occlusions'].numpy().sum(0))

        plt.show()

