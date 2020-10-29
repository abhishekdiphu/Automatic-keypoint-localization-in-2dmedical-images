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
import hiwi as hiwi
# from PIL import Image
import datasets.img as I
from random import randint

parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/aburagohain/long_leg_ruppertshofen/')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=.644)
parser.add_argument('--heatmap_sigma', type=float, default=2)
parser.add_argument('--occlusion_sigma', type=float, default=2)

class HANDXRAY(Dataset):
    '''
    X-Ray dataset
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = '/home/aburagohain/long_leg_ruppertshofen/'# cfg.path
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self.train_split = cfg.train_split
        self.heatmap_sigma = cfg.heatmap_sigma
        self.occlusion_sigma = cfg.occlusion_sigma
        self.out_size   = 256
        self.out_size_b = 256
        self.rotate = 5
        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        
        #print(self.files)
        #self.annot = sorted(glob(join(self.path, 'images/*.meta')))
        self.annot = hiwi.ImageList.load('/home/aburagohain/long_leg_ruppertshofen/ruppertshofen_cleaned.iml')
        #print("length of images",self.annot[821].data)
        print("length of images",len(self.annot))
        self.files = hiwi.ImageList.load('/home/aburagohain/long_leg_ruppertshofen/ruppertshofen_cleaned.iml')
        #self.files = sorted(glob(join(self.path, 'images/*.gz')))
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

       
        
        file_name = self.files[idx].data
        #-------------------------------------------------------------------------------------------------------------
        c = np.array([128 , 128])
        s = 256 #(randint(2, 3))*150
        s1= 64
        r = 0
        random = 1** I.Rnd(1)
        print("random :" , random)
        s = s * random
        s1= s1* random  
        r = 0 if np.random.random() < 0.6 else I.Rnd(self.rotate)
        #-------------------------------------------------------------------------------------------------------------
    
        image = cv2.cvtColor(file_name, cv2.COLOR_BGR2RGB)
        
        image_n = (image/257)#(image - 128.0)/ 128;
        #image   =image_n/ 255 
        
        crop_image = cv2.resize(image_n, (self.crop_size, self.crop_size))
        #---------------------------------------------------------------------
        crop_image = I.Crop(crop_image, c, s, r, self.crop_size)/255 
        #---------------------------------------------------------------------
        
        
        print("image shape:" , crop_image.shape)
        
        annot = self.files[idx]
        #print(annot)
        # Read annotations
        #for l in range(0,12):
        #with open(annot, 'r') as json_file:
        #    data = json.load(json_file)
        #print("::::::::::::::::::",data)


        annotations  =[]

        #print(str(idx[1][0]))
        #for i in range(0, 12):
        for name, obj in annot.parts.items():#[idx[i][0]]['position']:
                    #print("value of p:", p)
                    #print("value of j:", j)
                    #annotations[i].append(p[1]['position']) 
                    annotations.append(obj['position'])
        #print("###############",annotations)     
        
        
        x = range(256) 
        m = range(64)
        xx, yy = np.meshgrid(x, x)
        mm , nn = np.meshgrid(m, m)
        #print(len(annotations))       
        heatmaps = np.zeros((6, 256, 256)) #len(annotations)
        occlusion = np.zeros((6, 64,64))
        
        for joint_id in range(len(annotations)):
            x_cc, y_cc= annotations[joint_id]
          
            x_c = x_cc * (256*1/image.shape[1])
            y_c =   y_cc * (256*1/image.shape[0])
            
            annot_pt = I.Transform(np.array([x_c , y_c]), c, s, r, 256)
            
            x_c, y_c = annot_pt
            #print("vis :",x_c, y_c)
            heatmaps[joint_id] = np.exp(-5*((x_c - xx)**2 + (y_c - yy)**2)/(self.heatmap_sigma**2))
            
            
        for joint_id in range(len(annotations)):      
            
            m_cc, n_cc= annotations[joint_id]
            
            m_c = m_cc * (64*1/image.shape[1])
            n_c = n_cc * (64*1/image.shape[0])
            #print(m_c , n_c)
            
            annot_o_pt = I.Transform(np.array([m_c , n_c]), np.array([32 , 32]), s1, r, 64)
            
            m_c, n_c = annot_o_pt
            #print(m_c , n_c)
            occlusion[joint_id] = np.exp(-5*((m_c - mm)**2 + (n_c - nn)**2)/(self.occlusion_sigma**2))
            
        if np.random.random() < 0.5:
            crop_image = I.Flip(crop_image)
            occlusion = I.ShuffleLR_LEX(I.Flip(occlusion))
    
        return {
            # image is in CHW format
            'image': torch.Tensor(crop_image),#.transpose(2, 0, 1)),#,/255.,
            'kp_2d': torch.Tensor(annotations),
            'heatmaps': torch.Tensor(occlusion*1.7),
            'occlusions': torch.Tensor(occlusion*1.7),
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
        plt.figure(figsize=(15,15))
        plt.subplot(1, 3, 1 )
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

