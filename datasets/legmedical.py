'''
Lower Leg x-ray Dataset
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
import SimpleITK as sitk
import datasets.img as I


from random import randint
from skimage.util import random_noise
#from skimage.filters import median
import time




parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/aburagohain/long_leg_ruppertshofen/')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=.85)
parser.add_argument('--heatmap_sigma', type=float, default=3)
parser.add_argument('--occlusion_sigma', type=float, default=2)

class HANDXRAY(Dataset):
    '''
    X-Ray dataset of lower leg datasets
    '''
    def __init__(self, cfg):
        # Path = dataset path, mode = train/val
        self.path = cfg.path
        self.mode = cfg.mode
        self.crop_size = cfg.crop_size
        self.train_split = cfg.train_split
        self.heatmap_sigma = cfg.heatmap_sigma
        self.occlusion_sigma = cfg.occlusion_sigma
        self.out_size   = 256
        self.out_size_b = 256
        self.outputRes= 64
        self.inputRes = 256
        self.rotate = 0
        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        
        #print(self.files)
        #self.annot = sorted(glob(join(self.path, 'images/*.meta')))
        self.annot = hiwi.ImageList.load('/home/aburagohain/training_test_list/fold5_training.iml') #fold5_training
        #print("length of images",self.annot[821].data)
        print("length of images",len(self.annot))
        self.files = hiwi.ImageList.load('/home/aburagohain/training_test_list/fold5_training.iml') #fold5_training
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

       
       
        #start = time.process_time()
        
        
        file_name = self.files[idx].data
        c = np.array([file_name.shape[1]/2 , file_name.shape[0]/2])
        s = file_name.shape[0]
        
        
        #print("scale of  the image" , file_name.shape[0])
        r = 0
        
        
        #if self.mode == 'train':
        #    r = 0 if np.random.random() < 0.6 else I.Rnd(self.rotate)
        #file_name =sitk.GetImageFromArray(file_name)
        #image = cv2.imread(file_name)
        image = cv2.cvtColor(file_name, cv2.COLOR_BGR2RGB)/257
       
        
        #print(image.shape)
        #original = image.copy()
        #original = I.Crop(original, c, s, r, 64)/255                   #cv2.resize(original, (256, 256))/255
        inp = I.Crop(image, c, s, r, self.inputRes)/255
        
        
        
       
        
        #print(time.process_time() - start)
        
        
        
        
        
        
        
        
        #print(inp)
        #out = np.zeros((self.nJoints, self.outputRes, self.outputRes))
        
    
        
        #file_name = sitk.GetImageFromArray(file_name)
  
        #print(file_name.GetSize())
        #file_name.SetSpacing([4.1 , 4.1])
        #file_name =sitk.GetArrayFromImage(file_name) 
        #print(file_name.GetSpacing())
     
        #image = cv2.cvtColor(file_name, cv2.COLOR_BGR2RGB)
        #print(image.max())
        #image_n = (image/257.0).astype('uint8')#(image - 128.0)/ 128;
        #image   =(image/257).astype('uint8') 
        
        #crop_image = cv2.resize(image, (self.crop_size, self.crop_size))
        #print("image shape:" , crop_image.shape)
        
        annot = self.files[idx]
        #print(annot)
        # Read annotations
        #for l in range(0,12):
        #with open(annot, 'r') as json_file:
        #    data = json.load(json_file)
        #print("::::::::::::::::::",data)


        annotations  ={'ankle_left':[0.0, 0.0] , 'ankle_right':[0.0, 0.0] ,'femur_left':[0.0, 0.0] ,
                        'femur_right':[0.0, 0.0] , 'knee_left':[0.0, 0.0]  , 'knee_right':[0.0, 0.0]}
           
        #print(str(idx[1][0]))
        #for i in range(0, 12):
        for name, obj in annot.parts.items():#[idx[i][0]]['position']:
            #for key in annotations:
                    #print(name)
                    #print("value of p:", p)
                    #print("value of j:", j)
                    #annotations[i].append(p[1]['position'])
                    
                    if name == 'ankle_left':
                        annotations['ankle_left'] = (obj['position'])
                    elif name == 'ankle_right':
                        annotations['ankle_right'] = (obj['position'])
                    elif name == 'femur_left':
                        annotations['femur_left']  = (obj['position'])
                    elif name == 'femur_right':
                        annotations['femur_right']  = (obj['position'])
                    elif name == 'knee_left':
                        annotations['knee_left']  = (obj['position'])
                    elif name == 'knee_right':
                        annotations['knee_right']  = (obj['position'])
                    
        
        
                        
        #print("Annotations file :",annotations)
        annotations_l =[]
        for key, value in annotations.items():
               
                annotations_l.append(value)
        
        pts = annotations_l
        #print(annotations_l)
        out = np.zeros((6, self.outputRes, self.outputRes))
        
        
        
        
        #for i in range(6):
        #    if annotations_l[i][0]>1:
        #        pts[i] = I.Transform(annotations_l[i], c, s, r, self.outputRes)
        #        out[i] = I.DrawGaussian(out[i], annotations_l[i], 2, 0.5 if self.outputRes==32 else -1)
        #print("annotations :" , pts)
        #-----------------------------------------------------------------------------------------------------------------
        #x = range(256) 
        m = range(64)
        #xx, yy = np.meshgrid(x, x)
        mm , nn = np.meshgrid(m, m)
        #print(len(annotations))       
        #heatmaps = np.zeros((6, 256, 256)) #len(annotations)
        #occlusion = np.zeros((6, 64,64))
        
        for joint_id in range(len(annotations_l)):
            if pts[joint_id][0] > 0 and pts[joint_id][1] > 0:
                x_cc, y_cc= annotations_l[joint_id]
                
                #------------------------------------------------------------------------
                #x_c = x_cc * (256 / image.shape[1])
                #y_c =   y_cc * (256 / image.shape[0])
                #print("vis :",x_c, y_c)
                #------------------------------------------------------------------------
                
                pts[joint_id] = I.Transform(np.array([x_cc, y_cc]), c, s, r, self.outputRes)

                #print(pts[joint_id])
                #m_c , n_c = pts
                out[joint_id] =I.DrawGaussian(out[joint_id], pts[joint_id], 1, 0.5 if self.outputRes==32 else -1) 
                
                #------------------------------------------------------------------------
                #np.exp(-5*((m_c - mm)**2 + (n_c - nn)**2)/(2**2)) 
                #I.DrawGaussian(out[i], pts[i], 1, 0.5 if self.outputRes==32 else -1)
                #------------------------------------------------------------------------
        
        
        #if self.mode == 'train':
        #    if np.random.random() < 0.5:
        #            inp = I.Flip(inp)
        #            out = I.ShuffleLR_LEX(I.Flip(out))        
        
        #if self.mode == 'val':        
        inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
        inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
        inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
        
        
        #----Adding random noise to the images --------#
        #inp = random_noise(inp , 'salt' , amount = 0.001)
        #inp = random_noise(inp , 'speckle', mean = 0 , var =0.007)  #localvar
            
        #-----------------------------------------------------------------------------------------------------------------
        #    heatmaps[joint_id] = np.exp(-5*((x_c - xx)**2 + (y_c - yy)**2)/(self.heatmap_sigma**2))
            
        #    m_cc, n_cc= annotations_l[joint_id]
        #    m_c = m_cc * (64*1/image.shape[1])
        #    n_c =   n_cc * (64*1/image.shape[0])
        #    occlusion[joint_id] = np.exp(-5*((m_c - mm)**2 + (n_c - nn)**2)/(self.occlusion_sigma**2))
        #0-----------------------------------------------------------------------------------------------------------------
        
        return {
            # image is in CHW format
            'image': torch.Tensor(inp),#,/255.,
            #'kp_2d': torch.Tensor(annotations),
            'heatmaps': torch.Tensor(out*1.2), #1.2
            'occlusions': torch.Tensor(out*1.2),# 1.2
  #          'small_image'  : torch.Tensor(original) # .transpose(2 , 0 , 1)
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = HANDXRAY(args)
    print(len(dataset))
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.figure(figsize=(30 ,30))
        plt.clf()
        print(data['image'].min())
        print(data['image'].max())
        #print(data['image'].mean())
        #print(data['image'].std())
        plt.subplot(1, 4, 1)
        plt.imshow(data['image'].numpy().transpose(1, 2, 0)/255.0)  ## originally it was  there ,noramlized images are produced
        plt.imshow((data['image'].numpy().transpose(1, 2, 0)*255).astype(np.uint8) , cmap = plt.cm.gray)
        #plt.subplot(1, 4, 2)
        #plt.imshow((data['small_image'].numpy().transpose(1, 2, 0)*255).astype(np.uint8) , cmap = plt.cm.gray)
        #plt.scatter(data['kp_2d'][:, 0].numpy(), data['kp_2d'][:, 1].numpy(), c=data['kp_2d'][:, 1])

        plt.subplot(1, 4, 3)
        plt.imshow(data['heatmaps'].numpy().sum(0))
        print(data['heatmaps'].numpy().sum(0).min())
        print(data['heatmaps'].numpy().sum(0).max())

        plt.subplot(1, 4, 4)
        plt.imshow(data['occlusions'].numpy().sum(0))

        plt.show()

