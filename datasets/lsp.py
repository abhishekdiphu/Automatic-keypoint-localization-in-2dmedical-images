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
# from PIL import Image


parser = argparse.ArgumentParser()

parser.add_argument('--path', \
    default='/home/aburagohain/scripts/Adversarial-Pose-Estimation/lspet_dataset')
parser.add_argument('--mode', default='train')
parser.add_argument('--crop_size', default=256)
parser.add_argument('--train_split', type=float, default=.001)
parser.add_argument('--heatmap_sigma', type=float, default=3)
parser.add_argument('--occlusion_sigma', type=float, default=2)

class LSP(Dataset):
    '''
    LSP dataset
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
        assert self.mode in ['train', 'val'], 'invalid mode {}'.format(self.mode)
        assert cfg.train_split > 0 and cfg.train_split < 1, 'train_split should be a fraction'
        self._get_files()

    def _get_files(self):
        # Get files for train/val
        self.files = sorted(glob(join(self.path, 'images/*.jpg')))
        #print(self.files)
        self.annot = loadmat(join(self.path, 'joints.mat'))['joints']

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

        # Get the i'th entry
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
        

        # Read annotations
        annot = self.annot[:, :, idx] + 0.0
        # annot = K * 3
        annot[:, :2] = annot[:, :2] * np.array(\
            [[self.out_size*1.0/image.shape[1], self.out_size*1.0/image.shape[0]]])
        
        
        #crop_image_b = cv2.resize(image, (self.crop_size_b, self.crop_size_b))
        

        # Read annotations
        annot_b = self.annot[:, :, idx] + 0.0
        # annot = K * 3
        annot_b[:, :2] = annot_b[:, :2] * np.array(\
            [[self.out_size_b*1.0/image.shape[1], self.out_size_b*1.0/image.shape[0]]])


        # Generate 64 heatmaps
        x = range(self.out_size)  # x = range(self.crop_size) --- new
        xx, yy = np.meshgrid(x, x)
        
        
        # Generate  256 heatmaps
        m = range(self.out_size_b)  # x = range(self.crop_size) --- new
        mm, nn = np.meshgrid(m, m)
        
        #new heatmaps = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        heatmaps = np.zeros((annot.shape[0], self.out_size, self.out_size))
        #new occlusions = np.zeros((annot.shape[0], self.crop_size, self.crop_size))
        occlusions = np.zeros((annot_b.shape[0], self.out_size_b, self.out_size_b)) 
        #print(annot_b.shape[0])
        # Annotate heatmap
        for joint_id in range(annot.shape[0]):
            x_c, y_c, vis = annot[joint_id] + 0
            m_c, n_c, vis = annot_b[joint_id] + 0
            #print("vis :",vis,x_c, y_c)
            heatmaps[joint_id] = np.exp(-1*((x_c - xx)**2 + (y_c - yy)**2)/(self.heatmap_sigma**2))
            #occlusions[joint_id] =np.exp(-0.5*((x_c - xx)**2 + (y_c - yy)**2)/(self.occlusion_sigma**2))
            occlusions[joint_id] =np.exp(-0.5*((m_c - mm)**2 + (n_c - nn)**2)/(self.occlusion_sigma**2))
            #print("occluded part",occlusions[joint_id])
            # occlusions[joint_id] = (1 - vis)*np.exp(-0.5*((x_c - xx)**2 + (y_c - yy)**2)/(self.occlusion_sigma**2))

            
 #       for joint_id_b in range(annot_b.shape[0]):
 #           m_c, n_c, vis = annot_b[joint_id_b] + 0
 #          
 #           occlusions[joint_id] =np.exp(-0.5*((m_c - mm)**2 + (n_c - nn)**2)/(self.occlusion_sigma**2))
    
        return {
            # image is in CHW format
            'image': torch.Tensor(crop_image.transpose(2, 0, 1)),#,/255.,
            'kp_2d': torch.Tensor(annot),
            'heatmaps': torch.Tensor(heatmaps*1.5),
            'occlusions': torch.Tensor(occlusions),
            # TODO: Return heatmaps
        }

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = LSP(args)
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

