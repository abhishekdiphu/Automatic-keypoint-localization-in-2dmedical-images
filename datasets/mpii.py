import cv2
import torch
import h5py as  H
import numpy as np
import scipy.io as sio

import datasets.img as I ## originally it was import img as I
#img earlier
from matplotlib import pyplot as plt
import torch.utils.data as data
import torchvision.transforms.functional as F

class MPII(data.Dataset):
	def __init__(self, split):
		print('==> initializing 2D {} data.'.format(split))
		self.split = split
		self.maxScale = 1
		self.inputRes = 256
		self.outputRes = 256
		self.outputRes_s = 64
		self.nJoints = 16
		self.hmGauss = 2
		self.hmGauss_s = 1
		tags = ['imgname','part','center','scale']

		# self.stuff1 = sio.loadmat(open(opts.worldCoors[:-4] + ('train' if split is 'train' else '') + '.mat', 'rb'))['a']
		# self.stuff2 = sio.loadmat(open(opts.headSize[:-4] + ('train' if split is 'train' else '') + '.mat', 'rb'))['headSize']

		#f = H.File('/home/aburagohain/scripts/Adversarial-Pose-Estimation/mpii/pureannot/{}.h5'.format(split), 'r') 
		f = H.File('/home/aburagohain/scripts/Adversarial-Pose-Estimation/mpii/pureannot/{}.h5'.format(split), 'r') 

		annot = {}
		for tag in tags:
			annot[tag] = np.asarray(f[tag]).copy()
			#print(annot[tag]) ##originally not there
		f.close()
		self.annot = annot
		#print(self.annot)

		self.len = len(self.annot['scale'])
		print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))

	def LoadImage(self, index):
		#print(self.annot['imgname'][index]) ##originally not there
		#path = '/home/rohit/Adversarial-Pose-Estimation/mpii/images/{}'.format(''.join(chr(int(i)) for i in self.annot['imgname'][index]))
		#path = '/home/aburagohain/scripts/Adversarial-Pose-Estimation/datasets/mpii/images/{}'.format(self.annot['imgname'][index])#origianlly was there , now replacing it by below line.
        
        
        
		path = '/home/aburagohain/scripts/Adversarial-Pose-Estimation/mpii/images/{}'.format(''.join(chr(int(i)) for i in self.annot['imgname'][index]))#originally not there .it a replacement for above line       
		img = cv2.imread(path)
		#print(img) ### originally not here . to make sure it is not the none type       
		return img

	def GetPartInfo(self, index):
		pts = self.annot['part'][index].copy()
		c = self.annot['center'][index].copy()
		s = self.annot['scale'][index]
		#print("the value of s :" , s)
		#print("the value of pts :" , pts) 
		#print("the value of c :" , c) 
		s = s * 200
		return pts, c, s

	def __getitem__(self, index):
		img = self.LoadImage(index)
		pts, c, s = self.GetPartInfo(index)
        
        
		pts_s, _, _ = self.GetPartInfo(index)
        
        
		r = 0

		if self.split == 'train':
			# s = s * (2 ** I.Rnd(self.maxScale))
			s = s * 1
			r = 0
		inp = I.Crop(img, c, s, r, self.inputRes) / 255.            
		out = np.zeros((self.nJoints, self.outputRes, self.outputRes))
        
# for 256 x 256 heatmaps
#------------------------------------------------------------------------------
		out_s = np.zeros((self.nJoints, self.outputRes_s, self.outputRes_s))
#------------------------------------------------------------------------------    
		for i in range(self.nJoints):
			if pts[i][0] > 1:
				pts[i] = I.Transform(pts[i], c, s, r, self.outputRes)
				out[i] = I.DrawGaussian(out[i], pts[i], self.hmGauss, 0.5 if self.outputRes==32 else -1)

# for 256 x 256 heatmaps
#-----------------------------------------------------------------------------------------------------------
		for i in range(self.nJoints):
			if pts_s[i][0] > 1:
				pts_s[i] = I.Transform(pts_s[i], c, s, r, self.outputRes_s)
				out_s[i] = I.DrawGaussian(out_s[i], pts_s[i], self.hmGauss_s, 0.5 if self.outputRes==32 else -1)
#-----------------------------------------------------------------------------------------------------------

		if self.split == 'train':
			if np.random.random() < 0.5:
				inp = I.Flip(inp)
				out = I.ShuffleLR(I.Flip(out))
				pts[:, 0] = self.outputRes - pts[:, 0]
                
# for 256 x 256 heatmaps                
#--------------------------------------------------------------------------------------------------------------                
				out_s = I.ShuffleLR(I.Flip(out_s))
				pts_s[:, 0] = self.outputRes_s - pts_s[:, 0]
#--------------------------------------------------------------------------------------------------------------
        
			inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)

			#out_s = np.resize(out_s , (256, 256,3))
			#print(out_s.shape())
		return {
			'image': torch.Tensor(inp),
			'heatmaps': torch.Tensor(out*1.7),
			'occlusions': torch.Tensor(out_s*1.7),
		}
		# return inp, out
		# if self.opts.TargetType=='heatmap':
		# 	return inp, out#, self.stuff1[index], self.stuff2[index]
		# elif self.opts.TargetType=='direct':
		# 	return inp, np.reshape((pts/self.opts.outputRes), -1) - 0.5#, self.stuff1[index], self.stuff2[index]

	def __len__(self):
		return self.len


if __name__ == '__main__':
	dataset = MPII('train')
	for i in range(len(dataset)):
		ii = np.random.randint(len(dataset))
		data = dataset[ii]
		print(data['image'].min())
		print(data['image'].max())        
		plt.subplot(1, 3, 1)
		#plt.imshow(data['image'].transpose(1, 2, 0)[:, :, ::-1] + 0) ## orginal line replaced by below line
		plt.imshow(data['image'].numpy().transpose(1, 2, 0)[:, :, ::-1] + 0)
		plt.subplot(1, 3, 2)
		print(data['heatmaps'].shape)
		print(data['heatmaps'].min())
		print(data['heatmaps'].max()) 
		plt.imshow(data['heatmaps'].numpy().max(0)) ## originally numpy() was  not there
		plt.subplot(1, 3, 3)
		plt.imshow(data['occlusions'].numpy().max(0)) ## originally numpy() was  not there
		plt.show()