###---------------------------------------------------






import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBnRelu, DeconvBnRelu
from modules import ListModule






#-----------------------------------------------------discrimimator 2-------------------------------------------------------------#
class Discriminator1(nn.Module):

	def __init__(self, in_channels, num_channels, num_joints, num_residuals=5):
		'''
			Initialisation of the Discriminator network
			Contains the necessary modules
			Input is pose and confidence heatmaps
			in_channels = num_joints x 2 + 3 (for the image) (Pose network)
			in_channels = num_joints x 2 (Confidence network)
		'''
		
		super(Discriminator1, self).__init__()
		## Define Layers Here ##
        
        #---------------encoder---------------#
		self.ConvBnRelu1   = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1)
		self.relu1         = nn.ReLU(num_channels)
		self.ConvBnRelu2   = ConvBnRelu(in_channels=64, out_channels=128, kernel_size=4, stride=2,  padding=1)
		self.ConvBnRelu3   = ConvBnRelu(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        
        
        
        

        #-------------1x1 convolution --------# 
		self.ConvBnRelu4   = ConvBnRelu(in_channels=256, out_channels= 512, kernel_size=1, stride=1, padding=0)
		self.ConvBnRelu5   = ConvBnRelu(in_channels=512, out_channels= 512, kernel_size=1, stride=1, padding=0)
        
        
        
        
        #---------------decoder---------------#
		#self.skip_layer1   = torch.cat([self.ConvBnRelu3, self.ConvBnRelu5], dim=1)
		self.DeconvBnRelu6 = DeconvBnRelu(in_channels=512 + 256,
                                           out_channels=256, kernel_size=4, stride=2, padding=1)
        
		#self.skip_layer2   = torch.cat([self.ConvBnRelu2 , self.DeconvBnRelu6], dim=1)
		self.DeconvBnRelu7 = DeconvBnRelu(in_channels= 256 + 128 , 
                                           out_channels=128, kernel_size=4, stride=2, padding=1)
        
		#self.skip_layer3   = torch.cat([self.ConvBnRelu1 , self.DeconvBnRelu7], dim=1)
		self.DeconvBnRelu8 = DeconvBnRelu(in_channels=128 + 64  , 
                                           out_channels=num_joints, kernel_size=4, stride=2, padding=1)
		#self.relu8         = nn.ReLU(num_channels)
                    
		#self.max_pool = nn.MaxPool2d(kernel_size = 4, stride = 2, padding=1)
        
		#self.fc1 = nn.Linear(256*256*14,1)
		#self.bn1 = nn.BatchNorm2d(128)        
		#self.relu = nn.ReLU()        
		#self.fc2 = nn.Linear(128, num_joints)

	def forward(self, x):
		"""
			

		"""
		#print("0",x.size())
		x1 =self.ConvBnRelu1(x)
		x1 =self.relu1(x1)
		#print("1",x1.size())
		x2 =self.ConvBnRelu2(x1)
		#print("2",x2.size())
		x3 =self.ConvBnRelu3(x2)
		#print("3",x3.size())
		x4 =self.ConvBnRelu4(x3)
		#print("4",x4.size())
		x4 =self.ConvBnRelu5(x4)
		#print("5",x4.size())
        
		skip_1 =torch.cat([x3 , x4], dim=1) 
		x6 =self.DeconvBnRelu6(skip_1)
		#print("6",x6.size())

		skip_2 =torch.cat([x2 , x6], dim=1)
		x7 =self.DeconvBnRelu7(skip_2)
		#print("7",x7.size())
        
		skip_3 =torch.cat([x1 , x7], dim=1)
		x8 =self.DeconvBnRelu8(skip_3)
		#x8 =self.relu8= nn.ReLU(x8)
		#print("8",x8.size())
        
		#x9 = self.max_pool(x8)
		#print("9",x9.size())                
		# N x 512 x 16 x 16
        
		#x9 = x8.view(x8.shape[0] , -1)
		#print("10",x9.size())
		# N x (512 * 16 * 16)
		#x9 = self.fc1(x9)
		#print("11",x9.size())
		# N x 128
		#x9 = self.relu(x9)
		#print("12",x9.size())
		# N x 128
		#x9 = self.fc2(x9)
		#print("13",x9.size())
		# N x 16
		#x9 = F.sigmoid(x9)
		x10 = F.sigmoid(x8)
		#print("the shape of the output of the discriminator is : ", x8.size() )        


		return x10