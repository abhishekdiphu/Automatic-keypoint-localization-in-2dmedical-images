#------------------------------------------CONFIDENCE DISCRIMINATOR ---------------------------------------------------------#



import torch
from torch import nn
from layers import Conv, Hourglass, Pool, Residual
#from task.loss import HeatmapLoss
import torch.nn.functional as F



class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

#stack-hour glass pose discriminator #

class DiscPoseNet(nn.Module):
    def __init__(self, nstack=1, inp_dim = 256, oup_dim= 6, bn=False, increase=0):
        super(DiscPoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(9, 64, 7, 2, bn=True, relu=True),     ##  in place of 15  3 was there
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.fc1 = nn.Linear(64*64*6,6)
        #self.fc2 = nn.Linear(128,12)
        
        #self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        #x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(imgs)
        combined_hm_preds = []
#        final_hm_preds = [None for range(self.nstack)]
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            flatten = preds.view(preds.shape[0] , -1)
            #print("flatten:",flatten.size())
            fc      = self.fc1(flatten)
            #fc      = self.fc2(fc)
            out     = F.sigmoid(fc)
            #print("disc ouput :",out.size())
            
            
            
            #print("pred size",preds.size())
            #combined_hm_preds.append(preds)
            #if i < self.nstack - 1:
            #    x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        
        
        return out #, torch.stack(combined_hm_preds, 1)
    
    
    
    
    
    
    
    
    
    
    
# stack hour glass confidence discriminator #  

class DiscConfNet(nn.Module):
    def __init__(self, nstack=1, inp_dim = 256, oup_dim= 6, bn=False, increase=0):
        super(DiscConfNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(6, 64, 7, 2, bn=True, relu=True),     ##  in place of 15  3 was there
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.fc1 = nn.Linear(64*64*6,6)
        #self.fc2 = nn.Linear(128,12)
        
        #self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        #x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(imgs)
        combined_hm_preds = []
#        final_hm_preds = [None for range(self.nstack)]
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            flatten = preds.view(preds.shape[0] , -1)
            #print("flatten:",flatten.size())
            fc      = self.fc1(flatten)
            #fc      = self.fc2(fc)
            out     = F.sigmoid(fc)
            #print("disc ouput :",out.size())
            
            
            
            #print("pred size",preds.size())
            #combined_hm_preds.append(preds)
            #if i < self.nstack - 1:
            #    x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        
        
        return out #, torch.stack(combined_hm_preds, 1)
    
    
    
# stack hour glass pose small discriminator #

class DiscPoseNet_SIMG(nn.Module):
    def __init__(self, nstack=1, inp_dim = 256, oup_dim= 6, bn=False, increase=0):
        super(DiscPoseNet_SIMG, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(9, 64, 3, 1, bn=True, relu=True),     ##  in place of 15  3 was there
            Residual(64, 128),
            #Pool(1, 1),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.fc1 = nn.Linear(64*64*6,6)
        #self.fc2 = nn.Linear(128,12)
        
        #self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        #x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        
        x = self.pre(imgs)
        print("x :" , x.size())
        
        combined_hm_preds = []
#        final_hm_preds = [None for range(self.nstack)]
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            flatten = preds.view(preds.shape[0] , -1)
            #print("flatten:",flatten.size())
            fc      = self.fc1(flatten)
            #fc      = self.fc2(fc)
            out     = F.sigmoid(fc)
            #print("disc ouput :",out.size())
            
            
            
            #print("pred size",preds.size())
            #combined_hm_preds.append(preds)
            #if i < self.nstack - 1:
            #    x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        
        
        return out #, torch.stack(combined_hm_preds, 1)

    