### is not used at all ####



import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1
        
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError("Index {} is out of range".format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)
    
    def __iter__(self):
        return iter(self._modules.values())
    
    def __len__(self):
        return len(self._modules)

class ConvBnRelu(nn.Module):
    """
        A block of convolution, relu, batchnorm
    """	

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0 ):

        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # changing this to instance norm due to small batch_size of 1
        self.bn   = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DeconvBnRelu(nn.Module):
    """
        A block of de-convolution, relu, batchnorm
    """	

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):

        super(DeconvBnRelu, self).__init__()

        
        
        self.conv = nn.ConvTranspose2d(in_channels=in_channels , out_channels= out_channels, kernel_size= kernel_size, stride=stride, padding=padding )
        self.bn= nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
    

class DeconvRelu(nn.Module):
    """
        A block of de-convolution, relu, batchnorm
    """	

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):

        super(DeconvRelu, self).__init__()

        
        
        self.conv = nn.ConvTranspose2d(in_channels=in_channels , out_channels= out_channels, kernel_size= kernel_size, stride=stride, padding=padding )
        self.relu = nn.ReLU()
        
        

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)

        return x


class ConvTripleBlock(nn.Module):
    """
        A block of 3 ConvBnRelu blocks. 
        This triple block makes up a residual block as described in the paper
        Resolution h x w does not change across this block
    """	

    def __init__(self, in_channels, out_channels):

        super(ConvTripleBlock, self).__init__()

        out_channels_half = out_channels // 2

        self.convblock1 = ConvBnRelu(in_channels, out_channels_half)
        self.convblock2 = ConvBnRelu(out_channels_half, out_channels_half, kernel_size=3, stride=1, padding=1)
        self.convblock3 = ConvBnRelu(out_channels_half, out_channels)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        return x

class SkipLayer(nn.Module):
    """
        The skip connections are necessary for transferring global and local context
        Resolution h x w does not change across this block
    """

    def __init__(self, in_channels, out_channels):

        super(SkipLayer, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        
        if self.in_channels != self.out_channels:
            x = self.conv(x)

        return x

class Residual(nn.Module):
    """
        The highly used Residual block
        Resolution h x w does not change across this block
    """
    def __init__(self, in_channels, out_channels):

        super(Residual, self).__init__()

        self.convblock = ConvTripleBlock(in_channels, out_channels)
        self.skip    = SkipLayer(in_channels, out_channels)


    def forward(self, x):

        y = self.convblock(x)
        z = self.skip(x)
        out = y + z

        return out


class Hourglass(nn.Module):
    """
        Hourglass network - Core component of Generator
    """

    def __init__(self, num_channels, num_reductions=4, num_residual_modules=2):

        super(Hourglass, self).__init__()

        scale_factor = 2
        self.num_reductions = num_reductions

        skip = []
        for _ in range(num_residual_modules):
            skip.append(Residual(num_channels, num_channels))
        self.skip = nn.Sequential(*skip)

        self.pool = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        
        before_mid = []
        for _ in range(num_residual_modules):
            before_mid.append(Residual(num_channels, num_channels))
        self.before_mid = nn.Sequential(*before_mid)
        
        if (num_reductions > 1):
            self.sub_hourglass = Hourglass(num_channels, num_reductions - 1, 
                                            num_residual_modules)
        else:
            mid_residual = []
            for _ in range(num_residual_modules):
                mid_residual.append(Residual(num_channels, num_channels))
            self.mid_residual = nn.Sequential(*mid_residual)

        end_residual = []
        for _ in range(num_residual_modules):
            end_residual.append(Residual(num_channels, num_channels))
        self.end_residual = nn.Sequential(*end_residual)

        self.up_sample = nn.Upsample(scale_factor=scale_factor, mode='nearest')


    def forward(self, x):
        y = self.pool(x)
        y = self.before_mid(y)
        
        if (self.num_reductions > 1):
            y = self.sub_hourglass(y)
        else:
            y = self.mid_residual(y)

        y = self.end_residual(y)
        y = self.up_sample(y)
        
        x = self.skip(x)

        out = x + y
        
        return out

class StackedHourglass(nn.Module):
    """
        Stacking hourglass - gives precursors to pose and occlusion heatmaps  
    """

    def __init__(self, num_channels, hourglass_params):
        super(StackedHourglass, self).__init__()
        
        self.hg = []
        for _ in range(2):
            self.hg.append(Hourglass(num_channels, hourglass_params['num_reductions'], hourglass_params['num_residual_modules']))
        self.hg = ListModule(*self.hg)

        self.dim_reduction = nn.Conv2d(in_channels=2 * num_channels, out_channels=num_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        y = x
        out1 = self.hg[0](y)
        y = torch.cat((out1, y), dim=1)
        y = self.dim_reduction(y)
        out2 = self.hg[1](y)
        return [out1, out2]