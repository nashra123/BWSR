import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils import utils_deblur as deblur
from utils import utils_image as util


class BasicResidualBlock(nn.Module): 
    """Basic Residual Block
    
        x ---> [ Conv2D -> BN -> ReLU ] -> [ Conv2D -> BN ] --(+)-->
            \_________________________________________________/
    """

    def __init__(self, channel_num):
        super(BasicResidualBlock, self).__init__()
        ## 3 * 3 kernel conv 
        #input and output channels == channel_num
        self.conv_block1 = nn.Sequential(nn.Conv2d(channel_num, channel_num ,3, padding=1),
                                         nn.BatchNorm2d(channel_num),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(channel_num, channel_num, 3, padding=1),
                                         nn.BatchNorm2d(channel_num))
              
    def forward(self,x): 
          
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual 
        return x


class KernelEstimateNet(nn.Module):

    def __init__(self):
        super(KernelEstimateNet, self).__init__()
        
        self.conv1              = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2              = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.resblock1          = BasicResidualBlock(64)
        self.resblock2          = BasicResidualBlock(64)

    def forward(self, lr_image):
        """
        lr_image    : y       [m, n]
        hr_image_est: P hat   [m, n]
        kernel_est  : h tilde [m, n]
        """

        # estimate P hat [m, n]
        mod_image = self.conv1(lr_image)
        mod_image = F.relu(mod_image)
        mod_image = self.resblock1(mod_image)
        mod_image = self.resblock2(mod_image)
        mod_image = self.conv2(mod_image)
        mod_image = F.relu(mod_image)

        sharp_image_est = mod_image + lr_image

        # estimate h tilde [m, n]
        kernel_est = deblur.kernel_estimate(sharp_image_est, lr_image)

        with torch.no_grad():
            hr_image_est = deblur.wiener_deconv(lr_image, kernel_est)

        return sharp_image_est, hr_image_est, kernel_est

