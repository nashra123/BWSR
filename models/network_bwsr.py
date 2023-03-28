import torch
import torch.nn as nn
import torch.nn.functional as F


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


class KernelEstimateBlock(nn.Module):

    def __init__(self, epsilon=1e-3):
        super(KernelEstimateBlock, self).__init__()

        self.epsilon = epsilon

    def forward(self, lr_image, hr_image):
        """
        lr_image: y     [m, n]
        hr_image: P hat [m, n]
        """

        lr_image_spectral   = torch.fft.fft2(lr_image)
        hr_image_spectral   = torch.fft.fft2(hr_image)

        hr_image_spectral_abs   = torch.abs(hr_image_spectral)
        hr_image_spectral_conj  = torch.conj(hr_image_spectral)

        nominator   = hr_image_spectral_conj * lr_image_spectral
        denominator = hr_image_spectral_abs ** 2 + self.epsilon

        return torch.real(torch.fft.ifft2(nominator / denominator))


class KernelEstimateNet(nn.Module):

    def __init__(self):
        super(KernelEstimateNet, self).__init__()
        
        self.conv1              = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2              = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.resblock1          = BasicResidualBlock(64)
        self.resblock2          = BasicResidualBlock(64)
        self.kernel_est_block   = KernelEstimateBlock()

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

        hr_image_est = mod_image + lr_image

        # estimate h tilde [m, n]
        kernel_est = self.kernel_est_block(lr_image, hr_image_est)

        return hr_image_est, kernel_est

