import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_swinir import SwinIR
from .network_bwsr import KernelEstimateNet


class BWSR_SwinIR(nn.Module):

    def __init__(self, opt):
        super(BWSR_SwinIR, self).__init__()
        
        self.bwsr_net = KernelEstimateNet()

        self.f1_net = SwinIR(**opt["swinir"]["f1"])
        self.f2_net = SwinIR(**opt["swinir"]["f2"])
        self.f3_net = SwinIR(**opt["swinir"]["f3"])
        self.f4_net = SwinIR(**opt["swinir"]["f4"])


    def forward(self, lr_image):
        """forward function
            lr_image:               $y[m, n]$   \\
            sharp_image:            $\hat{p}[m, n]$ \\
            sharp_hr_image:         $f_1(\hat{p}[m, n])$    \\
            sharp_wiener_image:     $\hat{\hat{p}}[m, n]$   \\
            sharp_wiener_hr_image:  $f_2(\hat{\hat{p}}[m, n])$  \\
            hr_image:               $f_3(y[m, n])$  \\
            full_hr_image:          $\hat{x}[scale * m, scale * n]$ \\

        Args:
            lr_image (torch.Tensor): low resolution image
        """

        # generate \hat{p} and \hat{\hat{p}} using Blind Wiener SR module
        sharp_image, sharp_wiener_image, _ = self.bwsr_net(lr_image)

        # using f_1 network to upscale \hat{p} image
        sharp_hr_image, sharp_feature = self.f1_net(sharp_image)

        # using f_2 network to upscale \hat{\hat{p}} image
        sharp_wiener_hr_image, sharp_wiener_feature = self.f2_net(sharp_wiener_image)

        # using f_3 network to upscale y image directly
        direct_hr_image, direct_feature = self.f3_net(lr_image)

        # concat upscaled images over channel dimension
        fused_image = torch.cat([sharp_feature, sharp_wiener_feature, direct_feature], dim=1)

        # using f_4 network to fuse features of all three upscaled images
        full_hr_image, _ = self.f4_net(fused_image)

        return (sharp_hr_image,
                sharp_wiener_hr_image,
                direct_hr_image,
                full_hr_image)
