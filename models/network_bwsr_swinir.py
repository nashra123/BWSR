import torch
from torch import nn

from models.network_swinir import SwinIR
from models.network_bwsr import KernelEstimateNet

class BWSR_SwinIrNet(nn.Module):
    def __init__(self, opt):
        super(BWSR_SwinIrNet, self).__init__()
        # kernel estimation
        self.bwsr_net = KernelEstimateNet()
        # Freeze BWSR parameters during training
        self.disable_requires_grad(self.bwsr_net)

        # Swin Transformer blocks
        # 
        # we should pass to the constructor SwinIR parameters
        # a best practice is to use opt dictionary and pass arguments
        # of SwinIR network through it.
        self.f1_net = SwinIR(**opt["swinir"]["f1"]) 
        self.f2_net = SwinIR(**opt["swinir"]["f2"])
        self.f3_net = SwinIR(**opt["swinir"]["f3"])
        self.f4_net = SwinIR(**opt["swinir"]["f4"])
        
    def forward(self, lr_image):
        #### network_bwsr_swinir
        ####passes low resolution image and then it returns p_hat and p_hat image
        p_hat_img, p_hathat_img, _ = self.bwsr_net(lr_image)
        
        ###SwinIR first network p_hat image returns upscaled version of p_hat and feature image of p_hat
        net_1_upscale_op, net_1_feature_op = self.f1_net(p_hat_img)
        
        ##SwinIR  with upscale p_hathat_img as the input returns upscaled version of p_hathat_img , and feature img p_hathat_img
        net_2_upscale_op, net_2_feature_op = self.f2_net(p_hathat_img)
        
        ##SwinIR  for lr_img as the input ----- upscaled and feature img of lr_img
        net_3_upscale_op, net_3_feature_op = self.f3_net(lr_image)
        
        catenated_feature = torch.cat((net_1_feature_op, net_2_feature_op, net_3_feature_op), dim=1)
        # final combined_feature_img as input to the final swinIR returning the final_upscale_img 
        final_upscale_img, _ = self.f4(catenated_feature)
        
        return (
            net_1_upscale_op,
            net_2_upscale_op,
            net_3_upscale_op,
            final_upscale_img
        )
    
    def disable_requires_grad(self, model: torch.nn.Module) -> None:
        for p in model.parameters():
            p.requires_grad = False