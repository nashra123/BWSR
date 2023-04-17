from models.model_plain import ModelPlain
from utils import utils_deg as deg
from collections import OrderedDict
import torch
import torch.nn as nn
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from utils.utils_image import tensor2single, single2tensor4, single42tensor4, rgb2ycbcr
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelKernelEstimate(ModelPlain):
    """This Net Outputs two things:
        1. Sharp version of LR image
        2. Kernel estimate of bluring kernel
    """
    def __init__(self, opt):
        super(ModelKernelEstimate, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.prepro = deg.SRMDPreprocessing(
            scale=self.opt["scale"], cuda=True, **self.opt["degradation"]
        )

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        # --------------------------------------------------------
        # define loss function for image similarity $\hat{p} - p$
        # --------------------------------------------------------
        G_p_lossfn_type = self.opt_train['G_p_lossfn_type']
        if G_p_lossfn_type == 'l1':
            self.G_p_lossfn = nn.L1Loss().to(self.device)
        elif G_p_lossfn_type == 'l2':
            self.G_p_lossfn = nn.MSELoss().to(self.device)
        elif G_p_lossfn_type == 'l2sum':
            self.G_p_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_p_lossfn_type == 'ssim':
            self.G_p_lossfn = SSIMLoss().to(self.device)
        elif G_p_lossfn_type == 'charbonnier':
            self.G_p_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_p_lossfn_type))
        
        # --------------------------------------------------------
        # define loss function for kernel energy $\tilde{h}$
        # --------------------------------------------------------
        G_h_lossfn_type = self.opt_train['G_h_lossfn_type']
        if G_h_lossfn_type == 'l1':
            self.G_h_lossfn = nn.L1Loss().to(self.device)
        elif G_h_lossfn_type == 'l2':
            self.G_h_lossfn = nn.MSELoss().to(self.device)
        elif G_h_lossfn_type == 'l2sum':
            self.G_h_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_h_lossfn_type == 'ssim':
            self.G_h_lossfn = SSIMLoss().to(self.device)
        elif G_h_lossfn_type == 'charbonnier':
            self.G_h_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_h_lossfn_type))
        
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.G_lossfn_lambda_p = self.opt_train['G_lossfn_lambda_p']
        self.G_lossfn_lambda_h = self.opt_train['G_lossfn_lambda_h']

    # ----------------------------------------
    # feed L/P/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        # L     ~ L [m, n]
        # P     ~ P [m, n]
        self.L, self.P, sigma, noise_level, lamb = self.prepro(data['H'])
        self.L = single42tensor4(rgb2ycbcr(tensor2single(self.L), only_y=False))[:, 0, :, :].unsqueeze(1).to(self.device)
        self.P = single42tensor4(rgb2ycbcr(tensor2single(self.P), only_y=False))[:, 0, :, :].unsqueeze(1).to(self.device)
        self.deg_dict = {
            'sigma'         : sigma,
            'noise_level'   : noise_level,
            'lamb'          : lamb
        }
        if need_H:
            self.H = single42tensor4(rgb2ycbcr(tensor2single(data['H']), only_y=False))[:, 0, :, :].unsqueeze(1).to(self.device)

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def test_feed_data(self, data, need_H=True):
        self.L = data['L']
        _, self.P, sigma, noise_level, lamb = self.prepro(data['H'])
        self.L = single2tensor4(rgb2ycbcr(tensor2single(self.L), only_y=False))[:, 0, :, :].unsqueeze(1).to(self.device)
        self.L_rgb = data['L'].to(self.device)
        self.P = single2tensor4(rgb2ycbcr(tensor2single(self.P), only_y=False))[:, 0, :, :].unsqueeze(1).to(self.device)
        self.deg_dict = {
            'sigma'         : sigma,
            'noise_level'   : noise_level,
            'lamb'          : lamb
        }
        if need_H:
            self.H = single2tensor4(rgb2ycbcr(tensor2single(data['H']), only_y=False))[:, 0, :, :].unsqueeze(1).to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # E     ~ P^[m, n]
        self.E, self.K = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()

        G_loss = self.G_lossfn_weight * (                                                       \
                    self.G_lossfn_lambda_p * self.G_p_lossfn(self.E, self.P) +                  \
                    self.G_lossfn_lambda_h * self.G_h_lossfn(self.K, torch.zeros_like(self.K))  \
            )
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # get L, P, E, K, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['L_rgb'] = self.L_rgb.detach()[0].float().cpu()
        out_dict['P'] = self.P.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        out_dict['K'] = self.K.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, P, E, K, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['P'] = self.P.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        out_dict['K'] = self.K.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict
    