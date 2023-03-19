from models.model_plain import ModelPlain
from utils import utils_deg as deg
from collections import OrderedDict
import torch
import torch.nn as nn
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
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
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.G_lossfn_lambda = self.opt_train['G_lossfn_lambda']

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        # L     ~ L [m, n]
        # P     ~ P [m, n]
        self.L, self.P, sigma, noise_level, lamb = self.prepro(data['H'])
        self.L = self.L.to(self.device)
        self.P = self.P.to(self.device)
        self.deg_dict = {
            'sigma'         : sigma,
            'noise_level'   : noise_level,
            'lamb'          : lamb
        }
        if need_H:
            self.H = data['H'].to(self.device)

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

        G_loss = self.G_lossfn_weight * (                                                    \
                    self.G_lossfn(self.E, self.P) +                                          \
                    self.G_lossfn_lambda * self.G_lossfn(self.K, torch.zeros_like(self.K))   \
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
    # get L, P, E, K, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
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