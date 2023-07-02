from models.model_plain import ModelPlain
from utils import utils_deg as deg
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from utils.utils_image import tensor2single, single2tensor4, single42tensor4, rgb2ycbcr
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelBWSRSwinIR(ModelPlain):
    
    def __init__(self, opt):
        super(ModelBWSRSwinIR, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.prepro = deg.SRMDPreprocessing(
            scale=self.opt["scale"], cuda=True, **self.opt["degradation"]
        )

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        load_path_bwsr = self.opt['netG']['bwsr']['checkpoint_path']

        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
            
        if load_path_bwsr is not None:
            print('Loading model for BWSR [{:s}] ...'.format(load_path_bwsr))
            self.load_network(load_path_bwsr, self.netG.module.bwsr_net, strict=self.opt_train['G_param_strict'], param_key='params')

        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        # --------------------------------------------------------
        # define loss function for image similarity $f_1(\hat{p}) - x$
        # --------------------------------------------------------
        G_f1_lossfn_type = self.opt_train['G_f1_lossfn_type']
        if G_f1_lossfn_type == 'l1':
            self.G_f1_lossfn = nn.L1Loss().to(self.device)
        elif G_f1_lossfn_type == 'l2':
            self.G_f1_lossfn = nn.MSELoss().to(self.device)
        elif G_f1_lossfn_type == 'l2sum':
            self.G_f1_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_f1_lossfn_type == 'ssim':
            self.G_f1_lossfn = SSIMLoss().to(self.device)
        elif G_f1_lossfn_type == 'charbonnier':
            self.G_f1_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_f1_lossfn_type))
        
        # --------------------------------------------------------
        # define loss function for image similarity $f_2(\hat{\hat{p}}) - x$
        # --------------------------------------------------------
        G_f2_lossfn_type = self.opt_train['G_f2_lossfn_type']
        if G_f2_lossfn_type == 'l1':
            self.G_f2_lossfn = nn.L1Loss().to(self.device)
        elif G_f2_lossfn_type == 'l2':
            self.G_f2_lossfn = nn.MSELoss().to(self.device)
        elif G_f2_lossfn_type == 'l2sum':
            self.G_f2_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_f2_lossfn_type == 'ssim':
            self.G_f2_lossfn = SSIMLoss().to(self.device)
        elif G_f2_lossfn_type == 'charbonnier':
            self.G_f2_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_f2_lossfn_type))
        
        # --------------------------------------------------------
        # define loss function for image similarity $f_3(y) - x$
        # --------------------------------------------------------
        G_f3_lossfn_type = self.opt_train['G_f3_lossfn_type']
        if G_f3_lossfn_type == 'l1':
            self.G_f3_lossfn = nn.L1Loss().to(self.device)
        elif G_f3_lossfn_type == 'l2':
            self.G_f3_lossfn = nn.MSELoss().to(self.device)
        elif G_f3_lossfn_type == 'l2sum':
            self.G_f3_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_f3_lossfn_type == 'ssim':
            self.G_f3_lossfn = SSIMLoss().to(self.device)
        elif G_f3_lossfn_type == 'charbonnier':
            self.G_f3_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_f3_lossfn_type))
        
        # --------------------------------------------------------
        # define loss function for image similarity $\hat{x} - x$
        # --------------------------------------------------------
        G_f4_lossfn_type = self.opt_train['G_f4_lossfn_type']
        if G_f4_lossfn_type == 'l1':
            self.G_f4_lossfn = nn.L1Loss().to(self.device)
        elif G_f4_lossfn_type == 'l2':
            self.G_f4_lossfn = nn.MSELoss().to(self.device)
        elif G_f4_lossfn_type == 'l2sum':
            self.G_f4_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_f4_lossfn_type == 'ssim':
            self.G_f4_lossfn = SSIMLoss().to(self.device)
        elif G_f4_lossfn_type == 'charbonnier':
            self.G_f4_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_f4_lossfn_type))
        
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.G_lossfn_lambda_f1 = self.opt_train['G_lossfn_lambda_f1']
        self.G_lossfn_lambda_f2 = self.opt_train['G_lossfn_lambda_f2']
        self.G_lossfn_lambda_f3 = self.opt_train['G_lossfn_lambda_f3']
        self.G_lossfn_lambda_f4 = self.opt_train['G_lossfn_lambda_f4']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        self.requires_grad(self.netG.module.bwsr_net, flag=False)
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # feed L/P/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        # L     ~ L [m, n]
        # P     ~ P [m, n]
        L_img, _, sigma, noise_level, lamb = self.prepro(data['H'])
        self.L = L_img.to(self.device)
        self.deg_dict = {
            'sigma'         : sigma,
            'noise_level'   : noise_level,
            'lamb'          : lamb
        }
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def test_feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        # E     ~ f_1(\hat{p}[m, n])
        # Q     ~ f_2(\hat{\hat{p}}[m, n])
        # D     ~ f_3(y[m, n])
        # F     ~ f_4(concat[f_1(\hat{p}), f_2(\hat{\hat{p}}), f_3(y)][m, n])
        *_, self.F = self.netG(self.L)
        # print(f'{self.E.shape=}')
        # print(f'{self.Q.shape=}')
        # print(f'{self.D.shape=}')
        # print(f'{self.F.shape=}')
        
    # ----------------------------------------
    # feed L to netE
    # ----------------------------------------
    def netE_forward(self):
        # E     ~ f_1(\hat{p}[m, n])
        # Q     ~ f_2(\hat{\hat{p}}[m, n])
        # D     ~ f_3(y[m, n])
        # F     ~ f_4(concat[f_1(\hat{p}), f_2(\hat{\hat{p}}), f_3(y)][m, n])
        *_, self.F = self.netE(self.L)
        # print(f'{self.E.shape=}')
        # print(f'{self.Q.shape=}')
        # print(f'{self.D.shape=}')
        # print(f'{self.F.shape=}')

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()

        G_loss = self.G_lossfn_weight * (                                                       \
                    # self.G_lossfn_lambda_f1 * self.G_f1_lossfn(self.E, self.H) +                \
                    # self.G_lossfn_lambda_f2 * self.G_f2_lossfn(self.Q, self.H) +                \
                    # self.G_lossfn_lambda_f3 * self.G_f3_lossfn(self.D, self.H) +                \
                    self.G_lossfn_lambda_f4 * self.G_f4_lossfn(self.F, self.H)                  \
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
        with torch.no_grad():
            self.netE_forward()

    # ----------------------------------------
    # get L, E, Q, D, F, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        # out_dict['E'] = self.E.detach()[0].float().cpu()
        # out_dict['Q'] = self.Q.detach()[0].float().cpu()
        # out_dict['D'] = self.D.detach()[0].float().cpu()
        out_dict['F'] = self.F.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, Q, D, F, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        # out_dict['E'] = self.E.detach().float().cpu()
        # out_dict['Q'] = self.Q.detach().float().cpu()
        # out_dict['D'] = self.D.detach().float().cpu()
        out_dict['F'] = self.F.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict
    
