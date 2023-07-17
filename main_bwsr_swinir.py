import os
from glob import glob
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from skimage.restoration import unsupervised_wiener
import subprocess

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils import utils_deblur as deblur
from utils import utils_sisr as sisr

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for Kernel Estimate Network
# --------------------------------------------
# AmirMohammad Babaei (amirmohamad.babaei79@gmail.com)
# Nashra Babar (nashrababar111@gmail.com)
# github: https://github.com/cszn/KAIR  #TODO change the repo url
# --------------------------------------------
'''


def main(json_path='options/train_bwsr_swinir.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    writer = SummaryWriter(os.path.join(opt['path']['log'], 'runs'))


    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(opt['train']['epochs']):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
                writer.add_scalar('train_G_loss', logs['G_loss'], current_step)
                writer.add_scalar('train_G_lr', model.current_learning_rate(), current_step)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                for file in glob('/content/BWSR/superresolution/BWSR-SwinIRx4/models/*'):
                    os.remove(file)
                model.save(current_step)
                print(subprocess.run(['bash', '/content/BWSR/save_checkpoint.sh']))

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                
                test_step = current_step // opt['train']['checkpoint_test'] - 1
                test_size = len(test_loader)
                # avg_psnr_E = 0.0
                # avg_psnr_Q = 0.0
                # avg_psnr_D = 0.0
                avg_psnr_F = 0.0
                # avg_ssim_E = 0.0
                # avg_ssim_Q = 0.0
                # avg_ssim_D = 0.0
                avg_ssim_F = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.test_feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    L_img = util.tensor2uint(visuals['L'])
                    # E_img = util.tensor2uint(visuals['E'])
                    # Q_img = util.tensor2uint(visuals['Q'])
                    # D_img = util.tensor2uint(visuals['D'])
                    F_img = util.tensor2uint(visuals['F'])
                    H_img = util.tensor2uint(visuals['H'])

                    # E_img_ycbcr = util.rgb2ycbcr(E_img)
                    # Q_img_ycbcr = util.rgb2ycbcr(Q_img)
                    # D_img_ycbcr = util.rgb2ycbcr(D_img)
                    F_img_ycbcr = util.rgb2ycbcr(F_img)
                    H_img_ycbcr = util.rgb2ycbcr(H_img)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    # current_psnr_E    = util.calculate_psnr(E_img_ycbcr, H_img_ycbcr, border=border)
                    # current_psnr_Q    = util.calculate_psnr(Q_img_ycbcr, H_img_ycbcr, border=border)
                    # current_psnr_D    = util.calculate_psnr(D_img_ycbcr, H_img_ycbcr, border=border)
                    current_psnr_F    = util.calculate_psnr(F_img_ycbcr, H_img_ycbcr, border=border)

                    # logger.info('PSNR(E, H): {:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr_E))
                    # logger.info('PSNR(Q, H): {:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr_Q))
                    # logger.info('PSNR(D, H): {:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr_D))
                    logger.info('PSNR(F, H): {:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr_F))

                    # avg_psnr_E    += current_psnr_E
                    # avg_psnr_Q    += current_psnr_Q
                    # avg_psnr_D    += current_psnr_D
                    avg_psnr_F    += current_psnr_F

                    # writer.add_scalar('val_G_psnr_E', current_psnr_E, test_size * test_step + idx)
                    # writer.add_scalar('val_G_psnr_Q', current_psnr_Q, test_size * test_step + idx)
                    # writer.add_scalar('val_G_psnr_D', current_psnr_D, test_size * test_step + idx)
                    writer.add_scalar('val_G_psnr_F', current_psnr_F, test_size * test_step + idx)

                    # -----------------------
                    # calculate SSIM
                    # -----------------------
                    # current_ssim_E    = util.calculate_ssim(E_img_ycbcr, H_img_ycbcr, border=border)
                    # current_ssim_Q    = util.calculate_ssim(Q_img_ycbcr, H_img_ycbcr, border=border)
                    # current_ssim_D    = util.calculate_ssim(D_img_ycbcr, H_img_ycbcr, border=border)
                    current_ssim_F    = util.calculate_ssim(F_img_ycbcr, H_img_ycbcr, border=border)

                    # logger.info('SSIM(E, H): {:->4d}--> {:>10s} | {:<5.4f}'.format(idx, image_name_ext, current_ssim_E))
                    # logger.info('SSIM(Q, H): {:->4d}--> {:>10s} | {:<5.4f}'.format(idx, image_name_ext, current_ssim_Q))
                    # logger.info('SSIM(D, H): {:->4d}--> {:>10s} | {:<5.4f}'.format(idx, image_name_ext, current_ssim_D))
                    logger.info('SSIM(F, H): {:->4d}--> {:>10s} | {:<5.4f}'.format(idx, image_name_ext, current_ssim_F))

                    # avg_ssim_E    += current_ssim_E
                    # avg_ssim_Q    += current_ssim_Q
                    # avg_ssim_D    += current_ssim_D
                    avg_ssim_F    += current_ssim_F

                    # writer.add_scalar('val_G_ssim_E', current_ssim_E, test_size * test_step + idx)
                    # writer.add_scalar('val_G_ssim_Q', current_ssim_Q, test_size * test_step + idx)
                    # writer.add_scalar('val_G_ssim_D', current_ssim_D, test_size * test_step + idx)
                    writer.add_scalar('val_G_ssim_F', current_ssim_F, test_size * test_step + idx)

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}_{:05.2f}_{:5.4f}.png'.format(img_name, current_step, current_psnr_F, current_ssim_F))
                    # if test_step == 0:
                    #     util.imsave(P_img, os.path.join(img_dir, f'{img_name}.png'))
                    util.imsave(F_img, save_img_path)

                # avg_psnr_E = avg_psnr_E / idx
                # avg_psnr_Q = avg_psnr_Q / idx
                # avg_psnr_D = avg_psnr_D / idx
                avg_psnr_F = avg_psnr_F / idx

                # avg_ssim_E = avg_ssim_E / idx
                # avg_ssim_Q = avg_ssim_Q / idx
                # avg_ssim_D = avg_ssim_D / idx
                avg_ssim_F = avg_ssim_F / idx

                # testing log
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR E  : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr_E))
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR Q  : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr_Q))
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR D  : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr_D))
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR F  : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr_F))

                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average SSIM E  : {:<.4f}\n'.format(epoch, current_step, avg_ssim_E))
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average SSIM Q  : {:<.4f}\n'.format(epoch, current_step, avg_ssim_Q))
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average SSIM D  : {:<.4f}\n'.format(epoch, current_step, avg_ssim_D))
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average SSIM F  : {:<.4f}\n'.format(epoch, current_step, avg_ssim_F))

                # writer.add_scalar('val_G_avg_psnr_E', avg_psnr_E, current_step)
                # writer.add_scalar('val_G_avg_psnr_Q', avg_psnr_Q, current_step)
                # writer.add_scalar('val_G_avg_psnr_D', avg_psnr_D, current_step)
                writer.add_scalar('val_G_avg_psnr_F', avg_psnr_F, current_step)

                # writer.add_scalar('val_G_avg_ssim_E', avg_ssim_E, current_step)
                # writer.add_scalar('val_G_avg_ssim_Q', avg_ssim_Q, current_step)
                # writer.add_scalar('val_G_avg_ssim_D', avg_ssim_D, current_step)
                writer.add_scalar('val_G_avg_ssim_F', avg_ssim_F, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()

