import random
import torch.utils.data as data
import utils.utils_image as util


class DatasetGT(data.Dataset):
    '''
    # -----------------------------------------
    # Get H for SISR.
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetGT, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96

        # ------------------------------------
        # get paths of H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_H, 'Error: H path is empty.'
        

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # if train, get H patch
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_H.shape

            # --------------------------------
            # randomly crop the H patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H = util.single2tensor3(img_H)

        return {'H': img_H, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
