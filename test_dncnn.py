import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch

import sys
import datetime
import logging

from testfunctions.utils_logger import *
from testfunctions.utils_model import *
from testfunctions.utils_image import *

from model import DyNNet

'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# --------------------------------------------
# logger
# --------------------------------------------
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)
by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo          # model_zoo
   |--dncnn_15        # model_name
   |--dncnn_25
   |--dncnn_50
   |--dncnn_gray_blind
   |--dncnn_color_blind
   |--dncnn3
|--testset            # testsets
   |--set12           # testset_name
   |--bsd68
   |--cbsd68
|--results            # results
   |--set12_dncnn_15  # result_name = testset_name + '_' + model_name
   |--set12_dncnn_25
   |--bsd68_dncnn_15
# --------------------------------------------
"""


def main():
    n_channels = 3  # fixed, 1 for grayscale image, 3 for color image

    result_name = 'results_dncnn'  # fixed
    border = 1
    model_path = os.path.join('Pretrained/model_color.pth')  # path of the .pth model

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join('Datasets/Test')  # L_path, for Low-quality images
    H_path = L_path  # H_path, for High-quality images
    E_path = os.path.join('Results')  # E_path, for Estimated images

    need_degradation = True
    logger_name = result_name
    logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model = DyNNet()
    need_H = 'True'

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    model = model.to(device)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    test_results_L = OrderedDict()
    test_results_L['psnr'] = []
    test_results_L['ssim'] = []

    # logger.info('model_name:{}, image sigma:{}'.format(args.model_name, args.noise_level_img))
    logger.info(L_path)
    print(L_path)
    L_paths = get_image_paths(L_path)
    H_paths = get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))

        img_L = imread_uint(img, n_channels=n_channels)

        img_L = uint2single(img_L)

        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, 50 / 255., img_L.shape)

        imshow(single2uint(img_L), title='Noisy image with noise level {}'.format('50'))

        img_L = single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        img_E = model(img_L)

        img_E = tensor2uint(img_E)

        # Added
        img_L = tensor2uint(img_L)

        if need_H:
            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = calculate_psnr(img_E, img_H, border=border)
            ssim = calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            print('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name + ext, psnr, ssim))

            psnr_L = calculate_psnr(img_L, img_H, border=border)
            ssim_L = calculate_ssim(img_L, img_H, border=border)
            test_results_L['psnr'].append(psnr_L)
            test_results_L['ssim'].append(ssim_L)
            print('Low Quality: {:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name + ext, psnr_L, ssim_L))

            imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth')

            # ------------------------------------
        # save results
        # ------------------------------------

        imsave(img_E, os.path.join(E_path, img_name + ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))


if __name__ == '__main__':
    main()