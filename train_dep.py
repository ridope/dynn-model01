# =============================CONV KERNEL======================================#
# torchvisison >= 0.9.0
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import deform_conv2d
import math
import logging

logger = logging.getLogger('base')


class DCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deformable_groups=1,
                 extra_offset_mask=True,
                 offset_in_channel=32
                 ):
        super(DCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.extra_offset_mask = extra_offset_mask

        self.conv_offset_mask = nn.Conv2d(offset_in_channel,
                                          deformable_groups * 3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=self.padding,
                                          bias=True)

        self.init_offset()

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        torch.nn.init.constant_(self.conv_offset_mask.weight, 0.)
        torch.nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):

        if self.extra_offset_mask:
            # x = [input, features]
            offset_mask = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > max(x.shape[2:]):
            logger.warning('Offset mean is {}, larger than max(h, w).'.format(offset_mean))

        out = deform_conv2d(input=x,
                            offset=offset,
                            weight=self.weight,
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            mask=mask
                            )

        return out


# =============================SADNET===========================================#

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from distutils.version import LooseVersion
import torchvision


# ==============================================================================#
class ResBlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class RSABlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32, offset_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.dcnpack = DCN(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                           extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class OffsetBlock(nn.Module):

    def __init__(self, input_channel=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel * 2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4 * output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


# ===============================================================================#
class SADNET(nn.Module):

    def __init__(self, input_channel=3, output_channel=3, n_channel=128, offset_channel=128):
        super().__init__()

        self.res1 = ResBlock(input_channel, n_channel)
        self.down1 = nn.Conv2d(n_channel, n_channel * 2, 2, 2)
        self.res2 = ResBlock(n_channel * 2, n_channel * 2)
        self.down2 = nn.Conv2d(n_channel * 2, n_channel * 4, 2, 2)
        self.res3 = ResBlock(n_channel * 4, n_channel * 4)
        self.down3 = nn.Conv2d(n_channel * 4, n_channel * 8, 2, 2)
        self.res4 = ResBlock(n_channel * 8, n_channel * 8)

        self.context = ContextBlock(n_channel * 8, n_channel * 2, square=False)
        self.offset4 = OffsetBlock(n_channel * 8, offset_channel, False)
        self.dres4 = RSABlock(n_channel * 8, n_channel * 8, offset_channel)

        self.up3 = nn.ConvTranspose2d(n_channel * 8, n_channel * 4, 2, 2)
        self.dconv3_1 = nn.Conv2d(n_channel * 8, n_channel * 4, 1, 1)
        self.offset3 = OffsetBlock(n_channel * 4, offset_channel, True)
        self.dres3 = RSABlock(n_channel * 4, n_channel * 4, offset_channel)

        self.up2 = nn.ConvTranspose2d(n_channel * 4, n_channel * 2, 2, 2)
        self.dconv2_1 = nn.Conv2d(n_channel * 4, n_channel * 2, 1, 1)
        self.offset2 = OffsetBlock(n_channel * 2, offset_channel, True)
        self.dres2 = RSABlock(n_channel * 2, n_channel * 2, offset_channel)

        self.up1 = nn.ConvTranspose2d(n_channel * 2, n_channel, 2, 2)
        self.dconv1_1 = nn.Conv2d(n_channel * 2, n_channel, 1, 1)
        self.offset1 = OffsetBlock(n_channel, offset_channel, True)
        self.dres1 = RSABlock(n_channel, n_channel, offset_channel)

        self.out = nn.Conv2d(n_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.res1(x)
        pool1 = self.lrelu(self.down1(conv1))
        conv2 = self.res2(pool1)
        pool2 = self.lrelu(self.down2(conv2))
        conv3 = self.res3(pool2)
        pool3 = self.lrelu(self.down3(conv3))
        conv4 = self.res4(pool3)
        conv4 = self.context(conv4)

        L4_offset = self.offset4(conv4, None)
        dconv4 = self.dres4(conv4, L4_offset)

        up3 = torch.cat([self.up3(dconv4), conv3], 1)
        up3 = self.dconv3_1(up3)
        L3_offset = self.offset3(up3, L4_offset)
        dconv3 = self.dres3(up3, L3_offset)

        up2 = torch.cat([self.up2(dconv3), conv2], 1)
        up2 = self.dconv2_1(up2)
        L2_offset = self.offset2(up2, L3_offset)
        dconv2 = self.dres2(up2, L2_offset)

        up1 = torch.cat([self.up1(dconv2), conv1], 1)
        up1 = self.dconv1_1(up1)
        L1_offset = self.offset1(up1, L2_offset)
        dconv1 = self.dres1(up1, L1_offset)

        out = self.out(dconv1) + x

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                # torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# ==============================================================================#

##########################################################

###################### RDU NET ================================================


import torch
import torch.nn as nn


@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x


class DyNNet(nn.Module):
    """
    Residual-Dense U-net for image denoising.
    """

    def __init__(self, **kwargs):
        super().__init__()

        channels = 3
        filters_0 = 128
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0
        offset_channel = filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.offset_3 = OffsetBlock(filters_3, offset_channel, False)
        self.rsab_3 = RSABlock(filters_3, filters_3, offset_channel)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.offset_2 = OffsetBlock(filters_2, offset_channel, True)
        self.rsab_2 = RSABlock(filters_2, filters_2, offset_channel)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.offset_1 = OffsetBlock(filters_1, offset_channel, True)
        self.rsab_1 = RSABlock(filters_1, filters_1, offset_channel)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.offset_0 = OffsetBlock(filters_0, offset_channel, True)
        self.rsab_0 = RSABlock(filters_0, filters_0, offset_channel)

        self.output_block = OutputBlock(filters_0, channels)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)  # Level 0
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)  # Level 1
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)  # Level 2
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)  # (Bottleneck)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)
        L3_offset = self.offset_3(out_3, None)
        dconv3 = self.rsab_3(out_3, L3_offset)

        out_4 = self.up_2([dconv3, out_2])  # Level 2
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)
        L2_offset = self.offset_2(out_4, L3_offset)
        dconv2 = self.rsab_2(out_4, L2_offset)

        out_5 = self.up_1([dconv2, out_1])  # Level 1
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)
        L1_offset = self.offset_1(out_5, L2_offset)
        dconv1 = self.rsab_1(out_5, L1_offset)

        out_6 = self.up_0([dconv1, out_0])  # Level 0
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)
        L0_offset = self.offset_0(out_6, L1_offset)
        dconv0 = self.rsab_0(out_6, L0_offset)

        return self.output_block(dconv0) + inputs

# Data managment

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from skimage.util import view_as_windows



def data_augmentation(image):
    augmented_images_arrays, augmented_images_list = [], []
    to_transform = [image, np.rot90(image, axes=(1, 2))]

    for t in to_transform:
        t_ud = t[:, ::-1, ...]
        t_lr = t[:, :, ::-1, ...]
        t_udlr = t_ud[:, :, ::-1, ...]

        flips = [t_ud, t_lr, t_udlr]
        augmented_images_arrays.extend(flips)

    augmented_images_arrays.extend(to_transform)

    for img in augmented_images_arrays:
        img_unbatch = list(img)
        augmented_images_list.extend(img_unbatch)

    return augmented_images_list


def create_patches(image, patch_size, step):
    image = view_as_windows(image, patch_size, step)
    h, w = image.shape[:2]
    image = np.reshape(image, (h * w, patch_size[0], patch_size[1], patch_size[2]))

    return image


class DataSampler(Sampler):

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        self.rand = np.random.RandomState(0)
        self.perm = []

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self._num_samples is not None:
            while len(self.perm) < self._num_samples:
                perm = self.rand.permutation(n).astype('int32').tolist()
                self.perm.extend(perm)
            idx = self.perm[:self._num_samples]
            self.perm = self.perm[self._num_samples:]
        else:
            idx = self.rand.permutation(n).astype('int32').tolist()

        return iter(idx)

    def __len__(self):
        return self.num_samples


class NoisyImagesDataset(Dataset):
    def __init__(self, files, channels, patch_size, transform=None, noise_transform=None):
        self.channels = channels
        self.patch_size = patch_size
        self.transform = transform
        self.noise_transforms = noise_transform
        self.to_tensor = ToTensor()
        self.dataset = {'image': [], 'noisy': []}
        self.load_dataset(files)

    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        image, noisy = self.dataset.get('image')[idx], self.dataset.get('noisy')[idx]
        sample = {'image': image, 'noisy': noisy}
        if self.transform is not None:
            sample = self.transform(sample)
        sample = self.to_tensor(sample)

        return sample.get('noisy'), sample.get('image')

    def load_dataset(self, files):
        patch_size = (self.patch_size, self.patch_size, self.channels)
        for file in tqdm(files):
            image = load_image(file, self.channels)
            if image is None:
                continue

            image = create_patches(image, patch_size, step=self.patch_size)
            sample = {'image': image, 'noisy': None}

            for noise_transform in self.noise_transforms:
                _sample = noise_transform(sample)
                image, noisy = _sample['image'], _sample['noisy']
                image, noisy = list(image), list(noisy)

                self.dataset['image'].extend(image)
                self.dataset['noisy'].extend(noisy)


# Train

import csv
import os
import torch
import time
import numpy as np
from tqdm import tqdm
from torch import optim


class EpochLogger:
    r"""
    Keeps a log of metrics in the current epoch.
    """
    def __init__(self):
        self.log = {
            'train loss': 0., 'train psnr': 0., 'train ssim': 0., 'val loss': 0., 'val psnr': 0., 'val ssim': 0.
        }

    def update_log(self, metrics, phase):
        """
        Update the metrics in the current epoch, this method is called at every step of the epoch.
        :param metrics: dict
            Metrics to update: loss, PSNR and SSIM.
        :param phase: str
            Phase of the current epoch: training (train) or validation (val).
        :return: None
        """
        for key, value in metrics.items():
            self.log[' '.join([phase, key])] += value

    def get_log(self, n_samples, phase):
        """
        Returns the average of the monitored metrics in the current moment,
        given the number of evaluated samples.
        :param n_samples: int
            Number of evaluated samples.
        :param phase: str
            Phase of the current epoch: training (train) or validation (val).
        :return: dic
            Log of the current phase in the training.
        """
        log = {
            phase + ' loss': self.log[phase + ' loss'] / n_samples,
            phase + ' psnr': self.log[phase + ' psnr'] / n_samples,
            phase + ' ssim': self.log[phase + ' ssim'] / n_samples
        }
        return log


class FileLogger(object):
    """
    Keeps a log of the whole training and validation process.
    The results are recorded in a CSV files.

    Args:
        file_path (string): path of the csv file.
    """
    def __init__(self, file_path):
        """
        Creates the csv record file.
        :param f
        """
        self.file_path = file_path
        header = ['epoch', 'lr', 'train loss', 'train psnr', 'train ssim', 'val loss', 'val psnr', 'val ssim']

        with open(self.file_path, 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(header)

    def __call__(self, epoch_log):
        """
        Updates the CSV record file.
        :param epoch_log: dict
            Log of the current epoch.
        :return: None
        """

        # Format log file:
        # Epoch and learning rate:
        log = ['{:03d}'.format(epoch_log['epoch']), '{:.5e}'.format(epoch_log['learning rate'])]

        # Training loss, PSNR, SSIM:
        log.extend([
            '{:.5e}'.format(epoch_log['train loss']),
            '{:.5f}'.format(epoch_log['train psnr']),
            '{:.5f}'.format(epoch_log['train ssim'])
        ])

        # Validation loss, PSNR, SSIM
        # Validation might not be done at all epochs, in that case the default calue is zero.
        log.extend([
            '{:.5e}'.format(epoch_log.get('val loss', 0.)),
            '{:.5f}'.format(epoch_log.get('val psnr', 0.)),
            '{:.5f}'.format(epoch_log.get('val ssim', 0.))
        ])

        with open(self.file_path, 'a') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(log)


def fit_model(model, data_loaders, channels, criterion, optimizer, scheduler, device, n_epochs, val_freq, checkpoint_dir, model_name):
    """
    Training of the denoiser model.
    :param model: torch Module
        Neural network to fit.
    :param data_loaders: dict
        Dictionary with torch DataLoaders with training and validation datasets.
    :param channels: int
        Number of image channels
    :param criterion: torch Module
        Loss function.
    :param optimizer: torch Optimizer
        Gradient descent optimization algorithm.
    :param scheduler: torch lr_scheduler
        Learning rate scheduler.
    :param device: torch device
        Device used during training (CPU/GPU).
    :param n_epochs: int
        Number of epochs to fit the model.
    :param val_freq: int
        How many training epochs to run between validations.
    :param checkpoint_dir: str
        Path to the directory where the model checkpoints and CSV log files will be stored.
    :param model_name: str
        Prefix name of the trained model saved in checkpoint_dir.
    :return: None
    """
    psnr = PSNR(data_range=1., reduction='sum')
    ssim = SSIM(channels, data_range=1., reduction='sum')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logfile_path = os.path.join(checkpoint_dir,  ''.join([model_name, '_logfile.csv']))
    model_path = os.path.join(checkpoint_dir, ''.join([model_name, '-{:03d}-{:.4e}-{:.4f}-{:.4f}.pth']))
    file_logger = FileLogger(logfile_path)
    best_model_path, best_psnr = '', -np.inf
    since = time.time()

    for epoch in range(1, n_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        epoch_logger = EpochLogger()
        epoch_log = dict()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print('\nEpoch: {}/{} - Learning rate: {:.4e}'.format(epoch, n_epochs, lr))
                description = 'Training - Loss:{:.5e} - PSNR:{:.5f} - SSIM:{:.5f}'
            elif phase == 'val' and epoch % val_freq == 0:
                model.eval()
                description = 'Validation - Loss:{:.5e} - PSNR:{:.5f} - SSIM:{:.5f}'
            else:
                break

            iterator = tqdm(enumerate(data_loaders[phase], 1), total=len(data_loaders[phase]), ncols=110)
            iterator.set_description(description.format(0, 0, 0))
            n_samples = 0

            for step, (inputs, targets) in iterator:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                n_samples += inputs.size()[0]
                metrics = {
                    'loss': loss.item() * inputs.size()[0],
                    'psnr': psnr(outputs, targets).item(),
                    'ssim': ssim(outputs, targets).item()
                }
                epoch_logger.update_log(metrics, phase)
                log = epoch_logger.get_log(n_samples, phase)
                iterator.set_description(description.format(log[phase + ' loss'], log[phase + ' psnr'], log[phase + ' ssim']))

            if phase == 'val':
                # Apply Reduce LR On Plateau if it is the case and save the model if the validation PSNR is improved.
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(log['val psnr'])
                if log['val psnr'] > best_psnr:
                    best_psnr = log['val psnr']
                    best_model_path = model_path.format(epoch, log['val loss'], log['val psnr'], log['val ssim'])
                    torch.save(model.state_dict(), best_model_path)

            elif scheduler is not None:         # Apply another scheduler at epoch level.
                scheduler.step()

            epoch_log = {**epoch_log, **log}

        # Save the current epoch metrics in a CVS file.
        epoch_data = {'epoch': epoch, 'learning rate': lr, **epoch_log}
        file_logger(epoch_data)

    # Save the last model and report training time.
    best_model_path = model_path.format(epoch, log['val loss'], log['val psnr'], log['val ssim'])
    torch.save(model.state_dict(), best_model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best PSNR: {:4f}'.format(best_psnr))


# Metrics

import torch
from pytorch_msssim import SSIM as _SSIM


class PSNR(object):
    r"""
    Evaluates the PSNR metric in a tensor.
    It can return a result with different reduction methods.

    Args:
        data_range (int, float): Range of the input images.
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
        eps (float): Epsilon value to avoid division by zero.
    """
    def __init__(self, data_range, reduction='none', eps=1e-8):
        self.data_range = data_range
        self.reduction = reduction
        self.eps = eps

    def __call__(self, outputs, targets):
        with torch.set_grad_enabled(False):
            mse = torch.mean((outputs - targets) ** 2., dim=(1, 2, 3))
            psnr = 10. * torch.log10((self.data_range ** 2.) / (mse + self.eps))

            if self.reduction == 'mean':
                return psnr.mean()
            if self.reduction == 'sum':
                return psnr.sum()

            return psnr


class SSIM(object):
    r"""
    Evaluates the SSIM metric in a tensor.
    It can return a result with different reduction methods.

    Args:
        channels (int): Number of channels of the images.
        data_range (int, float): Range of the input images.
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    """
    def __init__(self, channels, data_range, reduction='none'):
        self.data_range = data_range
        self.reduction = reduction
        self.ssim_module = _SSIM(data_range=data_range, size_average=False, channel=channels)

    def __call__(self, outputs, targets):
        with torch.set_grad_enabled(False):
            ssim = self.ssim_module(outputs, targets)

            if self.reduction == 'mean':
                return ssim.mean()
            if self.reduction == 'sum':
                return ssim.sum()

            return ssim


# Transforms

import random
import torch
import numpy as np


class AdditiveWhiteGaussianNoise(object):
    """Additive white gaussian noise generator."""
    def __init__(self, noise_level, fix_sigma=False, clip=False):
        self.noise_level = noise_level
        self.fix_sigma = fix_sigma
        self.rand = np.random.RandomState(1)
        self.clip = clip
        if not fix_sigma:
            self.predefined_noise = [i for i in range(5, noise_level + 1, 5)]

    def __call__(self, sample):
        """
        Generates additive white gaussian noise, and it is applied to the clean image.
        :param sample:
        :return:
        """
        image = sample.get('image')

        if image.ndim == 4:                 # if 'image' is a batch of images, we set a different noise level per image
            samples = image.shape[0]        # (Samples, Height, Width, Channels) or (Samples, Channels, Height, Width)
            if self.fix_sigma:
                sigma = self.noise_level * np.ones((samples, 1, 1, 1))
            else:
                sigma = np.random.choice(self.predefined_noise, size=(samples, 1, 1, 1))
            noise = self.rand.normal(0., 1., size=image.shape)
            noise = noise * sigma
        else:                               # else, 'image' is a simple image
            if self.fix_sigma:              # (Height, Width, Channels) or (Channels , Height, Width)
                sigma = self.noise_level
            else:
                sigma = self.rand.randint(5, self.noise_level)
            noise = self.rand.normal(0., sigma, size=image.shape)

        noisy = image + noise

        if self.clip:
            noisy = np.clip(noisy, 0., 255.)

        return {'image': image, 'noisy': noisy.astype('float32')}


class ToTensor(object):
    """Convert data sample to pytorch tensor"""
    def __call__(self, sample):
        image, noisy = sample.get('image'), sample.get('noisy')
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype('float32') / 255.)

        if noisy is not None:
            noisy = torch.from_numpy(noisy.transpose((2, 0, 1)).astype('float32') / 255.)

        return {'image': image, 'noisy': noisy}


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.flipud(image)

            if noisy is not None:
                noisy = np.flipud(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.fliplr(image)

            if noisy is not None:
                noisy = np.fliplr(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


class RandomRot90(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            image, noisy = sample.get('image'), sample.get('noisy')
            image = np.rot90(image)

            if noisy is not None:
                noisy = np.rot90(noisy)

            return {'image': image, 'noisy': noisy}

        return sample


# Utils

import random
import torch
import numpy as np
from skimage import io, color, img_as_ubyte


def load_image(image_path, channels):
    """
    Load image and change it color space from RGB to Grayscale if necessary.
    :param image_path: str
        Path of the image.
    :param channels: int
        Number of channels (3 for RGB, 1 for Grayscale)
    :return: numpy array
        Image loaded.
    """
    image = io.imread(image_path)

    if image.ndim == 3 and channels == 1:       # Convert from RGB to Grayscale and expand dims.
        image = img_as_ubyte(color.rgb2gray(image))
        return np.expand_dims(image, axis=-1)
    elif image.ndim == 2 and channels == 1:     # Handling grayscale images if needed.
        if image.dtype != 'uint8':
            image = img_as_ubyte(image)
        return np.expand_dims(image, axis=-1)

    return image


def mod_crop(image, mod):
    """
    Crops image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to crop.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy array
        Copped image
    """
    size = image.shape[:2]
    size = size - np.mod(size, mod)
    image = image[:size[0], :size[1], ...]

    return image


def mod_pad(image, mod):
    """
    Pads image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to pad.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy  array, tuple
        Padded image, original image size.
    """
    size = image.shape[:2]
    h, w = np.mod(size, mod)
    h, w = mod - h, mod - w
    if h != mod or w != mod:
        if image.ndim == 3:
            image = np.pad(image, ((0, h), (0, w), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((0, h), (0, w)), mode='reflect')

    return image, size


def set_seed(seed=1):
    """
    Sets all random seeds.
    :param seed: int
        Seed value.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ensemble(image, normalize=True):
    """
    Create image ensemble to estimate denoised image.
    :param image: numpy array
        Noisy image.
    :param normalize: bool
        Normalize image to range [0., 1.].
    :return: list
        Ensemble of noisy image transformed.
    """
    img_rot = np.rot90(image)
    ensemble_list = [
        image, np.fliplr(image), np.flipud(image), np.flipud(np.fliplr(image)),
        img_rot, np.fliplr(img_rot), np.flipud(img_rot), np.flipud(np.fliplr(img_rot))
    ]

    ensemble_transformed = []
    for img in ensemble_list:
        if img.ndim == 2:                                           # Expand dims for channel dimension in gray scale.
            img = np.expand_dims(img.copy(), 0)                     # Use copy to avoid problems with reverse indexing.
        else:
            img = np.transpose(img.copy(), (2, 0, 1))               # Channels-first transposition.
        if normalize:
            img = img / 255.

        img_t = torch.from_numpy(np.expand_dims(img, 0)).float()    # Expand dims again to create batch dimension.
        ensemble_transformed.append(img_t)

    return ensemble_transformed


def separate_ensemble(ensemble, return_single=False):
    """
    Apply inverse transforms to predicted image ensemble and average them.
    :param ensemble: list
        Predicted images, ensemble[0] is the original image,
        and ensemble[i] is a transformed version of ensemble[i].
    :param return_single: bool
        Return also ensemble[0] to evaluate single prediction
    :return: numpy array or tuple of numpy arrays
        Average of the predicted images, original image denoised.
    """
    ensemble_np = []

    for img in ensemble:
        img = img.squeeze()                     # Remove additional dimensions.
        if img.ndim == 3:                       # Transpose if necessary.
            img = np.transpose(img, (1, 2, 0))

        ensemble_np.append(img)

    # Apply inverse transforms to vertical and horizontal flips.
    img = ensemble_np[0] + np.fliplr(ensemble_np[1]) + np.flipud(ensemble_np[2]) + np.fliplr(np.flipud(ensemble_np[3]))

    # Apply inverse transforms to 90º rotation, vertical and horizontal flips
    img = img + np.rot90(ensemble_np[4], k=3) + np.rot90(np.fliplr(ensemble_np[5]), k=3)
    img = img + np.rot90(np.flipud(ensemble_np[6]), k=3) + np.rot90(np.fliplr(np.flipud(ensemble_np[7])), k=3)

    # Average and clip final predicted image.
    img = img / 8.
    img = np.clip(img, 0., 1.)

    if return_single:
        return img, ensemble_np[0]
    else:
        return img


def predict_ensemble(model, ensemble, device):
    """
    Predict batch of images from an ensemble.
    :param model: torch Module
        Trained model to estimate denoised images.
    :param ensemble: list
        Images to estimate.
    :param device: torch device
        Device of the trained model.
    :return: list
        Estimated images of type numpy ndarray.
    """
    y_hat_ensemble = []

    for x in ensemble:
        x = x.to(device)

        with torch.no_grad():
            y_hat = model(x)
            y_hat_ensemble.append(y_hat.cpu().detach().numpy().astype('float32'))

    return y_hat_ensemble


import yaml
import torch
import torch.optim as optim
from os.path import join
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ptflops import get_model_complexity_info


def main():
    with open('config.yaml', 'r') as stream:                # Load YAML configuration file.
        config = yaml.safe_load(stream)

    model_params = config['model']
    train_params = config['train']
    val_params = config['val']

    # Defining model:
    set_seed(0)
    model = DyNNet()

    print('Model summary:')
    test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Define the model name and use multi-GPU if it is allowed.
    model_name = 'model_color' if model_params['channels'] == 3 else 'model_gray'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(train_params['device'])
    print("Using device: {}".format(device))
    #if torch.cuda.device_count() > 1 and 'cuda' in device.type and train_params['multi gpu']:
    #    model = nn.DataParallel(model)
    #    print('Using multiple GPUs')

    model = model.to(device)
    param_group = []
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            p = {'params': param, 'weight_decay': train_params['weight decay']}
        else:
            p = {'params': param, 'weight_decay': 0.}
        param_group.append(p)

    # Load training and validation file names.
    # Modify .txt files if datasets do not fit in memory.
    with open('train_files.txt', 'r') as f_train, open('val_files.txt', 'r') as f_val:
        raw_train_files = f_train.read().splitlines()
        raw_val_files = f_val.read().splitlines()
        train_files = list(map(lambda file: join(train_params['dataset path'], file), raw_train_files))
        val_files = list(map(lambda file: join(val_params['dataset path'], file), raw_val_files))

    training_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRot90()
    ])

    # Predefined noise level
    train_noise_transform = [AdditiveWhiteGaussianNoise(train_params['noise level'], clip=True)]
    val_noise_transforms = [AdditiveWhiteGaussianNoise(s, fix_sigma=True, clip=True) for s in val_params['noise levels']]

    print('\nLoading training dataset:')
    training_dataset = NoisyImagesDataset(train_files,
                                          model_params['channels'],
                                          train_params['patch size'],
                                          training_transforms,
                                          train_noise_transform)

    print('\nLoading validation dataset:')
    validation_dataset = NoisyImagesDataset(val_files,
                                            model_params['channels'],
                                            val_params['patch size'],
                                            None,
                                            val_noise_transforms)
    # Training in sub-epochs:
    print('Training patches:', len(training_dataset))
    print('Validation patches:', len(validation_dataset))
    n_samples = len(training_dataset) // train_params['dataset splits']
    n_epochs = train_params['epochs'] * train_params['dataset splits']
    sampler = DataSampler(training_dataset, num_samples=n_samples)

    data_loaders = {
        'train': DataLoader(training_dataset, train_params['batch size'], num_workers=train_params['workers'], sampler=sampler),
        'val': DataLoader(validation_dataset, val_params['batch size'], num_workers=val_params['workers']),
    }

    # Optimization:
    learning_rate = train_params['learning rate']
    step_size = train_params['scheduler step'] * train_params['dataset splits']

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(param_group, lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=train_params['scheduler gamma'])

    # Train the model
    fit_model(model, data_loaders, model_params['channels'], criterion, optimizer, lr_scheduler, device,
              n_epochs, val_params['frequency'], train_params['checkpoint path'], model_name)


if __name__ == '__main__':
    main()
