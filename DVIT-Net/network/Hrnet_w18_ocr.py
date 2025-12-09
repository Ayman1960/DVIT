from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import InstanceNorm2d

if torch.__version__.startswith("1"):
    relu_inplace = True
else:
    relu_inplace = False
from yacs.config import CfgNode as CN


# configs for HRNet48
HRNET_48 = CN()
HRNET_48.FINAL_CONV_KERNEL = 1

HRNET_48.STAGE1 = CN()
HRNET_48.STAGE1.NUM_MODULES = 1
HRNET_48.STAGE1.NUM_BRANCHES = 1
HRNET_48.STAGE1.NUM_BLOCKS = [4]
HRNET_48.STAGE1.NUM_CHANNELS = [64]
HRNET_48.STAGE1.BLOCK = "BOTTLENECK"
HRNET_48.STAGE1.FUSE_METHOD = "SUM"

HRNET_48.STAGE2 = CN()
HRNET_48.STAGE2.NUM_MODULES = 1
HRNET_48.STAGE2.NUM_BRANCHES = 2
HRNET_48.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_48.STAGE2.NUM_CHANNELS = [48, 96]
HRNET_48.STAGE2.BLOCK = "BASIC"
HRNET_48.STAGE2.FUSE_METHOD = "SUM"

HRNET_48.STAGE3 = CN()
HRNET_48.STAGE3.NUM_MODULES = 4
HRNET_48.STAGE3.NUM_BRANCHES = 3
HRNET_48.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_48.STAGE3.NUM_CHANNELS = [48, 96, 192]
HRNET_48.STAGE3.BLOCK = "BASIC"
HRNET_48.STAGE3.FUSE_METHOD = "SUM"

HRNET_48.STAGE4 = CN()
HRNET_48.STAGE4.NUM_MODULES = 3
HRNET_48.STAGE4.NUM_BRANCHES = 4
HRNET_48.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_48.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
HRNET_48.STAGE4.BLOCK = "BASIC"
HRNET_48.STAGE4.FUSE_METHOD = "SUM"


# configs for HRNet32
HRNET_32 = CN()
HRNET_32.FINAL_CONV_KERNEL = 1

HRNET_32.STAGE1 = CN()
HRNET_32.STAGE1.NUM_MODULES = 1
HRNET_32.STAGE1.NUM_BRANCHES = 1
HRNET_32.STAGE1.NUM_BLOCKS = [4]
HRNET_32.STAGE1.NUM_CHANNELS = [64]
HRNET_32.STAGE1.BLOCK = "BOTTLENECK"
HRNET_32.STAGE1.FUSE_METHOD = "SUM"

HRNET_32.STAGE2 = CN()
HRNET_32.STAGE2.NUM_MODULES = 1
HRNET_32.STAGE2.NUM_BRANCHES = 2
HRNET_32.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_32.STAGE2.NUM_CHANNELS = [32, 64]
HRNET_32.STAGE2.BLOCK = "BASIC"
HRNET_32.STAGE2.FUSE_METHOD = "SUM"

HRNET_32.STAGE3 = CN()
HRNET_32.STAGE3.NUM_MODULES = 4
HRNET_32.STAGE3.NUM_BRANCHES = 3
HRNET_32.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_32.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRNET_32.STAGE3.BLOCK = "BASIC"
HRNET_32.STAGE3.FUSE_METHOD = "SUM"

HRNET_32.STAGE4 = CN()
HRNET_32.STAGE4.NUM_MODULES = 3
HRNET_32.STAGE4.NUM_BRANCHES = 4
HRNET_32.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_32.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRNET_32.STAGE4.BLOCK = "BASIC"
HRNET_32.STAGE4.FUSE_METHOD = "SUM"


# configs for HRNet18
HRNET_18 = CN()
HRNET_18.PRETRAINED_LAYERS = ["*"]
HRNET_18.STEM_INPLANES = 64
HRNET_18.FINAL_CONV_KERNEL = 1
HRNET_18.WITH_HEAD = True

HRNET_18.STAGE2 = CN()
HRNET_18.STAGE2.NUM_MODULES = 1
HRNET_18.STAGE2.NUM_BRANCHES = 2
HRNET_18.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
HRNET_18.STAGE2.BLOCK = "BASIC"
HRNET_18.STAGE2.FUSE_METHOD = "SUM"

HRNET_18.STAGE3 = CN()
HRNET_18.STAGE3.NUM_MODULES = 4
HRNET_18.STAGE3.NUM_BRANCHES = 3
HRNET_18.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
HRNET_18.STAGE3.BLOCK = "BASIC"
HRNET_18.STAGE3.FUSE_METHOD = "SUM"

HRNET_18.STAGE4 = CN()
HRNET_18.STAGE4.NUM_MODULES = 3
HRNET_18.STAGE4.NUM_BRANCHES = 4
HRNET_18.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET_18.STAGE4.BLOCK = "BASIC"
HRNET_18.STAGE4.FUSE_METHOD = "SUM"


MODEL_CONFIGS = {
    "hrnet18": HRNET_18,
    "hrnet32": HRNET_32,
    "hrnet48": HRNET_48,
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def label_to_onehot(gt, num_classes, ignore_index=-1):
    """
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    """
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)

    return onehot.permute(0, 3, 1, 2)


class BatchNormReLU(nn.Module):
    def __init__(self, num_features, bn_type=None):
        super(BatchNormReLU, self).__init__()
        self.batchnorm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.size(2) == 1 and x.size(3) == 1:
            return self.relu(x)  # Skip batchnorm if spatial size is 1x1
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            bn_type=None,
            bn_momentum=bn_momentum,
        )
        self.fuse_layers = self._make_fuse_layers(bn_type=None, bn_momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        stride=1,
        bn_type=None,
        bn_momentum=0.1,
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=bn_momentum
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                bn_type=None,
                bn_momentum=bn_momentum,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    bn_type=None,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches, block, num_blocks, num_channels, bn_type, bn_momentum=0.1
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    bn_type=None,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, bn_type, bn_momentum=0.1):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=bn_momentum),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=bn_momentum
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=bn_momentum
                                    ),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, bn_type, bn_momentum, **kwargs):
        self.inplanes = 64
        super(HighResolutionNet, self).__init__()

        if os.environ.get("full_res_stem"):
            stem_stride = 1
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=False)
            self.layer1 = self._make_layer(
                Bottleneck, 64, 64, 4, bn_type=None, bn_momentum=bn_momentum
            )
        else:
            stem_stride = 2
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
            self.conv2 = nn.Conv2d(
                64, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=False)
            self.layer1 = self._make_layer(
                Bottleneck, 64, 64, 4, bn_type=None, bn_momentum=bn_momentum
            )

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition1 = self._make_transition_layer(
            [256], num_channels, bn_type=None, bn_momentum=bn_momentum
        )

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, bn_type=None, bn_momentum=bn_momentum
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=None, bn_momentum=bn_momentum
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, bn_type=None, bn_momentum=bn_momentum
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=None, bn_momentum=bn_momentum
        )

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            bn_type=None,
            bn_momentum=bn_momentum,
        )

        if os.environ.get("keep_imagenet_head"):
            self.incre_modules, self.downsamp_modules, self.final_layer = (
                self._make_head(
                    pre_stage_channels, bn_type=None, bn_momentum=bn_momentum
                )
            )

    def _make_head(self, pre_stage_channels, bn_type, bn_momentum):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block,
                channels,
                head_channels[i],
                1,
                bn_type=None,
                bn_momentum=bn_momentum,
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer, bn_type, bn_momentum
    ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=bn_momentum
                            ),
                            nn.ReLU(inplace=False),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=bn_momentum),
                            nn.ReLU(inplace=False),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self, block, inplanes, planes, blocks, stride=1, bn_type=None, bn_momentum=0.1
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample,
                bn_type=None,
                bn_momentum=bn_momentum,
            )
        )

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes, planes, bn_type=None, bn_momentum=bn_momentum)
            )

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config,
        num_inchannels,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    bn_type,
                    bn_momentum,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        if os.environ.get("full_res_stem"):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        if os.environ.get("drop_stage4"):
            return y_list

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        if os.environ.get("keep_imagenet_head"):
            # Classification Head
            x_list = []
            y = self.incre_modules[0](y_list[0])
            x_list.append(y)
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](
                    y
                )
                x_list.append(y)

            y = self.final_layer(y)
            del x_list[-1]
            x_list.append(y)

            return x_list

        return y_list


class HighResolutionNext(nn.Module):

    def __init__(self, cfg, bn_type, **kwargs):
        super(HighResolutionNext, self).__init__()
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(relu_inplace)

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage1_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition0 = self._make_transition_layer([64], num_channels, bn_type=None)
        self.stage1, pre_stage_channels = self._make_stage(
            self.stage1_cfg, num_channels, bn_type=None
        )

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=None
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, bn_type=None
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=None
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, bn_type=None
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=None
        )
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, bn_type=None
        )

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer, bn_type
    ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(relu_inplace),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(relu_inplace),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self, layer_config, num_inchannels, multi_scale_output=True, bn_type=None
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    bn_type,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_list = []
        for i in range(self.stage1_cfg["NUM_BRANCHES"]):
            if self.transition0[i] is not None:
                x_list.append(self.transition0[i](x))
            else:
                x_list.append(x)
        y_list = self.stage1(x_list)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                if i == 0:
                    x_list.append(self.transition1[i](y_list[0]))
                else:
                    x_list.append(self.transition1[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        return x


class _ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNormReLU(self.key_channels, bn_type=None),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNormReLU(self.key_channels, bn_type=None),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNormReLU(self.key_channels, bn_type=None),
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNormReLU(self.key_channels, bn_type=None),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNormReLU(self.key_channels, bn_type=None),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(
                in_channels=self.key_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            BatchNormReLU(self.in_channels, bn_type=None),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode="bilinear")

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(
            in_channels, key_channels, scale, bn_type=None
        )


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(
        self,
        in_channels,
        key_channels,
        out_channels,
        scale=1,
        dropout=0.1,
        bn_type=None,
    ):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(
            in_channels, key_channels, scale, bn_type
        )
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            BatchNormReLU(out_channels, bn_type=None),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class SpatialGather_Module(nn.Module):
    """
    Aggregate the context features according to the initial
    predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = (
            torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        )  # batch x k x c
        return ocr_context


class HRNet_W48_OCR(nn.Module):
    def __init__(
        self,
    ):
        super(HRNet_W48_OCR, self).__init__()
        self.num_classes = 1
        hrnet18 = HighResolutionNet(
            MODEL_CONFIGS["hrnet18"], bn_type="torchsyncbn", bn_momentum=0.1
        )
        self.backbone = hrnet18

        in_channels = 270
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=512,
            key_channels=256,
            out_channels=512,
        )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        out_aux_seg = []
        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)
        # feats : 512 128 out_aux:1 128
        context = self.ocr_gather_head(feats, out_aux)
        # context:512 1
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        out_aux = out_aux + out
        out_aux = F.interpolate(
            out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True
        )

        return F.sigmoid(out_aux)


if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 512, 512)
    model = HRNet_W48_OCR()
    output = model(input_tensor)
    print(output.shape)
