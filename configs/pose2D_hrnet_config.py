# ------------------------------------------------------------------------------
# HRNet code from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode


_C = CfgNode()

# Model
_C.MODEL = CfgNode()
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.IMAGE_SIZE = [288, 384]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [72, 96]  # width * height, ex: 24 * 32

_C.MODEL.EXTRA = CfgNode(new_allowed=True)
_C.MODEL.EXTRA.PRETRAINED_LAYERS = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
                                    'stage2', 'transition2', 'stage3', 'transition3', 'stage4']
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1

_C.MODEL.EXTRA.STAGE2 = CfgNode()
_C.MODEL.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
_C.MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE3 = CfgNode()
_C.MODEL.EXTRA.STAGE3.NUM_MODULES = 4
_C.MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
_C.MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.EXTRA.STAGE4 = CfgNode()
_C.MODEL.EXTRA.STAGE4.NUM_MODULES = 3
_C.MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
_C.MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
_C.MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'

# Testing
_C.TEST = CfgNode()
_C.TEST.POST_PROCESS = False
_C.TEST.OBJECT_DET_THRESH = 0.95


def get_pose2D_hrnet_cfg_defaults():
    return _C.clone()

