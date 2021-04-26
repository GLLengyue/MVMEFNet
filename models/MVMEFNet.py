import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision.transforms as transforms
import cv2

from .stackhourglass import PSMNet
from .AHDR import AHDRNet

def warp(image, disp):

    disp = disp.squeeze(0)

    bs, channel, height, width = image.size()
    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=image.dtype, device=image.device),
                                    torch.arange(0, width, dtype=image.dtype, device=image.device)])

    mh = mh.reshape(1, height, width).repeat(bs, 1, 1)
    mw = mw.reshape(1, height, width).repeat(bs, 1, 1)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw + disp

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0

    grid = torch.stack([coords_x, coords_y], dim=3)

    image_warped = F.grid_sample(image, grid.view(bs, height, width, 2), mode='bilinear',
                                padding_mode='zeros')

    return image_warped

class MVMEFNet(nn.Module):
    def __init__(self, max_disp=370):
        super(MVMEFNet, self).__init__()
        self.disp_net = PSMNet(maxdisp=max_disp)
        self.fusion_net = AHDRNet()
    
    def forward(self, imgL_d, imgR_d, imgL_g, imgR_g, imgL_o, imgR_o):
        pred1, pred2, pred3 = self.disp_net(imgL_d, imgR_d)

        w_imgL_o = warp(imgL_o, pred3)
        w_imgL_g = warp(imgL_g, pred3)

        imgL_cat = torch.cat((w_imgL_o, w_imgL_g), dim=1)
        imgR_cat = torch.cat((imgR_o, imgR_g), dim=1)

        result, a_map = self.fusion_net(imgL_cat, imgR_cat)

        return pred1, pred2, pred3, w_imgL_o, result, a_map
        
