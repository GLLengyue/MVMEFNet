from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import numpy as np
import torchvision.transforms as transforms
import cv2




class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, maxdisp=192):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):
        left_fea     = self.feature_extraction(left)
        right_fea  = self.feature_extraction(right)

        #matching
        left_CV = Variable(torch.FloatTensor(left_fea.size()[0], left_fea.size()[1]*2, self.maxdisp//4,  left_fea.size()[2],  left_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             left_CV[:, :left_fea.size()[1], i, :,i:]   = left_fea[:,:,:,i:]
             left_CV[:, left_fea.size()[1]:, i, :,i:] = right_fea[:,:,:,:-i]

            else:
             left_CV[:, :left_fea.size()[1], i, :,:]   = left_fea
             left_CV[:, left_fea.size()[1]:, i, :,:]   = right_fea

        left_CV = left_CV.contiguous()

        left_CV0 = self.dres0(left_CV)
        left_CV0 = self.dres1(left_CV0) + left_CV0

        out1, pre1, post1 = self.dres2(left_CV0, None, None) 
        out1 = out1+left_CV0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+left_CV0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+left_CV0

        left_CV1 = self.classif1(out1)
        left_CV2 = self.classif2(out2) + left_CV1
        left_CV3 = self.classif3(out3) + left_CV2

        left_CV1 = F.upsample(left_CV1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        left_CV2 = F.upsample(left_CV2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

        left_CV1 = torch.squeeze(left_CV1,1)
        left_pred1 = F.softmax(left_CV1,dim=1)
        left_pred1 = disparityregression(self.maxdisp)(left_pred1)

        left_CV2 = torch.squeeze(left_CV2,1)
        left_pred2 = F.softmax(left_CV2,dim=1)
        left_pred2 = disparityregression(self.maxdisp)(left_pred2)

        left_CV3 = F.upsample(left_CV3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        left_CV3 = torch.squeeze(left_CV3,1)
        left_pred3 = F.softmax(left_CV3,dim=1)

        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching left_CV' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based left_CV volume provided flexibility.
        left_pred3 = disparityregression(self.maxdisp)(left_pred3)

        return left_pred1, left_pred2, left_pred3
