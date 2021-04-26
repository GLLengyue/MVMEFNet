import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from math import exp
from dataloader import MiddleburyDataset
from models import MVMEFNet
from utils import batch_PSNR, batch_me

torch.manual_seed(0)
torch.cuda.manual_seed(0)

l = [
    'Adirondack-perfect',
    'Backpack-perfect',
    # 'Bicycle1-perfect',
    'Cable-perfect',
    # 'Classroom1-perfect',
    'Couch-perfect', # 630
    'Flowers-perfect', # 640
    'Jadeplant-perfect', # 640
    'Mask-perfect', # 460
    'Motorcycle-perfect',
    # 'Piano-perfect',
    'Pipes-perfect',
    'Playroom-perfect',
    'Playtable-perfect',
    'Recycle-perfect',
    'Shelves-perfect',
    # 'Shopvac-perfect', # 1110
    'Sticks-perfect',
    'Storage-perfect', # 650
    'Sword1-perfect',
    'Sword2-perfect',
    'Umbrella-perfect',
    'Vintage-perfect' # 740
 ]

BATCH_SIZE = 1
LR = 1e-3

MAX_DISP=370

train_dataset = MiddleburyDataset(os.path.expanduser('~/disk/middlebury'), l,
                            crop_height=256, crop_width=512)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = MVMEFNet(max_disp=MAX_DISP)
model = model.cuda()

save_num = -1
model_path = './savepoints/%d.pkl'%save_num
if os.path.exists(model_path):
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    print('loading dict')


L1loss = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(save_num+1, 10000):

    e_loss = 0
    count = 0
    e_PSNR = 0
    e_ME = 0
    for step, sample in enumerate(train_loader):
        left, right, left_g, right_g, left_o, right_o, right_gt, disp = \
        sample['left'], sample['right'], sample['left_g'], sample['right_g'], sample['left_o'], sample['right_o'], sample['right_gt'], sample['left_disparity']

        left, right, left_g, right_g, left_o, right_o, right_gt, disp = left.cuda(), right.cuda(), left_g.cuda(), right_g.cuda(), left_o.cuda(), right_o.cuda(), right_gt.cuda(), disp.cuda()
        
        # DISP MASK
        mask = (disp <= MAX_DISP) & (disp >= 0)

        # -----------------
        #  Train model
        # -----------------

        optimizer.zero_grad()

        # Generate a batch of images
        pred1, pred2, pred3, w_imgL_o, result, _ = model(left, right, left_g, right_g, left_o, right_o)

        # calculate PSNR
        PSNR = batch_PSNR(torch.clamp(result, 0., 1.), right_gt, 1.)

        # calculate disp ME
        ME = batch_me(pred3[mask], disp[mask])

        # loss aggregation
        g_loss = L1loss(result[:,:,:,:256], right_gt[:,:,:,:256])
        g_loss += L1loss(pred3[mask], disp[mask])
        g_loss += L1loss(w_imgL_o[:,:,:,:256], right_o[:,:,:,:256])

        if not np.isnan(ME):
            e_loss += g_loss.item()
            e_PSNR += PSNR
            e_ME += ME
            count+=1
            print("\r[Epoch %d][Loss: %7f][PSNR : %7f][ME : %7f]" % (epoch, g_loss.item(), PSNR, ME), end='')
        # else:
            # print(disp[mask])
        g_loss.backward()
        optimizer.step()
    e_loss = e_loss/count
    e_PSNR = e_PSNR/count
    e_ME = e_ME/count
    print("\r[Epoch %d][Loss: %7f][PSNR : %7f][ME : %7f]" % (epoch, e_loss, e_PSNR, e_ME), end='\n')
    with open('losses.txt', 'a') as f:
        f.write('[Loss: %7f]\n'%e_loss)
    if (epoch % 100 == 0):
        torch.save(model.state_dict(), './savepoints/%d.pkl' % (epoch))
