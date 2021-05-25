import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import cv2
import PIL

torch.manual_seed(0)
torch.cuda.manual_seed(0)

l = [
    # 'Adirondack-perfect',
    # 'Backpack-perfect',
    'Bicycle1-perfect',
    # 'Cable-perfect',
    'Classroom1-perfect',
    # 'Couch-perfect',
    # 'Flowers-perfect',
    # 'Jadeplant-perfect',
    # 'Mask-perfect',
    # 'Motorcycle-perfect',
    'Piano-perfect',
    # 'Pipes-perfect',
    # 'Playroom-perfect',
    # 'Playtable-perfect',
    # 'Recycle-perfect',
    # 'Shelves-perfect',
    # 'Shopvac-perfect',
]

BATCH_SIZE = 1
MAX_DISP=370

test_dataset = MiddleburyDataset(os.path.expanduser("~/disk/middlebury"), l,
                            crop_height=512, crop_width=768)

# test_dataset = MiddleburyDataset_I(os.path.expanduser("~/disk/middlebury"), l,
#                             crop_height=768, crop_width=1408)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

model = MVMEFNet(max_disp=MAX_DISP)
model = model.cuda()

save_num = 400
model_path = './savepoints/%d.pkl'%save_num
if os.path.exists(model_path):
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    print('loading dict')

model.eval()

for step, sample in enumerate(test_loader):
    left, right, left_g, right_g, left_o, right_o, right_gt, disp = \
        sample['left'], sample['right'], sample['left_g'], sample['right_g'], sample['left_o'], sample['right_o'], sample['right_gt'], sample['left_disparity']

    PIL.Image.fromarray(np.uint8(left_o.numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_left_o.png'%step)
    PIL.Image.fromarray(np.uint8(right_o.numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_right_o.png'%step)
    PIL.Image.fromarray(np.uint8(right_gt.numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_right_gt.png'%step)
    PIL.Image.fromarray(np.uint8(disp.numpy()[0])).save('./test_out/%d_disp.png'%step)

    left, right, left_g, right_g, left_o, right_o, right_gt, disp = left.cuda(), right.cuda(), left_g.cuda(), right_g.cuda(), left_o.cuda(), right_o.cuda(), right_gt.cuda(), disp.cuda()

    with torch.no_grad():
        pred1, pred2, pred3, w_imgL_o, result, a_map = model(left, right, left_g, right_g, left_o, right_o)
    

    PIL.Image.fromarray(np.uint8(w_imgL_o.cpu().numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_w_imgL_o.png'%step)
    PIL.Image.fromarray(np.uint8(torch.clamp(result, 0., 1.).cpu().numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_result.png'%step)
    PIL.Image.fromarray(np.uint8(pred3.cpu().numpy()[0])).save('./test_out/%d_pred.png'%step)
    PIL.Image.fromarray(np.uint8(255*np.mean(a_map.cpu().numpy()[0], axis=0))).save('./test_out/%d_map.png'%step)

    mask = (disp < MAX_DISP) & (disp > 0)

    PSNR = batch_PSNR(torch.clamp(result, 0., 1.), right_gt, 1.)
    ME = batch_me(pred3[mask], disp[mask])

    print("[step %d][PSNR : %7f][ME : %7f]" % (step, PSNR, ME))

    