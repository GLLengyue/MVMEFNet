import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from models import MVMEFNet
from utils import batch_PSNR, batch_me
from PIL import Image
import cv2

def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
torch.manual_seed(0)
torch.cuda.manual_seed(0)


model = MVMEFNet(max_disp=370)
model = model.cuda()

save_num = 1100
model_path = './savepoints/%d.pkl'%save_num
if os.path.exists(model_path):
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    print('loading dict')

model.eval()

left_img = Image.open('./imgL.png').convert('RGB')
right_img = Image.open('./imgR.png').convert('RGB')

left_img = np.asarray(left_img, dtype='float32')/255
right_img = np.asarray(right_img, dtype='float32')/255

high_y = np.mean(cv2.cvtColor(left_img, cv2.COLOR_RGB2YUV)[:,:,0])
low_y = np.mean(cv2.cvtColor(right_img, cv2.COLOR_RGB2YUV)[:,:,0])
mid_y = (high_y+low_y)/2

left_img_g = np.power(left_img, high_y/mid_y)
right_img_g = np.power(right_img, low_y/mid_y)

sample = {"left": processed(left_img_g),
"right": processed(right_img_g),
"left_g": transforms.ToTensor()(left_img_g),
"right_g": transforms.ToTensor()(right_img_g),
"left_o": transforms.ToTensor()(left_img),
"right_o": transforms.ToTensor()(right_img)
}

left, right, left_g, right_g, left_o, right_o = \
        sample['left'], sample['right'], sample['left_g'], sample['right_g'], sample['left_o'], sample['right_o']

left, right, left_g, right_g, left_o, right_o = left.cuda(), right.cuda(), left_g.cuda(), right_g.cuda(), left_o.cuda(), right_o.cuda()

with torch.no_grad():
    pred1, pred2, pred3, w_imgL_o, result, a_map = model(left, right, left_g, right_g, left_o, right_o)

PIL.Image.fromarray(np.uint8(w_imgL_o.cpu().numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_w_imgL_o.png'%step)
PIL.Image.fromarray(np.uint8(torch.clamp(result, 0., 1.).cpu().numpy()[0].transpose(1,2,0)*255)).save('./test_out/%d_result.png'%step)
PIL.Image.fromarray(np.uint8(pred3.cpu().numpy()[0])).save('./test_out/%d_pred.png'%step)
PIL.Image.fromarray(np.uint8(255*np.mean(a_map.cpu().numpy()[0], axis=0))).save('./test_out/%d_map.png'%step)
