import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from dataloader import MiddleburyDataset
from models import PSMNet
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
    # 'Playroom-perfect',
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
CROP_HEIGHT = 256
CROP_WIDTH = 512

MAX_DISP=370

model = PSMNet(maxdisp=MAX_DISP)

# torch.distributed.init_process_group(backend="nccl")
model = model.cuda()
# model = nn.DataParallel(model)

train_dataset = MiddleburyDataset(os.path.expanduser('~/disk/middlebury'), l,
                            crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


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
    e_ME = 0
    for step, sample in enumerate(train_loader):
        left_o, right_o, disp = sample['left_o'], sample['right_o'], sample['left_disparity']

        left_o, right_o,disp = left_o.cuda(), right_o.cuda(), disp.cuda()

        # DISP MASK
        mask = (disp <= MAX_DISP) & (disp >= 0)

        if torch.sum(mask) < 128*1024/4:
            continue

        # -----------------
        #  Train model
        # -----------------

        optimizer.zero_grad()

        # Generate a batch of images
        # pred1, pred2, pred3, w_imgL_o, result, a_map, result_r = model(left, right, left_g, right_g, left_o, right_o)
        pred1, pred2, pred3 = model(left_o, right_o)

        # calculate disp ME
        ME = batch_me(pred3[mask], disp[mask])


        g_loss = L1loss(pred3[mask], disp[mask])


        print("\r[Epoch %d][Loss: %7f][ME : %7f]" % (epoch, g_loss.item(), ME), end='')

        g_loss.backward()
        optimizer.step()
    e_loss = e_loss/count
    e_ME = e_ME/count
    print("\r[Epoch %d][Loss: %7f][ME : %7f]" % (epoch, e_loss, e_ME), end='\n')
    # print("\r[Epoch %d][Loss: %7f][PSNR : %7f][PSNR_r : %7f][ME : %7f]" % (epoch, e_loss, e_PSNR, e_PSNR_r, e_ME), end='\n')
    with open('losses.txt', 'a') as f:
        f.write("\r[Epoch %d][Loss: %7f][ME : %7f]" % (epoch, e_loss,e_ME))
    if (epoch % 100 == 0):
        torch.save(model.state_dict(), './savepoints/%d.pkl' % (epoch))
