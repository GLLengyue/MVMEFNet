import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from .data_io import get_transform, read_all_lines, pfm_imread



class MiddleburyDataset_I(Dataset):
    def __init__(self, datapath, list_filenames, crop_width, crop_height):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.right_gt_filenames, self.left_disp_filenames = self.load_path(list_filenames)
        self.crop_width = crop_width
        self.crop_height = crop_height
        assert self.right_gt_filenames is not None

    def crop_image(self, image, n=6):
        resolution = 2**n
        width, height = image.size

        max_width = resolution * (width // resolution)
        max_height = resolution * (height // resolution)
        
        begin_width = (width - max_width) // 2
        begin_height = (height - max_height) // 2

        image = image.crop((begin_width, begin_height, max_width + begin_width, max_height + begin_height))
        return image
    
    def crop_disp(self, disp, n=6):
        # disp = disp.T
        resolution = 2**n
        height, width = disp.shape

        max_width = resolution * (width // resolution)
        max_height = resolution * (height // resolution)
        
        begin_width = (width - max_width) // 2
        begin_height = (height - max_height) // 2

        disp = disp[begin_height:begin_height+max_height, begin_width:begin_width+max_width]
        return disp


    def load_path(self, l):
        left_files = []
        right_files = []
        right_gt_files = []
        left_disp_files = []
        
        for cate in l:
            for i in range(1, 5): 
                df = os.path.join(self.datapath, cate, "disp0.pfm")
                tp = os.path.join(self.datapath, cate, "L%s"%i)
                if not os.path.isfile(os.path.join(tp, 'im0e0.png')):
                    print(os.path.join(tp, 'im0e0.png'))
                    continue
                try:
                    names = os.listdir(tp)
                    num = len(names)//2-1
                    left_i = 0
                    right_i = num-1

                    if num > 3:
                        r = num - 3
                        left_i = r//2
                        right_i = left_i + 2

                    left_files.append(os.path.join(tp, 'im0e%d.png'%right_i))
                    right_files.append(os.path.join(tp, 'im1e%d.png'%left_i))
                    right_gt_files.append(os.path.join(tp, 'im1e%d.png'%(left_i+1)))
                    left_disp_files.append(df)
                except:
                    continue
        return left_files, right_files, right_gt_files, left_disp_files

    def load_image(self, filename):
        return self.crop_image(Image.open(filename).convert('RGB'))

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        left_img = left_img.resize((left_img.size[0]//2, left_img.size[1]//2))
        right_img = right_img.resize((right_img.size[0]//2, right_img.size[1]//2))

        left_disp = pfm_imread(os.path.join(self.datapath, self.left_disp_filenames[index]))
        left_disp = self.crop_disp(left_disp[0])
        left_disp = cv2.resize(left_disp, None, fx=0.5, fy=0.5)
        left_disp = left_disp/2
        mask = (left_disp>0) & (left_disp<1000)

        right_gt_img = self.load_image(os.path.join(self.datapath, self.right_gt_filenames[index]))
        right_gt_img = right_gt_img.resize((right_gt_img.size[0]//2, right_gt_img.size[1]//2))
        
        
        w, h = left_img.size
        crop_w, crop_h = self.crop_width, self.crop_height

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

        # random crop
        left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        right_gt_img = right_gt_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        
        left_disp = left_disp[y1:y1+crop_h, x1:x1+crop_w]

        # to CV2
        left_img = np.asarray(left_img, dtype='float32')/255
        right_img = np.asarray(right_img, dtype='float32')/255
        right_gt_img = np.asarray(right_gt_img, dtype='float32')/255

        high_y = np.mean(cv2.cvtColor(left_img, cv2.COLOR_RGB2YUV)[:,:,0])
        low_y = np.mean(cv2.cvtColor(right_img, cv2.COLOR_RGB2YUV)[:,:,0])
        mid_y = np.mean(cv2.cvtColor(right_gt_img, cv2.COLOR_RGB2YUV)[:,:,0])

        # gamma
        left_img_g = np.power(left_img, high_y/mid_y)
        right_img_g = np.power(right_img, low_y/mid_y)

        # to tensor, normalize
        processed = get_transform()
        # left_orig = transforms.ToTensor()(left_img)
        # right_orig = transforms.ToTensor()(right_img)

        # left_img = processed(left_img_g)
        # right_img = processed(right_img_g)

        # left_img_g = transforms.ToTensor()(left_img_g)
        # right_img_g = transforms.ToTensor()(right_img_g)

        # right_gt_img = transforms.ToTensor()(right_gt_img)

        return {"left": processed(left_img_g),
                "right": processed(right_img_g),
                "left_g": transforms.ToTensor()(left_img_g),
                "right_g": transforms.ToTensor()(right_img_g),
                "left_o": transforms.ToTensor()(left_img),
                "right_o": transforms.ToTensor()(right_img),
                "right_gt": transforms.ToTensor()(right_gt_img),
                "left_disparity": left_disp.copy()
                }
