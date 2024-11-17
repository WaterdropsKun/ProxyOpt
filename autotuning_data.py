import os
import glob
import shutil

from PIL import Image


def delete_file(file_path):
    if(os.path.isfile(file_path)):
    
        os.remove(file_path)
        
        #Printing the confirmation message of deletion
        print("File Deleted successfully", file_path)


def is_file(file_path):
    if(os.path.isfile(file_path)):
        pass
    else:
        print("File don't have", file_path)

        if "GT" in file_path:
            gt_file_name = file_path.replace("GT", "GT")
            noisy_file_name = file_path.replace("GT", "NOISY")
            red_file_name = file_path.replace("GT", "RED")
            param_file_name = file_path.replace("GT", "PARAM").replace("PNG", "txt")
        elif "NOISY" in file_path:
            gt_file_name = file_path.replace("NOISY", "GT")
            noisy_file_name = file_path.replace("NOISY", "NOISY")
            red_file_name = file_path.replace("NOISY", "RED")
            param_file_name = file_path.replace("NOISY", "PARAM").replace("PNG", "txt")
        elif "RED" in file_path:
            gt_file_name = file_path.replace("RED", "GT")
            noisy_file_name = file_path.replace("RED", "NOISY")
            red_file_name = file_path.replace("RED", "RED")
            param_file_name = file_path.replace("RED", "PARAM").replace("PNG", "txt")
        elif "PARAM" in file_path:
            gt_file_name = file_path.replace("PARAM", "GT").replace("txt", "PNG")
            noisy_file_name = file_path.replace("PARAM", "NOISY").replace("txt", "PNG")
            red_file_name = file_path.replace("PARAM", "RED").replace("txt", "PNG")
            param_file_name = file_path.replace("PARAM", "PARAM")
        
        delete_file(gt_file_name)
        delete_file(noisy_file_name)
        delete_file(red_file_name)
        delete_file(param_file_name)
        
        print("\n")


def traverse_folder_files(path):
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            # traverse_folder_files(file_path)  # 递归遍历子文件夹中的文件

            order = file_path.split('/')[-1].split('_')[0]

            old_file_GT_SRGB = os.path.join(file_path, "GT_SRGB_010.PNG")
            old_file_NOISY_SRGB = os.path.join(file_path, "NOISY_SRGB_010.PNG")

            new_folder_GT_SRGB = "./SIDD_crop/" + order + "_GT_SRGB"
            new_folder_NOISY_SRGB = "./SIDD_crop/" + order + "_NOISY_SRGB"
            new_file_GT_SRGB = os.path.join(new_folder_GT_SRGB, "GT_SRGB_010.PNG")
            new_file_NOISY_SRGB = os.path.join(new_folder_NOISY_SRGB, "NOISY_SRGB_010.PNG")

            file_dir = ['./SIDD_crop']
            for idx in range(len(file_dir)):
                if not os.path.exists(file_dir[idx]):
                    os.mkdir(file_dir[idx])

            if not os.path.exists(new_folder_GT_SRGB):
                os.mkdir(new_folder_GT_SRGB)
            if not os.path.exists(new_folder_NOISY_SRGB):
                os.mkdir(new_folder_NOISY_SRGB)

            shutil.copy(old_file_GT_SRGB, new_file_GT_SRGB)
            shutil.copy(old_file_NOISY_SRGB, new_file_NOISY_SRGB)

        else:
            print(file_path)  # 处理文件


def check_data(img_path, is_train=True):

    if is_train == True:
        img_path = '{}/train'.format(img_path)
    elif is_train == False:
        img_path = '{}/test'.format(img_path)

    # GT
    gt_file_name = '{}/GT/*'.format(img_path)
    gt_file = glob.glob(gt_file_name)
    gt_file.sort()
    gt_img = []
    for idx in range(len(gt_file)):
        tmp = glob.glob('{}/*.PNG'.format(gt_file[idx]))
        tmp.sort()
        gt_img.extend(tmp.copy())

    # NOISY
    noisy_file_name = '{}/NOISY/*'.format(img_path)
    noisy_file = glob.glob(noisy_file_name)
    noisy_file.sort()
    noisy_img = []
    for idx in range(len(noisy_file)):
        tmp = glob.glob('{}/*.PNG'.format(noisy_file[idx]))
        tmp.sort()
        noisy_img.extend(tmp.copy())
    
    # RED
    red_file_name = '{}/RED/*'.format(img_path)
    red_file = glob.glob(red_file_name)
    red_file.sort()
    red_img = []
    for idx in range(len(red_file)):
        tmp = glob.glob('{}/*.PNG'.format(red_file[idx]))
        tmp.sort()
        red_img.extend(tmp.copy())

    # PARAM
    param_file_name = '{}/PARAM/*'.format(img_path)
    param_file = glob.glob(param_file_name)
    param_file.sort()
    param = []
    for idx in range(len(param_file)):
        tmp = glob.glob('{}/*.txt'.format(param_file[idx]))
        tmp.sort()
        param.extend(tmp.copy())


    for image_file in gt_img:
        try:
            image = Image.open(image_file)
            x, y, w, h = 0, 0, 512, 512
            image = image.crop([x, y, x+w, y+h])
            
            gt_file_name = image_file.replace("GT", "GT")
            noisy_file_name = image_file.replace("GT", "NOISY")
            red_file_name = image_file.replace("GT", "RED")
            param_file_name = image_file.replace("GT", "PARAM").replace("PNG", "txt")
            
            is_file(gt_file_name)
            is_file(noisy_file_name)
            is_file(red_file_name)
            is_file(param_file_name)
        except Exception as e:
            print("[DebugMK] error", image_file)
            print(e)

            gt_file_name = image_file.replace("GT", "GT")
            noisy_file_name = image_file.replace("GT", "NOISY")
            red_file_name = image_file.replace("GT", "RED")
            param_file_name = image_file.replace("GT", "PARAM").replace("PNG", "txt")
            
            delete_file(gt_file_name)
            delete_file(noisy_file_name)
            delete_file(red_file_name)
            delete_file(param_file_name)
            

    for image_file in noisy_img:
        try:
            image = Image.open(image_file)
            x, y, w, h = 0, 0, 512, 512
            image = image.crop([x, y, x+w, y+h])
            
            gt_file_name = image_file.replace("NOISY", "GT")
            noisy_file_name = image_file.replace("NOISY", "NOISY")
            red_file_name = image_file.replace("NOISY", "RED")
            param_file_name = image_file.replace("NOISY", "PARAM").replace("PNG", "txt")
            
            is_file(gt_file_name)
            is_file(noisy_file_name)
            is_file(red_file_name)
            is_file(param_file_name)
        except Exception as e:
            print("[DebugMK] error", image_file)
            print(e)
            
            gt_file_name = image_file.replace("NOISY", "GT")
            noisy_file_name = image_file.replace("NOISY", "NOISY")
            red_file_name = image_file.replace("NOISY", "RED")
            param_file_name = image_file.replace("NOISY", "PARAM").replace("PNG", "txt")
            
            delete_file(gt_file_name)
            delete_file(noisy_file_name)
            delete_file(red_file_name)
            delete_file(param_file_name)


    for image_file in red_img:
        try:
            image = Image.open(image_file)
            x, y, w, h = 0, 0, 512, 512
            image = image.crop([x, y, x+w, y+h])
            
            gt_file_name = image_file.replace("RED", "GT")
            noisy_file_name = image_file.replace("RED", "NOISY")
            red_file_name = image_file.replace("RED", "RED")
            param_file_name = image_file.replace("RED", "PARAM").replace("PNG", "txt")
            
            is_file(gt_file_name)
            is_file(noisy_file_name)
            is_file(red_file_name)
            is_file(param_file_name)
        except Exception as e:
            print("[DebugMK] error", image_file)
            print(e)
            
            gt_file_name = image_file.replace("RED", "GT")
            noisy_file_name = image_file.replace("RED", "NOISY")
            red_file_name = image_file.replace("RED", "RED")
            param_file_name = image_file.replace("RED", "PARAM").replace("PNG", "txt")
            
            delete_file(gt_file_name)
            delete_file(noisy_file_name)
            delete_file(red_file_name)
            delete_file(param_file_name)


    for param_file in param:
        try:
            lines_ = []
            param_ = []

            with open(param_file, 'r') as f:
                line = f.readline()
                while line:
                    lines_.append(line)
                    line = f.readline()
            f.close()
            
            gt_file_name = param_file.replace("PARAM", "GT").replace("txt", "PNG")
            noisy_file_name = param_file.replace("PARAM", "NOISY").replace("txt", "PNG")
            red_file_name = param_file.replace("PARAM", "RED").replace("txt", "PNG")
            param_file_name = param_file.replace("PARAM", "PARAM")
            
            is_file(gt_file_name)
            is_file(noisy_file_name)
            is_file(red_file_name)
            is_file(param_file_name)
        except Exception as e:
            print("[DebugMK] error", param_file)
            print(e)
            
            gt_file_name = param_file.replace("PARAM", "GT").replace("txt", "PNG")
            noisy_file_name = param_file.replace("PARAM", "NOISY").replace("txt", "PNG")
            red_file_name = param_file.replace("PARAM", "RED").replace("txt", "PNG")
            param_file_name = param_file.replace("PARAM", "PARAM")
            
            delete_file(gt_file_name)
            delete_file(noisy_file_name)
            delete_file(red_file_name)
            delete_file(param_file_name)
            

from model import U_Net
import numpy as np
import torch
from tools.dataloader import train_dataloader, test_dataloader, val_dataloader
from tqdm import tqdm
import visdom
from matplotlib import pyplot as plt
from PIL import Image
from tools.utils import get_psnr
import pickle
import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

random.seed(1)

def train_step1():
    MAX_EPOCH = 1  # DebugMK
    LR = 0.003

    # viz = visdom.Visdom(env='step1')  # DebugMK
    image_dir = './SIDD_crop_bm3d'
    train_loader = train_dataloader(image_dir, batch_size=5, num_threads=0, img_size=512)
    test_loader = test_dataloader(image_dir, batch_size=1, num_threads=0, img_size=512)
    train_loader2 = train_dataloader(image_dir, batch_size=1, num_threads=0, img_size=512)
    net = U_Net(3, 3, step_flag=1, img_size=512)
    net = torch.nn.DataParallel(net).cuda()

    # net.load_state_dict(torch.load('./checkpoints_step1_20240101/net_iter265.pth'))  # DebugMK

    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), LR)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50, 75, 100, 125, 150, 175, 200], 0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)

    maxx = -1

    # if not os.path.exists('./checkpoints_step1'):  # DebugMK
    #     os.mkdir('./checkpoints_step1')

    for epoch in range(MAX_EPOCH):
        train_loss = 0.0
        net.train()

        try:
            for _, noisy, red, param in tqdm(train_loader):
                pass
                
        except Exception as e:
            print(e)



if __name__ == "__main__":
    # 1
    traverse_folder_files("./data/SIDD_Small_sRGB_Only/Data/")

    # 3
    # check_data("./SIDD_crop_bm3d", True)
    # check_data("./SIDD_crop_bm3d", False)
    # check_data("./SIDD_crop_net", True)
    # check_data("./SIDD_crop_net", False)

    # 4（可以不管）
    # train_step1()
    pass
