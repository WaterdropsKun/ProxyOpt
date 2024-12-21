import os
import glob
import shutil

from PIL import Image

import SIDD_crop_bm3d
import SIDD_crop_net


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


def generate_SIDD_crop(path_src, path_SIDD_crop):
    files = os.listdir(path_src)
    for file in files:
        file_path = os.path.join(path_src, file)
        if os.path.isdir(file_path):
            order = file_path.split('/')[-1].split('_')[0]

            old_file_GT_SRGB = os.path.join(file_path, "GT_SRGB_010.PNG")
            old_file_NOISY_SRGB = os.path.join(file_path, "NOISY_SRGB_010.PNG")

            new_folder_GT_SRGB = path_SIDD_crop + "/" + order + "_GT_SRGB"
            new_folder_NOISY_SRGB = path_SIDD_crop + "/" + order + "_NOISY_SRGB"
            new_file_GT_SRGB = os.path.join(new_folder_GT_SRGB, "GT_SRGB_010.PNG")
            new_file_NOISY_SRGB = os.path.join(new_folder_NOISY_SRGB, "NOISY_SRGB_010.PNG")

            if not os.path.exists(path_SIDD_crop):
                os.mkdir(path_SIDD_crop)

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

    # 打印训练集、测试集数量
    if is_train == True:
        print("train data num = {}".format(len(gt_img)))
    else:
        print("test data num = {}".format(len(gt_img)))

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


if __name__ == "__main__":
    # 1
    generate_SIDD_crop("./data/SIDD_Small_sRGB_Only/Data/", "./SIDD_crop/")

    # 2
    # 可扩充数据：修改日期, 原始数据路径里面每张图随机crop生成数据的组数
    SIDD_crop_bm3d.process("./SIDD_crop/", "20241221", 1)  # 原始数据路径, 日期, 原始数据路径里面每张图随机crop生成数据的组数
    SIDD_crop_net.process("./SIDD_crop/", "20241221", 1)  # 原始数据路径, 日期, 原始数据路径里面每张图随机crop生成数据的组数

    # 3
    # 打印当前路径训练集、测试集数量
    check_data("./SIDD_crop_bm3d", True)
    check_data("./SIDD_crop_bm3d", False)
    check_data("./SIDD_crop_net", True)
    check_data("./SIDD_crop_net", False)

    pass
