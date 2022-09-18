import glob

from data.base_dataset import get_params, get_transform
from PIL import Image
from scipy.io import savemat
import os
from options.test_options import TestOptions
import torchvision.transforms as transforms
import numpy as np
from util import util

dst_dir = "D:/new_train/blur2edge80/real_b/mat"

if __name__ == '__main__':
    opt = TestOptions().parse()
    image_list = glob.glob("D:/new_train/blur2edge80/real_b/nms/*.png")
    tmp = Image.open(image_list[0])
    # transform_params = get_params(opt, tmp.size)
    # B_transform = get_transform(opt, transform_params, grayscale=(opt.output_nc == 1))

    for image_path in image_list:
        B = np.array(Image.open(image_path))
        # Min, Max = np.min(B), np.max(B)
        # B = (B - Min) / (Max - Min)
        # B = B_transform(Image.open(image_path))
        # im = util.tensor2im(B)
        # util.save_image(np.asarray(B), os.path.join(dst_dir, os.path.split(image_path)[1]))
        # B = np.asarray(B)[0]
        tgt = np.full((opt.crop_size, opt.crop_size), True)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                tgt[i, j] = (B[i, j] >= 100)
        print(image_path)
        dst_file = os.path.join(dst_dir, os.path.split(image_path)[1][:-4] + ".mat")
        savemat(dst_file, {'groundTruth': [{'Boundaries': tgt}]})




