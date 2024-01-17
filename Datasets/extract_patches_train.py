import numpy as np
from PIL import Image
import os


if __name__=="__main__":
    path = "./DIV2K_train_HR"
    output_path = "./DIV2K_train_p128"
    files = os.listdir(path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in files:
        file_path = os.path.join(path, file)
        print(file_path)

        I = Image.open(file_path)
        w, h = I.size
        print(h, w)

        patch_size=128

        im = np.array(I)
        for i in range(0,h,patch_size):
            for j in range(0,w,patch_size):
                if i+patch_size <= h and j+patch_size <= w:
                    im_p = im[i:i+patch_size, j:j+patch_size, :]
                    I_p = Image.fromarray(im_p)
                    I_p.save(os.path.join(output_path,file[:-4]+str(i//patch_size)+str(j//patch_size)+".png"), compress_level= 0)