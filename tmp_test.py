from PIL import Image
import torch
import torchvision.transforms.functional as tf


def PatchMaskEmbedding(patch_size=8):
    tmp_mask_path = '/Users/zhenghui/Downloads/Image_Harmonization_Dataset/HAdobe5k/masks/a0005_1.png'
    mask = Image.open(tmp_mask_path).convert('1')
    mask = tf.to_tensor(mask)
    mask = torch.squeeze(tf.resize(mask, [256, 256]))
    print(mask.shape)
    # print(mask)
    res_list = []
    for i in range(0, 256, patch_size):
        cur_list = []
        for j in range(0, 256, patch_size):
            flag = False
            for ii in range(patch_size):
                if flag: break
                for jj in range(patch_size):
                    if mask[i + ii, j + jj] == 1:
                        flag = True
                        break
            cur_list.append(1 if flag else 0)
        res_list.append(cur_list)
    patch_mask_tensor = torch.tensor(res_list)
    print(patch_mask_tensor.shape)


PatchMaskEmbedding()
