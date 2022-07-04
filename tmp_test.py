from PIL import Image
import torch
import torchvision.transforms.functional as tf
import clip


def PatchMaskEmbedding(patch_size=32, img_size=224):
    tmp_mask_path = '/Users/zhenghui/Downloads/Image_Harmonization_Dataset/HAdobe5k/masks/a0005_1.png'
    mask = Image.open(tmp_mask_path).convert('1')
    mask = tf.to_tensor(mask)
    mask = torch.squeeze(tf.resize(mask, [img_size, img_size]))

    res_list = []
    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            cur_cnt = 0
            for ii in range(patch_size):
                for jj in range(patch_size):
                    if mask[i + ii, j + jj] == 1: cur_cnt += 1
            res_list.append(1 - cur_cnt / (patch_size * patch_size))
    res_list.insert(0, sum(res_list) / len(res_list))
    patch_mask_tensor = torch.tensor(res_list)
    print(torch.unsqueeze(patch_mask_tensor, 1).shape)
    print(patch_mask_tensor)
    return patch_mask_tensor


def clip_test():
    img_path = '/Users/zhenghui/Downloads/Image_Harmonization_Dataset/HAdobe5k/composite_images/a0002_1_1.jpg'
    mask_path = '/Users/zhenghui/Downloads/Image_Harmonization_Dataset/HAdobe5k/masks/a0005_1.png'
    clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
    # print(clip_model)
    total = sum([param.nelement() for param in clip_model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    # print(clip_image.shape)
    #
    # image_features = torch.squeeze(clip_model.encode_image(clip_image))
    # print(image_features.shape)

    # mask_image = preprocess(Image.open(mask_path).convert('1'))
    # print(mask_image.shape)


if __name__ == '__main__':
    clip_test()
    # PatchMaskEmbedding()
