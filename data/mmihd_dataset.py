import os.path

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
from PIL import Image

import clip
from data.base_dataset import BaseDataset
from util.util import PatchMaskEmbedding


class MMIhdDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
            :param is_train:
            :param parser:
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"),
                            new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device='cpu')

        if opt.isTrain:
            # self.real_ext='.jpg'
            print('loading training file')
            self.trainfile = opt.dataset_root + opt.dataset_name + '_train.txt'
            with open(self.trainfile, 'r') as f:
                for line in f.readlines():
                    self.image_paths.append(os.path.join(opt.dataset_root, 'composite_images', line.rstrip()))
        elif not opt.isTrain:
            # self.real_ext='.jpg'
            print('loading test file')
            self.trainfile = opt.dataset_root + opt.dataset_name + '_test.txt'
            with open(self.trainfile, 'r') as f:
                for line in f.readlines():
                    self.image_paths.append(os.path.join(opt.dataset_root, 'composite_images', line.rstrip()))
                    # print(line.rstrip())
        # get the image paths of your dataset; You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to
        # get all the image paths under the directory self.root define the default transform function. You can use
        # <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        name_parts = path.split('_')
        mask_path = self.image_paths[index].replace('composite_images', 'masks')
        mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')
        target_path = self.image_paths[index].replace('composite_images', 'real_images')
        target_path = target_path.replace(('_' + name_parts[-2] + '_' + name_parts[-1]), '.jpg')
        img_name = target_path.split('/')[-1]

        # comp = util.retry_load_images(path)
        # mask = util.retry_load_images(mask_path)
        # real = util.retry_load_images(target_path)

        comp = Image.open(path).convert('RGB')
        real = Image.open(target_path).convert('RGB')
        mask = Image.open(mask_path).convert('1')

        clip_image = self.preprocess(Image.open(path)).unsqueeze(0)
        image_features = torch.squeeze(self.clip_model.encode_image(clip_image))
        generated = torch.autograd.Variable(image_features, requires_grad=False)
        image_features = generated.data

        # tokens = clip.tokenize([self.img_caption_dic[img_name], self.img_reg_dic[img_name]])
        # text_features = self.text_model.encode_text(tokens).reshape(-1, 256)
        # generated = torch.autograd.Variable(text_features, requires_grad=False)
        # text_features = generated.data

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)

        if comp.size[0] != self.image_size:
            # assert 0
            comp = tf.resize(comp, [self.image_size, self.image_size])
            mask = tf.resize(mask, [self.image_size, self.image_size])
            real = tf.resize(real, [self.image_size, self.image_size])

        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        # mask = 1-mask
        real = self.transforms(real)

        # Add patch mask embedding
        mask_embedding = PatchMaskEmbedding(mask)
        image_features = mask_embedding * image_features

        # comp = real
        # mask = torch.zeros_like(mask)
        # inputs=torch.cat([real,mask],0)
        inputs = torch.cat([comp, mask], 0)

        return {'inputs': inputs, 'comp': comp, 'real': real,
                'img_path': path, 'mask': mask,
                'img_feat': image_features}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
