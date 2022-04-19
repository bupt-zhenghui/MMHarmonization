import os
import torch
import json
import torch.distributed as dist
from util import distributed as du
from util import html, util
from util.evaluation import evaluation
from util.visualizer import Visualizer
from util.visualizer import save_images
from models import create_model

from data import create_dataset
from util.misc import launch_job
from train_net import train
import caption_dataset
from d2l import torch as d2l

from options.train_options import TrainOptions


def main():
    cfg = TrainOptions().parse()  # get training options
    cfg.dataset_mode = 'mmihd'
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.batch_size = int(cfg.batch_size / max(1, cfg.NUM_GPUS))
    if not torch.cuda.is_available():
        cfg.dataset_root = "/Users/zhenghui/Downloads/Image_Harmonization_Dataset/HAdobe5k/"

    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join('../data/bert.small.torch/',
                                                     'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    cfg.vocab = vocab

    launch_job(cfg=cfg, init_method=cfg.init_method, func=train)

    # dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    # # it = next(iter(dataset))
    # # print(len(it['tokens']))
    # # print(it['tokens'].shape)
    #
    # postion_embedding = util.PositionEmbeddingSine(cfg)
    # patch_pos = util.PatchPositionEmbeddingSine(cfg)
    # model = create_model(cfg)  # create a model given cfg.model and other options
    # model.set_position(postion_embedding, patch_pos=patch_pos)
    #
    # for idx, data in enumerate(dataset):
    #     model.set_input(data)
    #     model.optimize_parameters()
    #     break


if __name__ == "__main__":
    main()
