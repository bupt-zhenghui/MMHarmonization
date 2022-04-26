import torch

from options.test_options import TestOptions
from train_net import test
from util.misc import launch_job


def main():
    cfg = TestOptions().parse()  # get training options
    cfg.NUM_GPUS = torch.cuda.device_count()
    cfg.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    cfg.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    cfg.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    cfg.phase = 'test'
    cfg.batch_size = int(cfg.batch_size / max(1, cfg.NUM_GPUS))
    if not torch.cuda.is_available():
        cfg.dataset_root = "/Users/zhenghui/Downloads/Image_Harmonization_Dataset/HAdobe5k/"

    launch_job(cfg=cfg, init_method=cfg.init_method, func=test)


if __name__ == "__main__":
    main()
