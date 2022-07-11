import torch
import torch.nn.functional as F

from util import util
from . import harmony_networks as networks
from .base_model import BaseModel


class mmhtModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(norm='instance', netG='MMHT', dataset_mode='ihd')
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['mask', 'harmonized', 'comp', 'real', 'reflectance', 'illumination']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)

        print(self.netG)

        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            clip_params = list(map(id, self.netG.clip_model.module.parameters()))
            base_params = filter(lambda p: id(p) not in clip_params, self.netG.module.parameters())
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam([
                {'params': base_params},
                {'params': self.netG.clip_model.module.parameters(), 'lr': opt.lr * 0.00001},
            ], lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_position(self, pos, patch_pos=None):
        b = self.opt.batch_size
        self.pixel_pos = pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.pixel_pos = self.pixel_pos.flatten(2).permute(2, 0, 1)
        self.patch_pos = patch_pos.unsqueeze(0).repeat(b, 1, 1, 1).to(self.device)
        self.patch_pos = self.patch_pos.flatten(2).permute(2, 0, 1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']
        self.mask_r = F.interpolate(self.mask, size=[64, 64])
        self.revert_mask = 1 - self.mask
        self.clip_image = input['clip_image'].to(self.device)
        self.mask_embedding = input['mask_embedding'].to(self.device)

    def data_dependent_initialize(self, data):
        pass

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print(self.pixel_pos.detach().shape)
        # print(self.patch_pos.detach())
        # print(self.patch_pos.detach().shape)
        # print('mask_r shape: ', self.mask_r.shape)
        # print('mask shape: ', self.mask.shape)
        self.harmonized, self.reflectance, self.illumination = self.netG(inputs=self.inputs,
                                                                         image=self.comp * self.revert_mask,
                                                                         pixel_pos=self.pixel_pos.detach(),
                                                                         patch_pos=self.patch_pos.detach(),
                                                                         mask_r=self.mask_r, mask=self.mask,
                                                                         clip_image=self.clip_image,
                                                                         mask_embedding=self.mask_embedding)
        if not self.isTrain:
            self.harmonized = self.comp * (1 - self.mask) + self.harmonized * self.mask

    def compute_G_loss(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt, net, distance_type='dot'):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(inputs=tgt, layers=self.nce_layers, encode_only=True)

        feat_k = self.netG(inputs=src, layers=self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = net(feat_k, self.opt.num_patches, None, fg_mask=self.mask)
        feat_q_pool, _ = net(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k, distance_type=distance_type)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
