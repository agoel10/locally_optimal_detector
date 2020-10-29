import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils import *
'''
Save the covaraices, precision matrices and mean vectors in a npy file (dictionary) to load them for training.
'''


class Score_Gaussian(nn.Module):
    '''
    Use this class to compute the score of the model with respect to a Gaussian distribution.
    '''

    def __init__(self, image_size, patch_size, init_loc='', train=False, preprocess=data_luminance, input_patches=False, output_patches=False, **kwargs):
        '''
         image_size: the size of image as a tuple (channel, height, width). Currently uses only image luminance for detection.
         patch_size: the size of a patch as a tuple (height, width). A square patch is assumed with both height and width equal to patch_size.

        '''
        super().__init__()
        self.image_channels, self.image_height, self.image_width = image_size[0], image_size[1], image_size[2]
        self.patch_size = patch_size
        self.preprocess = preprocess
        self.nz = 100
        norm_constant = 1.0 / np.sqrt(patch_size[0] * patch_size[1])
        # self.gen = Generator(self.nz, patch_size)
        # self.gen.to('cuda')
        self.L = nn.Parameter(norm_constant * torch.randn(patch_size[0] * patch_size[1], patch_size[0] * patch_size[1], device='cuda'))
        self.mean = nn.Parameter(torch.randn(patch_size[0] * patch_size[1], device='cuda'))
        self.init_loc = init_loc
        if self.init_loc:
            self.mean_list, self.precision_list, _ = load_gmm_model(init_loc)
            self.mean.data = self.mean_list[0].data
            self.L.data = torch.cholesky(self.precision_list[0].data)

        self.input_patches = input_patches
        self.output_patches = output_patches
        self.train_model = train

    def init_map(self):
        # noise = torch.randn((1,self.nz), device = 'cuda')
        # self.L = self.gen(noise)
        self.precision = torch.mm(self.L, self.L.t()) + 0.00001 * torch.eye(self.L.shape[0], device='cuda')
        if self.init_loc and (not self.train_model):
            self.mean.data = self.mean_list[0].data
            self.precision.data = self.precision_list[0]

        self.score_map_weight = -1 * self.precision
        self.score_map_bias = torch.mv(self.precision, self.mean)

    def forward(self, x):
        '''
        Accepts the input image as a tensor of shape - (B, C, H, W) or (B,H,W) or (B*N, P)
        Returns the score function of an image assuming Gaussian distribution for input.
        '''
        self.init_map()
        if self.preprocess != None:
            x = self.preprocess(x)
        # preprocess will remove the channel dimension from the image.
        else:
            assert len(x.shape) == 3 or len(x.shape) == 2

        # break the input into patches
        image_size = (self.image_height, self.image_width)

        if not self.input_patches:
            if len(x.shape) == 2:
                x = torch.unsqueeze(x, 0)
            x_patches = get_patches(x, image_size, self.patch_size)
        else:
            x_patches = x
        score_patches = F.linear(x_patches, self.score_map_weight, self.score_map_bias)
        # piece the patches together
        if not self.output_patches:
            self.score = combine_patches(score_patches, image_size, self.patch_size)
        else:
            self.score = score_patches

        return self.score
