
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from score_functions import *
import pdb


class LO_FirstOrder(nn.Module):
    '''
    Detector module to implement a score-function based detector.
    Use some pgd method scheme to train the detector directly from the dataset.
    Returns the maximum of the
    '''
    def __init__(self, image_size, patch_size, num_classes, init_perts, type='simple', score_function=None, preprocess=data_luminance, **kwargs):
        '''
        image_size: the size of image as a tuple (channel, height, width). Currently uses only image luminance for detection.
        patch_size: the size of a patch in int. A square patch is assumed with both height and width equal to patch_size.
        '''
        super().__init__()
        self.image_channels = image_size[0]
        self.image_height = image_size[1]
        self.image_width = image_size[2]
        self.image_size = (image_size[1], image_size[2])
        self.patch_size = patch_size
        self.type = type
        self.score_type = kwargs['score_type']
        if score_function == None:
            self.score_function = Score_Gaussian(image_size, patch_size, init_loc='', preprocess=None)
            
        else:
            score_function.preprocess = None
            self.score_function = score_function
            

        num_x = image_size[1] * 1.0 / patch_size[0]
        num_y = image_size[2] * 1.0 / patch_size[1]
        num_patches = int(num_x * num_y)
        self.num_patches = num_patches
        self.preprocess = preprocess
        self.mean_perturbations = nn.Parameter(torch.randn((num_classes, num_patches, patch_size[0] * patch_size[1]), device='cuda'))
        # self.perturbation_layer = nn.Linear(image_size[1] * image_size[2], num_classes, bias=False)
        if init_perts:
            self.init_perturbation(init_perts)

    def init_perturbation(self, init_perts):
        '''
           Initializing the correlation layer using mean perturbation vectors.
        '''
        data = np.load(init_perts)
        self.mean_perturbations.data = torch.from_numpy(data['mean'])

    def batch_matmul(self, a, b):
        dim1 = max(a.shape[0], b.shape[0])
        dim2 = max(a.shape[1], b.shape[1])
        assert a.shape[2] == b.shape[2]
        dim3 = a.shape[2]
        res = torch.zeros((dim1, dim2, dim3, 1, 1), device=a.device, dtype=a.dtype)
        for i in range(dim1):
            for j in range(dim2):
                mat1 = a[min(i, a.shape[0] - 1), min(j, a.shape[1] - 1)]
                mat2 = b[min(i, b.shape[0] - 1), min(j, b.shape[1] - 1)]
                res[i, j] = torch.matmul(mat1, mat2)
        return res

    def forward(self, x):
        '''
          Computes the detection statistic. 
        '''
        with torch.no_grad():
            if self.preprocess != None:
                x = self.preprocess(x)
            else:
                assert len(x.shape) == 3
            if self.score_type in ['Gaussian', 'GMM']:
                x_patches = get_patches(x, self.image_size, self.patch_size)
                score_patches = self.score_function(x_patches)
            else:
                score_patches = self.score_function(x)
            score_patches = torch.reshape(score_patches, (-1, self.num_patches, self.patch_size[0] * self.patch_size[1]))

            mean_extend = self.mean_perturbations.unsqueeze(-2).unsqueeze(0)
            score_patches_extend0 = score_patches.unsqueeze(-1).unsqueeze(1)
            self.first_order = -1.0 * torch.matmul(mean_extend, score_patches_extend0)
            self.first_order = torch.squeeze(self.first_order).mean(dim=-1)  # Output is BXC
            self.target_scores = self.first_order

            # The following layer depends on the adversarial perturbation.
            if self.type == 'simple':
                return self.target_scores.mean(dim=-1)
            if self.type == 'composite':
                return self.target_scores.max(dim=-1)[0]


