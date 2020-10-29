import torch.nn as nn
import torch
import numpy as np

def target_indices(chosen_target_classes, num_classes, nz):
    return np.array([np.array(range(cls * nz, nz * (1 + cls))) for cls in chosen_target_classes])
class targeted_sampler(nn.Module):
    ''' The attacker model is given an image and outputs a perturbed version of that image.'''

    def __init__(self, num_perts, num_classes, imageSize):
        super(targeted_sampler, self).__init__()
        self.imageSize = imageSize
        self.num_perts = num_perts
        self.num_classes = num_classes
        self.gaussian = nn.Linear(self.num_perts*self.num_classes, 3 * self.imageSize * self.imageSize, bias=False)


    def forward(self, noise):
        x = self.gaussian(noise)
        x = x.view(-1, 3, self.imageSize, self.imageSize)
        x = nn.Tanh()(x)
        return x

    def get_noise(self, input_size, chosen_target_class =  None, indices_type = "OneClass"):
        noise_size = self.num_classes*self.num_perts
        if chosen_target_class == None:
            chosen_target_class = np.random.choice(self.num_classes)
        target_class_indices = np.arange(chosen_target_class*self.num_perts, self.num_perts*(1+chosen_target_class))
        indices = np.random.choice(target_class_indices, input_size)
        if indices_type =="MultiClass":
            chosen_target_classes = np.random.choice(self.num_classes, input_size)
            target_class_indices = target_indices(chosen_target_classes, self.num_classes, self.num_perts)
            indices = np.random.choice(target_class_indices.ravel(), input_size)
        noise = np.sqrt(noise_size)*torch.eye(noise_size, dtype = torch.float)[indices]
        targets = torch.LongTensor(input_size)
        if indices_type=='OneClass':
            targets.fill_(torch.tensor(chosen_target_class, dtype=torch.float)) 
        else:
            targets.data = torch.tensor(chosen_target_classes, dtype=torch.int)
        return (noise, targets)
