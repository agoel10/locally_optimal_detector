import lo_glrt
from prn import *
import torch
from score_functions import Score_Gaussian


def get_detectors(detector_arch, checkpoint_detector=None, **kwargs):
    if detector_arch.startswith('gaussian'):
        detector_arch = detector_arch.split('_')
        if detector_arch[1] == 'firstorder':
            score_function = Score_Gaussian(
                image_size=(3, kwargs['image_size'], kwargs['image_size']),
                patch_size=(kwargs['patch_size'], kwargs['patch_size']),
                init_loc=kwargs['location_generative_model'],
                train=False,
                input_patches=True,
                output_patches=True,
                preprocess=None
            )
            if detector_arch[2] == 'simple':
                detector = lo_glrt.LO_FirstOrder((3, kwargs['image_size'], kwargs['image_size']), (kwargs['patch_size'], kwargs['patch_size']), kwargs['num_classes'],
                                                 init_perts=kwargs['location_perts_means'] + '_patchsize_{}.npz'.format(kwargs['patch_size']),
                                                 type='simple', score_type = kwargs['score_type'], score_function=score_function)
            if detector_arch[2] == 'composite':
                detector = lo_glrt.LO_FirstOrder((3, kwargs['image_size'], kwargs['image_size']), (kwargs['patch_size'], kwargs['patch_size']), kwargs['num_classes'],
                                                 init_perts=kwargs['location_perts_means'] + '_patchsize_{}.npz'.format(kwargs['patch_size']),
                                                 type='composite', score_type = kwargs['score_type'], score_function=score_function)

        if checkpoint_detector:
            checkpoint_detector = torch.load(checkpoint_detector)
            detector.load_state_dict(checkpoint_detector['state_dict'])


    elif detector_arch.startswith('prn'):
        detector = PRN_detector(kwargs['image_size'])
        checkpoint_detector = torch.load(checkpoint_detector)
        detector.load_state_dict(checkpoint_detector)


    return detector
