from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
from utils import *
import torch.optim as optim
import pickle
from sklearn import mixture
from torch.autograd import Variable
import scipy
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from score_functions import Score_Gaussian
import attack_model
import lo_glrt
import torchvision.models as tv_models
from prn import *
from get_detectors import get_detectors


parser = argparse.ArgumentParser(description='PyTorch Score-Function implementation')
# Datasets
parser.add_argument('-d', '--dataset', default='ImageNet', type=str)
parser.add_argument('--location_dataset', default='data/ImageNet', type=str, metavar='PATH',
                    help='path to the datasets')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes')
parser.add_argument('--image_size', type=int, default=224, help='Image size')

parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# attacker
parser.add_argument('--attacker_type', type = str, default = 'targeted-sampler', help = 'Type of the attacker to use') # use tiny-imagenet for Tiny Imagenet dataset
parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector per target class')
parser.add_argument('--num_perts', type=int, default=2, help='Number of perturbations per-class')
parser.add_argument('--targeted', type = int, default = 1, help = 'if the attack is targeted (default False)')
parser.add_argument('--synthetic', type = int, default = 0, help = 'if the synthetic perturbations allowed (default False)')
parser.add_argument('--location_attacker', default='models/attacker/checkpoint/ImageNet/attacker_targeted-sampler_ImageNet_resnet18.pth.tar')

# detector
parser.add_argument('--detector_arch', '-da', metavar='ARCH', default='gaussian_firstorder_composite_supervised',
                    help='model architecture: '
                    + ' (default: gaussian)')
parser.add_argument('--patch_size', type=int, default=8, help='Patch size')
parser.add_argument('--num_components', type=int, default=1, help='Components of GMM')
parser.add_argument('--score_type', type=str, default = 'Gaussian', help='Type of Score function to use in the detector')

# Use null string for checkpoint to evaluate the unsupervised form detector. 
parser.add_argument('--checkpoint_detector', default='models/detector/prn/checkpoint/ImageNet/PRN_detector_ImageNet_googlenet_targeted-sampler.pth.tar', help="")

parser.add_argument('--location_generative_model', default='models/generative_models/dataset_ImageNet_patch_8_components_1_ML_luminance.p', type=str, metavar='PATH',
                    help='path to the generative model trained using ML/EM algorithm in sci-kit learn (default: none)')
parser.add_argument('--location_perts_means', default='models/detector/lo_glrt/mean_perturbations_targeted-sampler/mean_ImageNet_resnet18', type=str, metavar='PATH',
                    help='path to mean of the perturbations')
parser.add_argument('--save_attack_statistics', type = int, default = 0, help = 'save the mean perturbation vectors for each target class')


# classifier
parser.add_argument('--classifier_arch', '-ca', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')


# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--cuda', action = 'store_true', default = True, help = 'enables cuda')


args = parser.parse_args()
# outfile_name = 'Results/%s/GMM3_unsupervised_patch_%d_classifier_%s'%(args.dataset, args.patch_size, args.classifier_arch)
outfile_name = 'Results/%s/detector_%s_patch_%s_classifier_%s_attacker_%s' % (args.dataset, args.detector_arch, args.patch_size, args.classifier_arch, args.attacker_type)
args.train = True


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


if args.dataset == 'CIFAR10':
    min_val = -8
    max_val = 8
    multiplier = 10
if args.dataset == 'CIFAR100':
    min_val = -8
    max_val = 8
    multiplier = 10

if args.dataset == 'ImageNet':
    min_val = -10
    max_val = 10
    multiplier = 15


def target_indices(chosen_target_classes, num_classes, nz):
    return np.array([np.array(range(cls * nz, nz * (1 + cls))) for cls in chosen_target_classes])


def preprocess_classifier(x, dataset, type='input'):
    if dataset == 'ImageNet':
        if type == 'input':
            x = x * 255.0
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        y = torch.zeros(x.shape, device='cuda', dtype=torch.float32)
        if type == 'input':
            for ch in range(3):
                y[:, ch, :, :] = (x[:, ch, :, :] * 1.0 / 255 - mean[ch]) / std[ch]
        if type == 'perturbation':
            for ch in range(3):
                y[:, ch, :, :] = (x[:, ch, :, :] * (1.0 / 255)) / std[ch]
        return y


def preprocess_perts(x):
        return data_luminance(x, type = 'perturbation')


def synthetic_perturbations(delta):
    eta = np.linalg.norm(delta)
    v_syn = np.zeros(delta.shape)
    while np.linalg.norm(v_syn)<=eta:
        alpha = np.rand(1)
        pert_new = np.random.choice(delta.shape[0], delta.shape[0])
        v_new = delta[pert_new]
        v_syn = alpha*v_new + (1-alpha)*v_syn
    return v_syn


def show_sample(outfile, input, adv_input, perturb):
   clean_images = input.cpu().numpy().transpose(0, 2, 3, 1)
   adv_images = adv_input.cpu().numpy().transpose(0, 2, 3, 1)
   mean = [0.485, 0.456, 0.406]
   for ch in range(3):
        perturb[:, ch, :, :] = 10 * perturb[:, ch, :, :] * (1.0 / 255)
        perturb[:,ch,:,:] = perturb[:,ch,:,:] + torch.abs(perturb[:,ch,:,:].min())

   perturb=perturb.cpu().numpy().transpose(0, 2, 3, 1)
   plt.rcParams['figure.dpi']=120  # Increase size of pyplot plots
   plt.tight_layout()
   fig, axes=plt.subplots(2, 5, figsize = (10, 5))  # sharex=True, sharey=True)
   axes=axes.flatten()
   for ind, ax in enumerate(axes):
       if ind < 5:
           ax.imshow(clean_images[ind])
           ax.set_axis_off()
       else:
           ax.imshow(adv_images[ind - 5])
           ax.set_axis_off()
   fig.subplots_adjust(wspace = 0.02, hspace = 0)
   fig.suptitle('Sample clean and adversarial images', fontsize = 14)
   fig.savefig(outfile + '_samples_' + '.png')

   fig, axes=plt.subplots(1, 5, figsize = (10, 3))  # sharex=True, sharey=True)
   for ind, ax in enumerate(axes):
       ax.imshow(perturb[ind])
       ax.set_axis_off()

   fig.subplots_adjust(wspace = 0.02, hspace = 0)
   fig.suptitle('Sample perturbations', fontsize = 14)
   fig.savefig(outfile + '_perts_' + '.png')


def show_sample_perts(outfile, cls, perturb):
   for ch in range(3):
        perturb[:, ch, :, :] = 10 * perturb[:, ch, :, :] * (1.0 / 255)
        perturb[:,ch,:,:] = perturb[:,ch,:,:] + torch.abs(perturb[:,ch,:,:].min())

   perturb=perturb.cpu().numpy().transpose(0, 2, 3, 1)
   plt.rcParams['figure.dpi']=120  # Increase size of pyplot plots
   plt.tight_layout()
   fig, axes=plt.subplots(1, 5, figsize = (10, 3))  # sharex=True, sharey=True)
   for ind, ax in enumerate(axes):
       ax.imshow(perturb[ind])
       ax.set_axis_off()

   fig.subplots_adjust(wspace = 0.02, hspace = 0)
   fig.suptitle('Sample perturbations', fontsize = 14)
   fig.savefig(outfile + '/sampleperts/' + 'class_'+str(cls)+ '_arch_' + args.classifier_arch + '.png')


def save_perturbation_correlations(outfile, perts, patch_size):
    # assumes that input perturbation is already processed using image luminance.
    perts_original = perts
    perts_processed = preprocess_perts(perts/ 255.0)
    perts = perts_processed

    assert len(perts.shape) <= 4
    num_classes=perts.shape[0]
    num_perts=perts.shape[1]
    image_size=(perts.shape[2], perts.shape[3])

    num_x=image_size[0] * 1.0 / patch_size[0]
    num_y=image_size[1] * 1.0 / patch_size[1]
    num_patches=int(num_x * num_y)
    mean=np.zeros((num_classes, num_patches, patch_size[0] * patch_size[1]), dtype = np.float32)
    for cls in range(args.num_classes):
        correlation_mat=[]
        covariance_mat=[]
        mean_mat=[]
        norm=0
        for i in range(0, image_size[0], patch_size[0]):
            for j in range(0, image_size[1], patch_size[1]):
                patch = perts[cls, :, i:i + patch_size[0], j:j + patch_size[1]]
                patch = torch.reshape(patch, (-1, patch_size[0] * patch_size[1]))
                norm += torch.norm(patch, dim = -1)
                mean_patch = torch.mean(patch, dim = 0)
                mean_mat += [mean_patch]

                err = patch - mean_patch

        mean_mat = torch.stack(mean_mat)
        mean[cls]=mean_mat.cpu().numpy()
    suffix="_patchsize_{sz}".format(sz = args.patch_size)
    np.savez(outfile + suffix, mean = mean)


def main():
    print('==> Preparing dataset %s' % args.dataset)
    testloader=get_test_loader(args.dataset, args.location_dataset, batch_size = args.batch_size, num_workers = args.workers)
    length_testset={'CIFAR10': 10000, 'CIFAR100': 10000, 'ImageNet-100': 19338, 'ImageNet': 10000}
    cudnn.benchmark=True

    if args.attacker_type == 'targeted-sampler':
        attacker=attack_model.targeted_sampler(args.num_perts, args.num_classes,  args.image_size)
  
    else:
        print("Invalid attaacker-type")
        exit()

    attacker.load_state_dict(torch.load(args.location_attacker))

    attacker.eval()
    if args.cuda:
        attacker.to('cuda')
    
    samples_per_class = args.num_perts
    perturb = torch.zeros((args.num_classes, samples_per_class, 3, args.image_size, args.image_size))
    with torch.no_grad():
        for cls in np.arange(args.num_classes):
            if args.attacker_type == 'targeted-sampler':
                noise,targets = attacker.get_noise(samples_per_class, chosen_target_class = cls)
                if args.cuda:
                    targets = targets.cuda()
                    noise = noise.cuda()
                delta = attacker(noise)
            pert = torch.clamp(multiplier * delta, min_val, max_val)
            perturb[cls]=  pert
        if args.save_attack_statistics:
            save_perturbation_correlations(args.location_perts_means, perturb, (args.patch_size, args.patch_size))

     
    print("==> creating model '{}'".format(args.detector_arch))

    kwargs={
        'dataset': args.dataset,
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'num_components': args.num_components,
        'num_classes': args.num_classes,
        'location_generative_model': args.location_generative_model,
        'location_perts_means': args.location_perts_means,
        'score_type': args.score_type
    }

    detector = get_detectors(args.detector_arch, args.checkpoint_detector, **kwargs)
    detector.eval()
    if args.cuda:
        detector.to('cuda')
    

    if args.dataset == 'ImageNet':
        print("==> creating model '{}'".format(args.classifier_arch))
        if args.classifier_arch=='resnet':
            model = tv_models.resnet18(pretrained=True)
        if args.classifier_arch=='alexnet':
            model = tv_models.alexnet(pretrained=True)
        if args.classifier_arch=='googlenet':
            model = tv_models.googlenet(pretrained=True)
        if args.cuda:
            model = model.to('cuda')
        classifier = model
        classifier.eval()

    epsilon_mat = np.arange(0.0, 1.4, 0.1)
    top1_accuracy_mat = np.zeros((len(epsilon_mat),))
    top5_accuracy_mat = np.zeros((len(epsilon_mat),))
    top1_accuracy_targeted_mat = np.zeros((len(epsilon_mat),))
    conf_score_mat = np.zeros((len(epsilon_mat),))
    samples_class_clean = np.zeros((len(epsilon_mat), args.num_classes,))
    samples_class_adv = np.zeros((len(epsilon_mat), args.num_classes,))
    error_matrix = np.zeros((len(epsilon_mat), args.num_classes, args.num_classes))
    Pfa = 0.05

    Pd = np.zeros((len(epsilon_mat), 1))

    Psu = np.zeros((len(epsilon_mat), 1))

    correct_classified = np.zeros((length_testset[args.dataset],)) == 0
    test_signal = np.zeros((length_testset[args.dataset], ))
    test_null = np.zeros((length_testset[args.dataset],))
    shown = False


    for i, epsilon in enumerate(epsilon_mat):
        batch_ip = 0
        filled = 0
        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1_accuracy = AverageMeter()
        top1_accuracy_targeted = AverageMeter()
        top5_accuracy = AverageMeter()
        confidence_score = AverageMeter()
        detection_probability = AverageMeter()
        success_attack_probability = AverageMeter()
        bar = Bar('Processing', max=len(testloader))
        end = time.time()
        noise_size = args.nz
        for batch_idx, (inputs, ground_truths) in enumerate(testloader):
            data_time.update(time.time() - end)
            inputs, ground_truths = inputs.cuda(), ground_truths.cuda()
            if inputs.size(0)<=1:
                continue
            with torch.no_grad():
                if args.attacker_type == 'targeted-sampler':
                    chosen_target_classes = np.random.choice(args.num_classes, inputs.size(0))
                    noise_size = args.num_classes*args.num_perts
                    noise, targets = attacker.get_noise(inputs.size(0), indices_type = "MultiClass")
                    chosen_target_class = targets.data[0].cpu().numpy()
                    if args.cuda:
                        targets = targets.cuda()
                        noise = noise.cuda()
                    delta = attacker(noise)

                
                if args.synthetic:    
                    delta = synthetic_perturbations(delta)
                perturb = torch.clamp(delta * multiplier, min_val, max_val)
                adv_input = torch.clamp(inputs+epsilon*perturb/255, 0, 1)

                if epsilon==1. and shown ==False:
                    show_sample(outfile_name, inputs, adv_input, perturb)
                    shown=True

                input_processed = preprocess_classifier(inputs, dataset=args.dataset, type='input')
                adv_input_processed = preprocess_classifier(adv_input, dataset=args.dataset, type='input')

                prediction_clean = classifier(input_processed)
                prediction_adv = classifier(adv_input_processed)

                pred_clean = torch.argmax(prediction_clean, dim=1)
                pred_adv = torch.argmax(prediction_adv, dim=1)

                for gt_iter in range(pred_clean.shape[0]):
                    samples_class_clean[i, pred_clean[gt_iter].cpu().numpy()]+=1
                    samples_class_adv[i, pred_adv[gt_iter].cpu().numpy()]+=1
                    error_matrix[i, pred_clean[gt_iter].cpu().numpy(), pred_adv[gt_iter].cpu().numpy()]+=1
                correct = pred_adv == pred_clean
                correct_classified[filled:filled+len(pred_adv)] = correct.cpu().numpy()

            res, conf_score = accuracy(prediction_adv.data, ground_truths.data, topk=(1, 5))
            prec1, prec5 = res
            top1_accuracy.update(prec1.item(), inputs.size(0))
            top5_accuracy.update(prec5.item(), inputs.size(0))
            confidence_score.update(conf_score.item(), inputs.size(0))
            if args.detector_arch.startswith('gaussian'):
                test_statistics = detector(adv_input).cpu()
            if args.detector_arch.startswith('prn'):
                test_statistics = detector(adv_input_processed.cuda()).cpu()
                test_statistics = test_statistics.squeeze().cpu()



            
            if epsilon == 0:
                batch_len = len(test_statistics)

                test_null[filled:filled + batch_len] = test_statistics.detach().cpu().numpy()
                test_signal[filled:filled + batch_len] = test_null[filled:filled + batch_len]

                filled = filled + batch_len
            

            else:
                batch_len = len(test_statistics)
                test_signal[filled:filled + batch_len] = test_statistics.detach().cpu().numpy()
                filled = filled + batch_len


            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Epsilon" {epsilon: .4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                epsilon=epsilon,
                top1=top1_accuracy.avg,
                top5=top5_accuracy.avg,
                #top1_targeted = top1_accuracy_targeted.avg
            )
            bar.next()
        bar.finish()
        
        if epsilon == 0:
            testStatistics = (test_null, test_signal)
            ROC_params = computeROC(testStatistics)
            idx = approx(ROC_params[0], Pfa)
            learned_th = ROC_params[2][idx]
            forgery_detected = test_signal > learned_th

            Pd[i] = np.mean(forgery_detected)

            incorrect_classified = np.logical_not(correct_classified)
            miss_detect = np.logical_not(forgery_detected)

            Psu[i] = miss_detect.mean() * incorrect_classified[miss_detect].mean()
            print("Miss-Detect: ",miss_detect.mean())
            print("Incorrect Classified: ",incorrect_classified.mean())

        else:
            forgery_detected = test_signal > learned_th
            Pd[i] = np.mean(forgery_detected)
            incorrect_classified = np.logical_not(correct_classified)
            miss_detect = np.logical_not(forgery_detected)
            if miss_detect.mean() ==0.0:
                Psu[i] = 0
            else:
                Psu[i] = miss_detect.mean() * incorrect_classified[miss_detect].mean()
                print("Miss-Detect: ",miss_detect.mean())
                print("Incorrect Classified: ",incorrect_classified.mean())

        top1_accuracy_mat[i] = top1_accuracy.avg/100
        top5_accuracy_mat[i] = top5_accuracy.avg/100
        conf_score_mat[i] = confidence_score.avg

        print("Average Probability of detection of Semi-White Box attack", Pd[i])
        print("Average Probability of successful undetected attack", Psu[i])

    plt.figure()
    plt.plot(epsilon_mat,top1_accuracy_mat,'r--',marker ='s',linewidth =2,
        label ='Classifier Accuracy')
    plt.plot(epsilon_mat,conf_score_mat,'b--',marker ='o', linewidth =2,
        label ='Classifier Confidence')

    plt.plot(epsilon_mat,Pd,'g--',marker ='v', linewidth =2,
        label = 'Prob(Detection)')
    plt.plot(epsilon_mat,Psu,'k--',marker ='p', linewidth =2,
        label ='Prob(Successful undetected attack)')
    plt.xlabel('Perturbation strength ($\epsilon$)', fontsize ='14', color ='red')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        ncol=2, mode="expand", borderaxespad=0.)
    plt.grid()
    plt.savefig('%s_plots'%outfile_name)
    np.savez(outfile_name, epsilon = epsilon_mat, acc = top1_accuracy_mat, conf_score = conf_score_mat, acc_adversary_undeected = top1_accuracy_targeted_mat, Pd = Pd, Psu = Psu, error_matrix = error_matrix, samples_class_clean =  samples_class_clean, samples_class_adv = samples_class_adv)


if __name__ == '__main__':
    main()
