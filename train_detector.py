from __future__ import print_function

import argparse
import os
import shutil
import time
import pdb
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
from torch.autograd import grad
from utils import *
import torch.optim as optim
import pickle
from sklearn import mixture
from torch.autograd import Variable
import scipy
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import attack_model
import pdb
import torch_dct as dct
from sklearn import linear_model
from sklearn.utils import shuffle
import lo_glrt
from score_functions import Score_Gaussian, Score_GMM
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

# # Checkpoints
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# attacker
parser.add_argument('--attacker_type', type=str, default='targeted-sampler', help='Type of the attacker to use')  # use tiny-imagenet for Tiny Imagenet dataset
parser.add_argument('--num_perts', type=int, default=2, help='Number of perturbations per-class to obtain')
parser.add_argument('--targeted', type=int, default=1, help='if the attack is targeted (default False)')

parser.add_argument('--location_attacker', default='models/attacker/checkpoint/ImageNet/attacker_targeted-sampler_ImageNet_ressnet18.pth.tar', type=str, metavar='PATH',
                    help='path to the attacker\'s model')

# detector
parser.add_argument('--detector_arch', '-da', metavar='ARCH', default='gaussian_firstorder_composite_supervised',
                    help='model architecture: '
                    + ' (default: gaussian)')
parser.add_argument('--patch_size', type=int, default=8, help='Patch size')
parser.add_argument('--num_components', type=int, default=1, help='Components of GMM')
parser.add_argument('--score_type', type=str, default='Gaussian', help='Type of Score function to use in the detector')
parser.add_argument('--checkpoint_detector', default='', help="")

# /home/agoel10/Documents/advml_watermark/UAP_LOD/code/detector/prn_detector/detectors/CIFAR10/PRN_detector_CIFAR10_vgg16_bn_targeted-sampler.pth.tar
# Important
parser.add_argument('--location_generative_model', default='models/generative_models/dataset_ImageNet_patch_8_components_1_ML_luminance.p', type=str, metavar='PATH',
                    help='path to the generative model trained using ML/EM algorithm in sci-kit learn (default: none)')
# parser.add_argument('--location_detector', default='', type=str, metavar='PATH',
#                     help='path to the generative model trained usinqg ML/EM algorithm in sci-kit learn (default: none)')


parser.add_argument('--location_perts_means', default='mean_perturbations_targeted-sampler/mean_ImageNet_resnet18', type=str, metavar='PATH',
                    help='path to mean of the perturbations')
parser.add_argument('--save_attack_statistics', type=int, default=0, help='if the attack is targeted (default False)')


# classifier
parser.add_argument('--classifier_arch', '-ca', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')


# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                    help='batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--train_val_split', default=0.8, type=float,
                    metavar='split', help='training validation split')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')

args = parser.parse_args()
args.train = True
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
best_acc = 0  # best test accuracy


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
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


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_' + filename))


def target_indices(chosen_target_classes, num_classes, num_perts):
    return np.array([np.random.choice(range(cls * num_perts, num_perts * (1 + cls))) for cls in chosen_target_classes])


def preprocess_perts(x):
    return data_luminance(x, type='perturbation')


def preprocess_classifier(x, dataset, type='input'):
    y = torch.zeros(x.shape, device='cuda', dtype=torch.float32)
    y[:] = x
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        if type == 'input':
            y = y * 255.0
        mean, std = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        if type == 'input':
            for ch in range(3):
                y[:, ch, :, :] = (x[:, ch, :, :] * 1.0 / 255 - mean[ch]) / std[ch]
        if type == 'perturbation':
            for ch in range(3):
                y[:, ch, :, :] = (x[:, ch, :, :] * (1.0 / 255)) / std[ch]
        return y

    if dataset == 'ImageNet':
        y = torch.zeros(x.shape, device='cuda', dtype=torch.float32)
        y[:] = x
        if type == 'input':
            y = y * 255.0
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        y = torch.zeros(x.shape, device='cuda', dtype=torch.float32)
        if type == 'input':
            for ch in range(3):
                y[:, ch, :, :] = (x[:, ch, :, :] * 1.0 / 255 - mean[ch]) / std[ch]
        if type == 'perturbation':
            for ch in range(3):
                y[:, ch, :, :] = (x[:, ch, :, :] * (1.0 / 255)) / std[ch]
        return y


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing dataset %s' % args.dataset)

    train_loader, val_loader = get_train_valid_loader(args.dataset, args.location_dataset, batch_size=args.batch_size, num_workers=args.workers, valid_size=args.train_val_split)
    test_loader = get_test_loader(args.dataset, args.location_dataset, batch_size=int(args.batch_size / 2), num_workers=args.workers)

    length_testset = {'CIFAR10': 10000, 'CIFAR100': 10000}
    cudnn.benchmark = True

    print("==> creating attacker's model ")
    if args.attacker_type == 'NAG':
        attacker = attack_model.NAG_netAttacker(args.nz, ngpu, args.num_classes, args.image_size, args.dataset)
    elif args.attacker_type == 'non-targeted-sampler':
        attacker = attack_model.non_targeted_sampler(args.num_perts, args.image_size)
    elif args.attacker_type == 'targeted-sampler':
        attacker = attack_model.targeted_sampler(args.num_perts, args.num_classes, args.image_size)

    attacker.load_state_dict(torch.load(args.location_attacker))

    attacker.eval()
    attacker.to('cuda')

    kwargs = {
        'dataset': args.dataset,
        'image_size': args.image_size,
        'patch_size': args.patch_size,
        'num_components': args.num_components,
        'num_classes': args.num_classes,
        'location_generative_model': args.location_generative_model,
        'location_perts_means': args.location_perts_means,
        'score_type': args.score_type

    }

    print("==> creating model for detector'{}'".format(args.detector_arch))
    detector = get_detectors(args.detector_arch, args.checkpoint_detector, **kwargs)

    detector.to('cuda')

    criterion = torch.nn.BCELoss()
    threshold = nn.Parameter(torch.tensor(1.5, dtype=torch.float, device='cuda'))
    optimizer = optim.Adam(list(detector.parameters()) + [threshold], lr=args.lr, weight_decay=args.weight_decay)

    for name, param in detector.named_parameters():
        print(name)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, attacker, detector, criterion, optimizer, threshold, epoch)

        append logger file

        is_best = train_acc > best_acc
        # pdb.set_trace()
        best_acc = max(train_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': detector.state_dict(),
            'acc': train_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint + '/' + args.dataset, filename='LOGLRT_' + args.detector_arch + '_' + args.classifier_arch + '.pth.tar')

        test_loss, test_acc = test(test_loader, attacker, detector, criterion, threshold, epoch)
        print("test_accuracy: ", test_acc)
    print('Best acc:')
    print(best_acc)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def train(trainloader, attacker, detector, criterion, optimizer, threshold, epoch):
    detector.train()
    detector.to('cuda')
    data_X = []
    data_Y = []
    data_time = AverageMeter()
    batch_time = AverageMeter()
    train_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, (inputs, groundtruths) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        with torch.no_grad():
            epsilon = torch.tensor(1.0 + np.zeros(1), dtype=torch.float, device='cuda')
            if args.attacker_type == 'targeted-sampler':
                chosen_target_classes = np.random.choice(args.num_classes, inputs.size(0))

                # A randomly chosen class and
                noise_size = args.num_classes * args.num_perts
                noise, _ = attacker.get_noise(inputs.size(0), indices_type="MultiClass")
                if args.cuda:
                    noise = noise.cuda()
                delta = attacker(noise)

            perturb = torch.clamp(delta * multiplier, min_val, max_val)
        adv_input = torch.clamp(inputs + epsilon * perturb / 255, 0, 1)

        # input_processed = preprocess_classifier(inputs, dataset=args.dataset, type='input')
        # adv_input_processed = preprocess_classifier(adv_input, dataset=args.dataset, type='input')

        if np.random.rand(1) < 0.5:
            test_st = detector(adv_input)
            if args.detector_arch.startswith('prn'):
                pdb.set_trace()
                pred = torch.sigmoid(test_st)
            else:
                pred = torch.sigmoid(test_st - threshold)

            target = torch.ones(adv_input.size(0), dtype=torch.float, device='cuda')

        else:
            test_st = detector(inputs)
            if args.detector_arch.startswith('prn'):
                pred = torch.sigmoid(test_st)
            else:
                pred = torch.sigmoid(test_st - threshold)
            target = torch.zeros(inputs.size(0), dtype=torch.float, device='cuda')

        labels = pred > 0.5
        correct = labels == target
        accuracy.update(correct.type(torch.float).mean().item())
        loss = criterion(pred, target)
        try:
            losses.update(loss.item(), inputs.size(0))
        except:
            pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Train: {train:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | Accuracy: {acc:.4f}| Total: {total:} | ETA: {eta:} '.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            train=train_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            acc=accuracy.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )
        bar.next()
    bar.finish()

    return losses.avg, accuracy.avg


def test(testloader, attacker, detector, criterion, threshold, epoch):
    detector.eval()
    detector.to('cuda')
    data_X = []
    data_Y = []
    data_time = AverageMeter()
    batch_time = AverageMeter()
    train_time = AverageMeter()
    accuracy = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    for batch_idx, (inputs, groundtruths) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs = inputs.cuda()
        with torch.no_grad():
            epsilon = 1
            if args.attacker_type == 'targeted-sampler':
                chosen_target_classes = np.random.choice(args.num_classes, inputs.size(0))
                noise_size = args.num_classes * args.num_perts
                noise, _ = attacker.get_noise(inputs.size(0), indices_type="MultiClass")
                if args.cuda:
                    noise = noise.cuda()
                delta = attacker(noise)

            perturb = torch.clamp(delta * multiplier, min_val, max_val)
        adv_input = torch.clamp(inputs + epsilon * perturb / 255, 0, 1)

        input_processed = preprocess_classifier(inputs, dataset=args.dataset, type='input')
        adv_input_processed = preprocess_classifier(adv_input, dataset=args.dataset, type='input')

        if np.random.rand(1) < 0.5:
            if args.detector_arch.startswith('gaussian') or args.detector_arch.startswith('gmm'):
                test_st = detector(adv_input)
            if args.detector_arch.startswith('prn'):
                test_st = detector(adv_input_processed.cuda())

            if args.detector_arch.startswith('prn'):
                pred = torch.sigmoid(test_st)
            else:
                pred = torch.sigmoid(test_st - threshold)
            target = torch.ones(adv_input.size(0), dtype=torch.float, device='cuda')

        else:
            if args.detector_arch.startswith('gaussian') or args.detector_arch.startswith('gmm'):
                test_st = detector(inputs)
            if args.detector_arch.startswith('prn'):
                test_st = detector(input_processed.cuda())

            if args.detector_arch.startswith('prn'):
                pred = torch.sigmoid(test_st)
            else:
                pred = torch.sigmoid(test_st - threshold)

            target = torch.zeros(inputs.size(0), dtype=torch.float, device='cuda')

        labels = pred > 0.5
        correct = labels == target
        accuracy.update(correct.type(torch.float).mean().item())
        loss = criterion(pred, target)
        losses.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progressf
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Train: {train:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}| Accuracy: {acc:.4f} | Total: {total:} | ETA: {eta:} '.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            train=train_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            acc=accuracy.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )
        bar.next()
    bar.finish()

    return losses.avg, accuracy.avg


if __name__ == '__main__':
    main()
