import argparse
import logging
import os
import random
import sys
import numpy as np

from data.BraTS import BraTS
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from medpy import metric


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')

parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

parser.add_argument('--root', default='/common/ostapviniavskyi/RSNA_ASNR_MICCAI_BraTS2021_TrainingData', type=str)
parser.add_argument('--valid_dir', default='', type=str)
parser.add_argument('--valid_file', default='valid.txt', type=str)
parser.add_argument('--weights_path', default='/home/ostapvinianskyi/projects/BraTS2021_Challenge/TransBTS/checkpoint/TransBTS2021-07-20-12:04:53/model_epoch_last.pth', type=str)
parser.add_argument('--gpu', default=None, type=str)

args = parser.parse_args()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def inference(args, model, db_test, testloader, test_save_path=None):
    if args.gpu:
        model.cuda(0)
    model.eval()

    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):

        x, target = sampled_batch
        target[target == 4] = 3

        if args.gpu:
            x = x.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

        with torch.no_grad():
            out = torch.argmax(torch.softmax(model(x), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

        metric_i = []
        for i in range(1, args.num_classes):
            metric_i.append(calculate_metric_percase(prediction == i, target.squeeze(0).numpy() == i))
        metric_list += np.array(metric_i)

        print('idx %d mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if args.gpu:
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.cuda.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    valid_root = os.path.join(args.root, args.valid_dir)
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_set = BraTS(valid_list, valid_root, mode='valid')  # mode='test'

    valid_loader = DataLoader(dataset=valid_set, shuffle=False, batch_size=1,
                              drop_last=False, num_workers=10, pin_memory=True)

    _, net = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    if args.gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')

    state_dict = torch.load(args.weights_path,  map_location=device)['state_dict']

    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

    net.load_state_dict(state_dict)

    inference(args, net, valid_set, valid_loader)


