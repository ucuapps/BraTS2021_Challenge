import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data.BraTS import BraTS
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int,
                        default=4, help='output channel of network')

    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    parser.add_argument('--root', default='/common/ostapviniavskyi/RSNA_ASNR_MICCAI_BraTS2021_TrainingData', type=str)
    parser.add_argument('--valid_dir', default='', type=str)
    parser.add_argument('--valid_file', default='valid.txt', type=str)
    parser.add_argument('--weights_path',
                        default='/home/ostapvinianskyi/projects/BraTS2021_Challenge/TransBTS/checkpoint/TransBTS2021-07-21-13:09:15/model_epoch_n29.pth',
                        type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=2, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.gpu:
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.cuda.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = args.weights_path[:-4] + '_prediction'
    os.makedirs(save_dir, exist_ok=True)

    valid_root = os.path.join(args.root, args.valid_dir)
    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_set = BraTS(valid_list, valid_root, mode='valid', return_names=True)

    valid_loader = DataLoader(dataset=valid_set, shuffle=False, batch_size=1,
                              drop_last=False, pin_memory=True)
    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="mlp")

    if args.gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    state_dict = torch.load(args.weights_path, map_location='cpu')['state_dict']

    for key in list(state_dict.keys()):
        state_dict[key.replace('module.', '')] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for x, y, name in tqdm(valid_loader):
            y_pred = model(x.to(device))[0]
            y_pred = torch.argmax(y_pred, dim=0).cpu().numpy()
            y_pred = y_pred.astype(np.uint8)
            np.save(os.path.join(save_dir, name[0] + '.npy'), y_pred)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
