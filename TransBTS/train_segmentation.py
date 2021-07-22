# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import argparse
import logging
import os
import random
import time

import numpy as np
import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from models import criterions
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

local_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='ostapviniavskyi', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--date', default=local_time, type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='/common/ostapviniavskyi/RSNA_ASNR_MICCAI_BraTS2021_TrainingData', type=str)

parser.add_argument('--train_dir', default='.', type=str)

parser.add_argument('--valid_dir', default='.', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--dataset', default='brats', type=str)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--input_C', default=4, type=int)

parser.add_argument('--input_H', default=240, type=int)

parser.add_argument('--input_W', default=240, type=int)

parser.add_argument('--input_D', default=160, type=int)

parser.add_argument('--crop_H', default=128, type=int)

parser.add_argument('--crop_W', default=128, type=int)

parser.add_argument('--crop_D', default=128, type=int)

parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=[0.0001, 0.001], type=list)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_diceCE', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=10, type=int)

parser.add_argument('--batch_size', default=2, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=80, type=int)

parser.add_argument('--save_freq', default=5, type=int)

parser.add_argument('--grad_accum_steps', default=8, type=int)

parser.add_argument('--resume_path',
                    default='/home/ostapvinianskyi/projects/BraTS2021_Challenge/TransBTS/checkpoint/TransBTS2021-07-18-19:24:41/model_epoch_last.pth',
                    type=str)

parser.add_argument('--resume', default=False, type=bool)

parser.add_argument('--pretrained_encoder_path',
                    default='/home/ostapvinianskyi/projects/BraTS2021_Challenge/TransBTS/checkpoint/autoencoder__learned_pe__mae/model_epoch_last.pth',
                    type=str)

parser.add_argument('--pretrained_encoder', default=True, type=bool)

args = parser.parse_args()


def load_encoder(model, state_dict):
    state_dict = {k: v for k, v in state_dict.items() if
                  k.startswith('Unet') or k.startswith('bn') or k.startswith('conv_x')}
    model.load_state_dict(state_dict, strict=False)


def split_params(model):
    g1, g2 = [], []
    for name, param in model.named_parameters():
        if name.startswith('Unet') or name.startswith('bn') or name.startswith('conv_x'):
            g1.append(param)
        else:
            g2.append(param)
    return g1, g2


def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0')

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="mlp")

    model.train()

    param_lr = [{'params': params, 'lr': lr} for params, lr in zip(split_params(model), args.lr)]
    optimizer = torch.optim.Adam(param_lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = getattr(criterions, args.criterion)

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment + args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter()

    if os.path.isfile(args.resume_path) and args.resume:
        logging.info('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume_path, args.start_epoch))
    else:
        logging.info('re-training!!!')

    if os.path.isfile(args.pretrained_encoder_path) and args.pretrained_encoder:
        logging.info('loading encoder checkpoint {}'.format(args.pretrained_encoder_path))
        checkpoint = torch.load(args.pretrained_encoder_path, map_location='cpu')

        load_encoder(model, checkpoint['state_dict'])

        logging.info('Successfully loading encoder checkpoint {}'
                     .format(args.pretrained_encoder_path))
    model.to(device)

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)

    train_set = BraTS(train_list, train_root, args.mode)
    valid_set = BraTS(valid_list, valid_root, args.mode)

    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for valid = {}'.format(len(valid_set)))

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=args.batch_size,
                              drop_last=False, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()

        train_loss_history = []
        for i, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            x, y = data
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            output = model(x)
            # output = torch.softmax(output, dim=1)
            loss, *_ = criterion(output, y)

            train_loss_history.append(loss.item())
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f}'
                         .format(epoch, i, loss.item()))

            loss.backward()

            if (i + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            valid_loss_history = []
            for i, data in enumerate(valid_loader):
                x, y = data
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                output = model(x)
                # output = torch.softmax(output, dim=1)
                loss, *_ = criterion(output, y)

                valid_loss_history.append(loss.item())
                logging.info('Epoch: {}_Valid_Iter:{}  loss: {:.5f}'
                             .format(epoch, i, loss.item()))

        end_epoch = time.time()

        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                or (epoch + 1) % int(args.end_epoch - 3) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_n{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

        writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train dice-loss:', np.mean(train_loss_history), epoch)
        writer.add_scalar('valid dice-loss:', np.mean(valid_loss_history), epoch)

        epoch_time_minute = (end_epoch - start_epoch) / 60
        remaining_time_hour = (args.end_epoch - epoch - 1) * epoch_time_minute / 60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    writer.close()

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        final_name)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for init_lr_, param_group in zip(init_lr, optimizer.param_groups):
        param_group['lr'] = round(init_lr_ * np.power(1 - (epoch) / max_epoch, power), 8)


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
