# python -m torch.distributed.launch --nproc_per_node=1 train.py --batch_size 2 --accum_iter 4

import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models import criterions
from predict import softmax_mIOU_score, softmax_output_dice

from data.BraTS import BraTS
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='./', type=str)

parser.add_argument('--train_dir', default='train', type=str)

parser.add_argument('--valid_dir', default='val', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='val.txt', type=str)

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
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_workers', default=16, type=int)

parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--accum_iter', default=4, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=100, type=int)

parser.add_argument('--save_freq', default=5, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
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

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    model.cuda(0)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    criterion = getattr(criterions, args.criterion)

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    resume = ''

    writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    valid_root = os.path.join(args.root, args.valid_dir)

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    train_set = BraTS(train_list, train_root, args.mode)
    valid_set = BraTS(valid_list, valid_root, args.mode)   # mode='test'

    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for valid = {}'.format(len(valid_set)))

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_set, shuffle=False, batch_size=args.batch_size,
                              drop_last=False, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        train_loss_history, train_ce_history, train_hd_history, train_dice_history = [], [], [], []
        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            output = model(x)

            loss, ce, dice, hd = criterion(output, target)
            loss = loss / args.accum_iter

            train_loss_history.append(loss.item()), train_ce_history.append(ce), train_dice_history.append(dice)
            if hd:
                train_hd_history.append(hd)

            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{}  loss: {:.5f} | CE: {:.5f} | Dice sum: {:.5f}'
                             .format(epoch, i, loss.item(), ce, dice))

            loss.backward()

            if (i + 1) % args.accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 5 == 0:
                slice_id = 64
                target[target == 4] = 3
                pred_mask = torch.argmax(output, dim=1)
                writer.add_image('Ground_Truth', target[0, ...].unsqueeze(0).cpu().detach().numpy()[:, :, slice_id], i, dataformats='CHW')
                writer.add_image('Prediction', pred_mask[0, ...].unsqueeze(0).cpu().detach().numpy()[:, :, slice_id], i, dataformats='CHW')

            torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            valid_loss_history = []
            metrics_dice = []
            for i, data in enumerate(valid_loader):
                x, target = data
                x = x.cuda(args.local_rank, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

                output = model(x)
                loss, _, _, _ = criterion(output, target)

                # Calculate metrics
                mdice = softmax_output_dice(F.softmax(output,  dim=1).argmax(1), target)

                metrics_dice.append([x.cpu().detach().numpy() for x in mdice])

                logging.info('Epoch: {}_Valid_Iter:{} loss: {:.5f}, mdice0: {:.5f}, mdice1: {:.5f}, mdice2: {:.5f}'.format(epoch, i,
                                                    loss.item(), mdice[0].item(), mdice[1].item(), mdice[2].item()))
                valid_loss_history.append(loss.item())

            writer.add_scalar('Validation Mean dice 1:', np.mean([x[0] for x in metrics_dice]), epoch)
            writer.add_scalar('Validation Mean dice 2:', np.mean([x[1] for x in metrics_dice]), epoch)
            writer.add_scalar('Validation Mean dice 3:', np.mean([x[2] for x in metrics_dice]), epoch)

        end_epoch = time.time()
        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                or (epoch + 1) % int(args.end_epoch - 3) == 0:
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

        writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train loss:', np.mean(train_loss_history), epoch)
        writer.add_scalar('valid loss:', np.mean(valid_loss_history), epoch)

        writer.add_scalar('train CE loss:', np.mean(train_ce_history), epoch)
        writer.add_scalar('train HD loss:', np.mean(train_hd_history), epoch)
        writer.add_scalar('train HD loss:', np.mean(train_dice_history), epoch)

        torch.cuda.empty_cache()

        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
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
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


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
