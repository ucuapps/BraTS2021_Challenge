import argparse

from medpy import metric
import numpy as np
import glob
import os
import pickle
import concurrent.futures as cf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int,
                        default=4, help='output channel of network')

    parser.add_argument('--root', default='/common/ostapviniavskyi/RSNA_ASNR_MICCAI_BraTS2021_TrainingData', type=str)
    parser.add_argument('--valid_dir', default='', type=str)
    parser.add_argument('--valid_file', default='valid.txt', type=str)
    parser.add_argument('--pred_dir',
                        default='/home/ostapvinianskyi/projects/BraTS2021_Challenge/TransBTS/checkpoint/TransBTS2021-07-21-13:09:15/model_epoch_n29_prediction',
                        type=str)

    parser.add_argument('--num_workers', default=20, type=int)

    args = parser.parse_args()
    return args


def calculate_metric_percase(pred, gt):
    pred, gt = pred.astype(np.uint8), gt.astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def run_case(pred, gt, num_classes=4):
    metric_i = []
    for i in range(1, num_classes):
        metric_i.append(calculate_metric_percase(pred == i, gt == i))
    return np.mean(np.array(metric_i), axis=0)


def main():
    args = parse_args()
    pred_list = glob.glob(args.pred_dir + '/*.npy')
    key2path = {path.split('/')[-1][:-4]: path for path in pred_list}

    with cf.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        fs = []
        for key, path in key2path.items():
            y_pred = np.load(path)[..., :-5]

            with open(os.path.join(args.root, args.valid_dir, key, key + '_data_f32b0.pkl'), 'rb') as f:
                _, y = pickle.load(f)
                y[y == 4] = 3
            future = executor.submit(run_case, y_pred, y, num_classes=args.num_classes)
            fs.append(future)

        results = []
        for i, future in enumerate(cf.as_completed(fs)):
            result = future.result()
            results.append(result)
            print(f'[{i + 1}/{len(fs)}] Mean Dice: {result[0]}, HD@95: {result[1]}')

        results = np.array(results).mean(axis=0)
        print(f'Val Mean Dice: {results[0]}, Val HD@95: {results[1]}')


if __name__ == '__main__':
    main()
