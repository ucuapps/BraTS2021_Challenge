import torch
import logging
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from torch.autograd import Variable


def expand_target(x, n_class, mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        self.n_classes = 4

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(
            self, pred: torch.Tensor, target: torch.Tensor, debug=False, softmax=True
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """

        if softmax:
            pred = torch.softmax(pred, dim=1)

        target = self._one_hot_encoder(target)

        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
                pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        total_loss = 0.0

        for i in range(0, self.n_classes):
            pred_dt = torch.from_numpy(self.distance_field(pred[:, i].cpu().detach().numpy())).float().cuda()
            target_dt = torch.from_numpy(self.distance_field(target[:, i].cpu().detach().numpy())).float().cuda()

            pred_error = (pred[:, i] - target[:, i]) ** 2
            distance = pred_dt ** self.alpha + target_dt ** self.alpha

            dt_field = pred_error * distance
            loss = dt_field.mean()

            total_loss += loss

        return total_loss / self.n_classes


def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num / den


def softmax_dice(output, target):
    '''
    The dice loss for using logits
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def softmax_diceCE(output, target, ce_weight=0.8):
    '''
    The dice & cross-entropy losses for using logits
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    # output = torch.softmax(output, dim=1)
    output = F.log_softmax(output, dim=1)

    ce_loss = nn.NLLLoss(weight=torch.Tensor([1, 2, 1, 1]).to('cuda:0'))
    target[target == 4] = 3
    ce_output = ce_loss(output, target)

    output = torch.exp(output)

    # loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    total_loss = (1 - ce_weight) * (loss1 + loss2 + loss3) + \
                 ce_weight * ce_output
    return total_loss, ce_output.item(), (loss1 + loss2 + loss3).item(), None


def softmax_diceCE_Focal(output, target, ce_weight=0.6):
    '''
    The dice & cross-entropy losses for using logits
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    # output = torch.softmax(output, dim=1)
    output = F.log_softmax(output, dim=1)

    ce_loss = nn.NLLLoss(weight=torch.Tensor([1, 2, 1, 1]).to('cuda:0'))
    target[target == 4] = 3
    ce_output = ce_loss(output, target)

    output = torch.exp(output)

    # loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    # Focal loss
    gamma = 2
    pt = torch.exp(-ce_output)
    focal_out = ((1 - pt) ** gamma * ce_output).mean()

    dice_weight = (1 - ce_weight)/2
    total_loss = dice_weight * (loss1 + loss2 + loss3) + \
                 ce_weight * ce_output + \
                 dice_weight * focal_out

    return total_loss, ce_output.item(), (loss1 + loss2 + loss3).item(), focal_out.item()



def softmax_diceCE_HD(output, target, ce_weight=0.4, hd_weight=0.1):
    '''
    The dice & cross-entropy & Hausdorf distance losses for using logits
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''

    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    ce_loss = nn.NLLLoss()
    target[target == 4] = 3
    ce_output = ce_loss(torch.log(output), target)

    hd_loss = HausdorffDTLoss()
    hd_out = hd_loss(output, target, softmax=False)
    total_loss = (1 - ce_weight - hd_weight) * (loss0 + loss1 + loss2 + loss3) + \
                 ce_weight * ce_output + \
                 hd_weight * hd_out

    return total_loss, ce_output.item(), (loss0 + loss1 + loss2 + loss3).item(), hd_out


def softmax_dice2(output, target):
    '''
    The dice loss for using logits
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    output = nn.Softmax(dim=1)(output)

    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3 + loss0, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    loss1 = Dice(output[:, 0, ...], (target == 1).float())
    loss2 = Dice(output[:, 1, ...], (target == 2).float())
    loss3 = Dice(output[:, 2, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1 - loss1.data, 1 - loss2.data, 1 - loss3.data


def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  # (b, h, w, d)
        target[target == 4] = 3  # transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  # extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2 * intersect[0] / (denominator[0] + eps)
    loss2 = 2 * intersect[1] / (denominator[1] + eps)
    loss3 = 2 * intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1, loss2, loss3


def Dual_focal_loss(output, target):
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    if target.dim() == 4:  # (b, h, w, d)
        target[target == 4] = 3  # transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  # extend target from (b, h, w, d) to (b, c, h, w, d)

    target = target.permute(1, 0, 2, 3, 4).contiguous()
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    target = target.view(4, -1)
    output = output.view(4, -1)
    log = 1 - (target - output) ** 2

    return -(F.log_softmax((1 - (target - output) ** 2), 0)).mean(), 1 - loss1.data, 1 - loss2.data, 1 - loss3.data
