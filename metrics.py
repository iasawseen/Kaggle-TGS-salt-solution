import torch
import abc
import numpy as np

SMOOTH = 1e-6


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, input, target):
        pass

    @abc.abstractmethod
    def compute(self):
        pass


class BinaryAccuracy(Metric):
    def __init__(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, input, target):
        target = target > 0.5
        target = target.float()

        correct = torch.eq(torch.round(input).type(target.type()), target).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.size()[0]

    def compute(self):
        return self._num_correct / self._num_examples


class DiceCoefficient(Metric):
    def __init__(self):
        self._dice_coeffs = 0
        self._num_examples = 0

    def update(self, input, target, threshold=0.5):
        input = input > threshold
        input = input.float()

        target = target > 0.5
        target = target.float()

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        self._dice_coeffs += ((2. * intersection.item() + SMOOTH)
                              / (iflat.sum().item() + tflat.sum().item() + SMOOTH))

        self._num_examples += 1

    def compute(self):
        return self._dice_coeffs / self._num_examples


class CompMetric(Metric):
    def __init__(self):
        self.reset()

    def update(self, input, target, threshold=0.5):
        input = input > threshold
        target = target > threshold
        input = input.cpu().numpy()
        target = target.cpu().numpy()

        self._comp_metrics += calc_metric(input, target).item()

        self._num_examples += input.shape[0]

    def compute(self):
        return self._comp_metrics / self._num_examples

    def reset(self):
        self._comp_metrics = 0
        self._num_examples = 0


def calc_iou(actual, pred):
    intersection = np.count_nonzero(actual * pred)
    union = np.count_nonzero(actual) + np.count_nonzero(pred) - intersection
    iou_result = intersection / union if union != 0 else 0.
    return iou_result


def calc_ious(actuals, preds):
    ious_ = np.array([calc_iou(a, p) for a, p in zip(actuals, preds)])
    return ious_


def calc_precisions(thresholds, ious):
    thresholds = np.reshape(thresholds, (1, -1))
    ious = np.reshape(ious, (-1, 1))
    ps = ious > thresholds
    mps = ps.mean(axis=1)
    return mps


def indiv_scores(masks, preds):
    masks[masks > 0] = 1
    preds[preds > 0] = 1
    ious = calc_ious(masks, preds)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = calc_precisions(thresholds, ious)
    emptyMasks = np.count_nonzero(masks.reshape((len(masks), -1)), axis=1) == 0
    emptyPreds = np.count_nonzero(preds.reshape((len(preds), -1)), axis=1) == 0
    adjust = (emptyMasks == emptyPreds).astype(np.float)
    precisions[emptyMasks] = adjust[emptyMasks]

    return precisions


def calc_metric(preds, masks):
    return np.sum(indiv_scores(masks, preds))


# PyTroch version


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).sum((1, 2)).float()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).sum((1, 2)).float()  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.sum()  # Or thresholded.mean() if you are interested in average across the batch


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()

