import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from transformations import (
    ToTensor, Normalize, MasksAdder,
    AffineAugmenter, CropAugmenter, ClassAdder
)

from utils import build_data_loader

from models.sawseennet import SawSeenNet

from modules.losses import LovaszLoss, LossAggregator
from attrdict import AttrDict
from metrics import BinaryAccuracy, DiceCoefficient, CompMetric
from utils import load_train_data, load_test_data, trim_masks
from sklearn.model_selection import KFold
from utils import make_if_not_exist
from trainers import Trainer

IMG_SIZE_ORIGIN = 101
IMG_SIZE_TARGET = 12
IMG_INPUT_CHANNELS = 1

INPUT_PATH = './input/'
TRAIN_PATH = INPUT_PATH + 'train/'
TEST_PATH = INPUT_PATH + 'test/'

TRAIN_IMAGES_PATH = TRAIN_PATH + 'images/'
TRAIN_MASKS_PATH = TRAIN_PATH + 'masks/'
TEST_IMAGES_PATH = TEST_PATH + 'images/'

EXP_NAME = 'final_model'
FOLDS_FILE_PATH = './' + EXP_NAME + '_fold_{}.npy'
CUDA_ID = 1

LR = 0.01
TUNE_LR = 0.01
MIN_LR = 0.0001
MOMENTUM = 0.9
CYCLE_LENGTH = 64
CYCLES = 4

BATCH_SIZE = 16
RANDOM_SEED = 42
FOLDS = 5
THRESHOLD = 0.5
BCE_EPOCHS = 32
INTERMEDIATE_EPOCHS = 2
PRETRAINED_COOLDOWN = 2
DROPOUT_COOLDOWN = 2
NUM_EPOCHS = BCE_EPOCHS + INTERMEDIATE_EPOCHS + CYCLES * CYCLE_LENGTH

CLASS_WEIGHT = 0.05
MASKS_WEIGHT = 0.12
PROB = 0.5
PROB_CLASS = 0.8

VAL_METRIC_CRITERION = 'comp_metric'
MODEL_FILE_DIR = './saved_models_' + str(EXP_NAME)
LOGS_DIR = './logs/logs_' + str(EXP_NAME)

make_if_not_exist(MODEL_FILE_DIR)
make_if_not_exist(LOGS_DIR)

MODEL_FILE_PATH = MODEL_FILE_DIR + '/model_{}_{:.4f}'


def predict(config, model, data_loader, thresholding=True, threshold=THRESHOLD, tta=True):
    model.set_training(False)

    y_preds = []
    with torch.no_grad():
        for sample_batch in data_loader:
            x = sample_batch['x']
            y_pred, *preds = model(x.cuda(config.cuda_index))
            y_pred = torch.sigmoid(y_pred)

            if tta:
                x_flipped = x.flip(3)
                y_pred_flipped, *preds = model(x_flipped.cuda(config.cuda_index))
                y_pred_flipped = torch.sigmoid(y_pred_flipped)
                y_pred_flipped = y_pred_flipped.flip(3)

                y_pred += y_pred_flipped
                y_pred /= 2

            if thresholding:
                y_pred = y_pred > threshold

            y_pred = y_pred.cpu().numpy().transpose((0, 2, 3, 1))
            y_preds.append(y_pred)

    y_preds = np.concatenate(y_preds, axis=0)
    return y_preds


def k_fold():
    images, masks = load_train_data(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)
    test_file_paths, test_images = load_test_data(TEST_IMAGES_PATH, load_images=True, to256=False)

    train_transformer = transforms.Compose([CropAugmenter(), AffineAugmenter(), MasksAdder(), ToTensor(),
                                            Normalize(), ClassAdder()])

    eval_transformer = transforms.Compose([MasksAdder(), ToTensor(), Normalize(), ClassAdder()])

    predict_transformer = transforms.Compose([ToTensor(predict=True), Normalize(predict=True)])

    test_images_loader = build_data_loader(test_images, None, predict_transformer, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=4, predict=True)

    k_fold = KFold(n_splits=FOLDS, random_state=RANDOM_SEED, shuffle=True)

    test_masks_folds = []

    config = AttrDict({
        'cuda_index': CUDA_ID,
        'momentum': MOMENTUM,
        'lr': LR,
        'tune_lr': TUNE_LR,
        'min_lr': MIN_LR,
        'bce_epochs': BCE_EPOCHS,
        'intermediate_epochs': INTERMEDIATE_EPOCHS,
        'cycle_length': CYCLE_LENGTH,
        'logs_dir': LOGS_DIR,
        'masks_weight': MASKS_WEIGHT,
        'class_weight': CLASS_WEIGHT,
        'val_metric_criterion': 'comp_metric'
    })

    for index, (train_index, valid_index) in list(enumerate(k_fold.split(images))):
        print('fold_{}\n'.format(index))

        x_train_fold, x_valid = images[train_index], images[valid_index]
        y_train_fold, y_valid = masks[train_index], masks[valid_index]

        train_data_loader = build_data_loader(x_train_fold, y_train_fold, train_transformer, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4, predict=False)
        val_data_loader = build_data_loader(x_valid, y_valid, eval_transformer, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=4, predict=False)
        test_data_loader = build_data_loader(x_valid, y_valid, eval_transformer, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=4, predict=False)

        data_loaders = AttrDict({
            'train': train_data_loader,
            'val': val_data_loader,
            'test': test_data_loader
        })

        zers = np.zeros(BCE_EPOCHS)
        zers += 0.1
        lovasz_ratios = np.linspace(0.1, 0.9, INTERMEDIATE_EPOCHS)
        lovasz_ratios = np.hstack((zers, lovasz_ratios))
        bce_ratios = 1.0 - lovasz_ratios
        loss_weights = [(bce_ratio, lovasz_ratio) for bce_ratio, lovasz_ratio in zip(bce_ratios, lovasz_ratios)]

        loss = LossAggregator((nn.BCEWithLogitsLoss(), LovaszLoss()), weights=[0.9, 0.1])

        metrics = {'binary_accuracy': BinaryAccuracy,
                   'dice_coefficient': DiceCoefficient,
                   'comp_metric': CompMetric}

        segmentor = SawSeenNet(base_channels=64, pretrained=True, frozen=False).cuda(config.cuda_index)

        trainer = Trainer(config=config, model=segmentor, loss=loss, loss_weights=loss_weights,
                          metrics=metrics, data_loaders=data_loaders)

        segmentor = trainer.train(num_epochs=NUM_EPOCHS, model_pattern=MODEL_FILE_PATH + '_{}_fold.pth'.format(index))

        test_masks = predict(config, segmentor, test_images_loader, thresholding=False)
        test_masks = trim_masks(test_masks, height=IMG_SIZE_ORIGIN, width=IMG_SIZE_ORIGIN)

        test_masks_folds.append(test_masks)

        np.save(FOLDS_FILE_PATH.format(index), test_masks)

    result_masks = np.zeros_like(test_masks_folds[0])

    for test_masks in test_masks_folds:
        result_masks += test_masks

    result_masks = result_masks.astype(dtype=np.float32)
    result_masks /= FOLDS
    result_masks = result_masks > THRESHOLD

    return test_file_paths, result_masks


if __name__ == '__main__':
    test_file_paths, result_masks = k_fold()

    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    all_masks = []

    for p_mask in list(result_masks):
        p_mask = rle_encoding(p_mask)
        all_masks.append(' '.join(map(str, p_mask)))

    test_ids = [test_file_path.split('.')[0] for test_file_path in test_file_paths]

    submit = pd.DataFrame([test_ids, all_masks]).T
    submit.columns = ['id', 'rle_mask']
    submit.to_csv('submit_exp_{}_cuda_{}.csv'.format(EXP_NAME, CUDA_ID), index=False)
