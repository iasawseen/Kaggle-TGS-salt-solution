import torch
import time
import torch.nn as nn
import numpy as np
from logger import Logger
from cyclir_lr import CosineWithRestarts
from metrics import BinaryAccuracy


class Trainer:
    def __init__(self, config, model, loss, loss_weights,
                 metrics, data_loaders):
        self.config = config
        self.cuda_index = self.config.cuda_index
        self.model = model

        self.model.cuda(self.cuda_index)

        self.criterion = loss
        self.small_criterion = nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights
        self.metrics = metrics
        self.data_loaders = data_loaders
        self.data_loader_train = data_loaders.train
        self.data_loader_val = data_loaders.val
        self.data_loader_test = data_loaders.test
        self.model_pattern = None

        self.optimizer = None
        self.scheduler = None

        self.class_loss = nn.BCEWithLogitsLoss()

        self._init_optimizer(self.config.lr)

        self.logger = Logger(self.config.logs_dir)
        self.train_iteration = 0
        self.epoch_iteration = 0
        self.encoder_frozen = True
        self.best_val_loss = +np.inf
        self.best_val_metric = 0
        self.best_model_path = None

    def _init_optimizer(self, init_lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=init_lr,
                                         momentum=self.config.momentum,
                                         weight_decay=0.0001)

        self.scheduler = CosineWithRestarts(self.optimizer,
                                            t_max=self.config.cycle_length,
                                            eta_min=self.config.min_lr)

    def _train_per_epoch(self):
        losses = []

        self.model.set_training(True)
        if self.loss_weights is not None and self.epoch_iteration < len(self.loss_weights):
            self.criterion.set_weights(self.loss_weights[self.epoch_iteration])

        metrics = {metric_name: self.metrics[metric_name]() for metric_name in self.metrics}
        class_metric = BinaryAccuracy()

        for sample_batch in self.data_loader_train:
            x, y, y_class, y_64, y_32, y_16, y_8 = \
                sample_batch['x'], sample_batch['y'], sample_batch['y_class'],\
                sample_batch['y_64'], sample_batch['y_32'], sample_batch['y_16'], sample_batch['y_8']

            y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8 = self.model(x.cuda(self.cuda_index))

            self.optimizer.zero_grad()

            loss = self.calc_loss(y, y_class, y_64, y_32, y_16, y_8,
                                  y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8)

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            y_pred = torch.sigmoid(y_pred)
            y_pred_class = torch.sigmoid(y_pred_class)

            for metric_name in metrics:
                metrics[metric_name].update(y_pred, y.cuda(self.cuda_index))

            class_metric.update(y_pred_class, y_class.cuda(self.cuda_index))

            self.train_iteration += 1

        train_loss = sum(losses) / len(losses)
        self.logger.scalar_summary('train_loss', train_loss, self.epoch_iteration)

        metrics['class_binary_accuracy'] = class_metric

        for metric_name in metrics:
            metrics[metric_name] = metrics[metric_name].compute()
            self.logger.scalar_summary('tr_{}'.format(metric_name),
                                       metrics[metric_name], self.epoch_iteration)

    def _validate(self, val_type='val'):
        losses = []

        self.model.set_training(False)

        if val_type == 'val':
            dataloader = self.data_loader_val
        elif val_type == 'test':
            dataloader = self.data_loader_test

        metrics = {metric_name: self.metrics[metric_name]() for metric_name in self.metrics}
        class_metric = BinaryAccuracy()

        with torch.no_grad():
            for sample_batch in dataloader:
                x, y, y_class, y_64, y_32, y_16, y_8 = \
                    sample_batch['x'], sample_batch['y'], sample_batch['y_class'], \
                    sample_batch['y_64'], sample_batch['y_32'], sample_batch['y_16'], sample_batch['y_8']

                y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8 = self.model(x.cuda(self.cuda_index))

                loss = self.calc_loss(y, y_class, y_64, y_32, y_16, y_8,
                                      y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8)

                losses.append(loss.item())

                y_pred = torch.sigmoid(y_pred)
                y_pred_class = torch.sigmoid(y_pred_class)

                for metric_name in metrics:
                    metrics[metric_name].update(y_pred, y.cuda(self.cuda_index))

                class_metric.update(y_pred_class, y_class.cuda(self.cuda_index))

                self.train_iteration += 1

        val_loss = sum(losses) / len(losses)

        self.logger.scalar_summary('{}_loss'.format(val_type), val_loss, self.epoch_iteration)

        metrics['class_binary_accuracy'] = class_metric

        for metric_name in metrics:
            metrics[metric_name] = metrics[metric_name].compute()
            self.logger.scalar_summary('{}_{}'.format(val_type, metric_name),
                                       metrics[metric_name], self.epoch_iteration)

        val_metric = metrics[self.config.val_metric_criterion]

        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_model_path = self.model_pattern.format(self.epoch_iteration, val_metric)
            self.save_checkpoint(self.best_model_path)

        return val_metric

    def calc_loss(self,
                  y, y_class, y_64, y_32, y_16, y_8,
                  y_pred, y_pred_class, y_pred_64, y_pred_32, y_pred_16, y_pred_8):

        y_64_cuda = y_64.cuda(self.cuda_index)
        y_64_cuda_class = torch.sum(y_64_cuda, dim=(1, 2, 3)) > 0

        y_32_cuda = y_32.cuda(self.cuda_index)
        y_32_cuda_class = torch.sum(y_32_cuda, dim=(1, 2, 3)) > 0

        y_16_cuda = y_16.cuda(self.cuda_index)
        y_16_cuda_class = torch.sum(y_16_cuda, dim=(1, 2, 3)) > 0

        y_8_cuda = y_8.cuda(self.cuda_index)
        y_8_cuda_class = torch.sum(y_8_cuda, dim=(1, 2, 3)) > 0

        loss = self.criterion(y_pred, y.cuda(self.cuda_index))
        loss += self.config.masks_weight * self.criterion(y_pred_64[y_64_cuda_class], y_64_cuda[y_64_cuda_class])
        loss += self.config.masks_weight * self.criterion(y_pred_32[y_32_cuda_class], y_32_cuda[y_32_cuda_class])
        loss += self.config.masks_weight * self.criterion(y_pred_16[y_16_cuda_class], y_16_cuda[y_16_cuda_class])
        loss += self.config.masks_weight * self.criterion(y_pred_8[y_8_cuda_class], y_8_cuda[[y_8_cuda_class]])
        loss += self.config.class_weight * self.class_loss(y_pred_class, y_class.cuda(self.cuda_index))

        return loss

    def save_checkpoint(self, checkpoint_path, optimizer=False):
        if optimizer:
            state = {'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict()}
        else:
            state = {'state_dict': self.model.state_dict()}

        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, optimizer=False):
        state = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state['state_dict'])

        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])

    def train(self, num_epochs, model_pattern):
        self.model_pattern = model_pattern
        for i in range(num_epochs):
            init_time = time.time()

            cur_lr = self.scheduler.get_lr()[0]
            print('cur_lr: {:.4f}'.format(cur_lr))
            if cur_lr == self.config.lr and self.epoch_iteration > (self.config.bce_epochs + self.config.intermediate_epochs):
                print('reinit best val metric')
                self.best_val_metric = 0

            self._train_per_epoch()
            self._validate('val')

            if self.epoch_iteration == (self.config.bce_epochs + self.config.intermediate_epochs):
                print('reinit optimizer')
                self._init_optimizer(self.config.tune_lr)

            if self.epoch_iteration > (self.config.bce_epochs + self.config.intermediate_epochs):
                self.scheduler.step()

            self.epoch_iteration += 1
            print('elapsed time per epoch: {:.1f} s'.format(time.time() - init_time))
            print()

        self.load_checkpoint(self.best_model_path)
        self._validate('test')

        return self.model
