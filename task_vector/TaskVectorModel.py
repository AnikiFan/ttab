import math
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from task_vector.utils import softmax_entropy
from copy import deepcopy


class TaskVectorModel(nn.Module):
    def __init__(self, model, pool_size, num_classes, batch_size, img_size, max_batch, logger=None, writer=None):
        super(TaskVectorModel, self).__init__()
        self.model = model
        self.model.train()
        self.model.requires_grad = True
        self.pool_size = pool_size
        self.base_vector = model.state_dict()
        self.pool = [
            {key, torch.zeros_like(params, requires_grad=False)}
            for key, params
            in self.base_vector.items()
            for _ in range(self.pool_size)
        ]
        self.coefficients = torch.zeros(self.pool_size, requires_grad=True)
        self.scores = torch.zeros(self.pool_size)
        self.n = 0
        self.num_classes = num_classes
        self.init_c = 0.7
        self.lr = 0.001
        self.threshold = 0.1  # 熵值小于threshold*log(num_classes)便被认为是优质的
        self.batch_pool = torch.zeros((batch_size * max_batch, *img_size))
        self.max_batch = max_batch
        self.batch_size = batch_size
        self.cur_batch = 0
        self.step = 0
        if logger is None:
            self.logger = logging.getLogger("TaskVectorModel")
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
        if writer is None:
            self.writer = SummaryWriter()
        else:
            self.writer = writer

    @torch.no_grad()
    def score(self, vector, batch):
        new_vector = deepcopy(self.base_vector)
        for key, params in vector.items():
            new_vector[key] += params
        self.model.load_state_dict(new_vector, strict=False)
        y_hat = self.model(batch)
        score = softmax_entropy(y_hat).mean(0)
        return score

    def agg_pool(self):
        if not self.n:
            return {}
        result = deepcopy(self.pool[0])
        for i in range(1, self.n):
            for key, params in self.pool[i].items():
                result[key] += self.coefficients[i] * params
        return result

    def empty_pool(self):
        vector = self.agg_pool()
        for key, params in vector.items():
            self.base_vector[key] += params.detach()
        self.n = 0

    def insert(self, vector):
        assert self.n < self.pool_size
        self.n += 1
        for key, params in vector.items():
            self.pool[self.n][key] = params
        self.coefficients[self.n] = self.init_c

    def update_pool(self, vector, batch):
        score = self.score(vector, batch)
        self.writer.add_scalar("score",score.item(), self.step)
        self.writer.flush()
        self.logger.debug(self.step)
        if score < self.threshold * math.log(self.num_classes):
            self.logger.debug(f'update sample pool with score {score.item()}')
            self.batch_pool[(self.cur_batch) * self.batch_size:(self.cur_batch + 1) * self.batch_size] = batch
            self.cur_batch += 1
            self.cur_batch %= self.max_batch
        update = False
        for i in range(self.n):
            if self.scores[i] > score:
                update = True
                break
        if not update:
            return
        if self.n == self.pool_size:
            self.logger.debug(f'empty pool')
            self.empty_pool()
        self.logger.debug(f'update vector pool')
        self.insert(vector)
        self.update_coefficients()

    def update_coefficients(self):
        optimizer = torch.optim.Adam(params=self.coefficients, lr=0.001)
        optimizer.zero_grad()
        for i in range(min(self.cur_batch, self.max_batch)):
            y_hat = self.infer(self.batch_pool[i * self.batch_size:(i + 1) * self.batch_size])
            loss = softmax_entropy(y_hat).mean(0)
            loss.backward()
            optimizer.step()

    def infer(self, batch):
        vector = deepcopy(self.base_vector)
        for key, params in self.agg_pool():
            vector[key] += params
        self.model.load_state_dict(vector, strict=False)
        y_hat = self.model(batch)
        return y_hat

    def forward(self, batch):
        self.step += 1
        self.model.load_state_dict(deepcopy(self.base_vector), strict=False)
        self.model.requires_grad = True
        y_hat = self.model(batch)
        loss = softmax_entropy(y_hat).mean(0)
        loss.backward()
        vector = {
            key: -params.grad * self.lr
            for key, params
            in self.model.named_parameters()
        }
        self.update_pool(vector, batch)
        y_hat = self.infer(batch)
        return y_hat
