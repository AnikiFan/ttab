import math
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from task_vector.utils import softmax_entropy,load_weight
from copy import deepcopy


class TaskVectorModel(nn.Module):
    def __init__(self, model, pool_size, num_classes, batch_size, img_size, logger=None, writer=None):
        super(TaskVectorModel, self).__init__()
        self.model = model
        self.model.train()
        self.model.requires_grad = False
        self.pool_size = pool_size
        self.base_vector = dict(model.named_parameters())
        self.pool = [
            {key:torch.zeros_like(params, requires_grad=False)}
            for key, params
            in self.base_vector.items()
            for _ in range(self.pool_size)
        ]
        self.coefficients = nn.Parameter(torch.zeros(self.pool_size),requires_grad=False).cuda()
        self.scores = torch.ones(self.pool_size)*0.1*math.log(num_classes)
        self.n = 0
        self.num_classes = num_classes
        self.init_c = 0.7
        self.lr = 0.001
        # self.threshold = 0.1  # 熵值小于threshold*log(num_classes)便被认为是优质的
        self.batch_pool = torch.zeros((batch_size * pool_size, *img_size)).cuda()
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

    def agg_pool(self):
        if not self.n:
            return {}
        result = deepcopy(self.pool[0])
        for key in result.keys():
            result[key] = result[key]*self.coefficients[0]
        for i in range(1,self.n):
            for key, params in self.pool[i].items():
                result[key] =result[key] + self.coefficients[i] * params
        return result

    def empty_pool(self):
        vector = self.agg_pool()
        for key, params in vector.items():
            self.base_vector[key] += params.detach()
        self.n = 0

    def insert(self, vector):
        assert self.n < self.pool_size
        for key, params in vector.items():
            self.pool[self.n][key] = nn.Parameter(params)
        self.coefficients[self.n] = self.init_c
        self.n += 1

    def update_pool(self, vector, batch,score):
        self.writer.add_scalar("score",score.item(), self.step)
        self.writer.flush()
        update = False
        for i in range(self.n):
            if self.scores[i] > score:
                update = True
                break
        if not update and self.n != 0:
            return
        if self.n == self.pool_size:
            self.logger.debug(f'empty pool')
            self.empty_pool()
        self.logger.debug(f'update sample pool with score {score.item()}')
        self.batch_pool[(self.cur_batch%self.pool_size) * self.batch_size:(self.cur_batch%self.pool_size + 1) * self.batch_size] = batch
        self.cur_batch += 1
        self.logger.debug(f'update vector pool')
        self.insert(vector)
        self.update_coefficients()

    def update_coefficients(self):
        self.coefficients.requires_grad_(True)
        self.logger.debug(f'coefficients before update:{self.coefficients.detach()}')
        vector = deepcopy(self.base_vector)
        for key, params in self.agg_pool().items():
            vector[key] = nn.Parameter(vector[key] + params)
        load_weight(vector,self.model)
        optimizer = torch.optim.Adam([self.coefficients], lr=0.001)
        for i in range(min(self.cur_batch, self.pool_size)):
            optimizer.zero_grad()
            y_hat = self.model(self.batch_pool[i * self.batch_size:(i + 1) * self.batch_size])
            loss = softmax_entropy(y_hat).mean(0)
            loss.backward()
            optimizer.step()
        self.logger.debug(f'coefficients after update:{self.coefficients.detach()}')
        self.coefficients.requires_grad_(False)

    def infer(self, batch):
        vector = deepcopy(self.base_vector)
        for key, params in self.agg_pool().items():
            vector[key] = nn.Parameter(vector[key] + params)
        load_weight(vector,self.model)
        y_hat = self.model(batch)
        return y_hat

    def forward(self, batch):
        self.step += 1
        load_weight(deepcopy(self.base_vector),self.model)
        self.model.requires_grad = True
        y_hat = self.model(batch)
        loss = softmax_entropy(y_hat).mean(0)
        loss.backward()
        vector = {
            key: -params.grad * self.lr
            for key, params
            in self.model.named_parameters()
        }
        self.update_pool(vector, batch,loss)
        y_hat = self.infer(batch)
        return y_hat
