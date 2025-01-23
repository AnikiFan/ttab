import math
import logging
import sys

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.tensorboard import SummaryWriter

from task_vector.utils import softmax_entropy,load_weight
from copy import deepcopy


class TaskVectorModel(nn.Module):
    def __init__(self, model, pool_size, num_classes, batch_size, img_size,lambda0=0.4,threshold=0.1,lr=0.001,update_iteration=20, logger=None, writer=None):
        super(TaskVectorModel, self).__init__()
        self.model = model
        self.pool_size = pool_size
        self.base_vector = {key:param.requires_grad_(False).cuda() for key,param in  model.state_dict().items()}
        self.pool = [
            {key:torch.zeros_like(params)}
            for key, params
            in self.base_vector.items()
            for _ in range(self.pool_size)
        ]
        self.lambdas = nn.Parameter(torch.zeros(self.pool_size).cuda())
        self.threshold = threshold
        self.scores = torch.ones(self.pool_size)*self.threshold*math.log(num_classes)
        self.criteria = self.threshold*math.log(num_classes)
        # 熵值小于threshold*log(num_classes)便被认为是优质的
        self.num_classes = num_classes
        self.n = 0 # 当前task vector的数量
        self.lambda0 = lambda0
        self.lr = lr
        self.batch_pool = torch.zeros((batch_size * pool_size, *img_size)).cuda()
        self.batch_size = batch_size
        self.cur_batch = 0
        self.step = 0
        self.update_iteration = update_iteration
        self.last_update = 0
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
            result[key] = result[key]*self.lambdas[0]
        for i in range(1,self.n):
            for key, params in self.pool[i].items():
                result[key] =result[key] + self.lambdas[i] * params
        return result

    def empty_pool(self):
        vector = self.agg_pool()
        for key, params in vector.items():
            self.base_vector[key] += params.detach()
        self.n = 0

    @torch.no_grad()
    def insert(self, vector):
        assert self.n < self.pool_size
        for key, params in vector.items():
            self.pool[self.n][key] = params
        self.lambdas[self.n] = self.lambda0
        self.n += 1

    def update_pool(self, vector, batch,score):
        self.writer.add_scalar("score",score.item(), self.step)
        self.writer.flush()
        update = self.n==0 or self.scores[:self.n].min()>score.item() or (self.last_update>10 and score.item()<self.criteria)
        if not update:
            self.last_update += 1
            return
        self.last_update = 0
        self.logger.debug(f'update vector pool with score {score.item()}')
        self.insert(vector)
        self.logger.debug(f'update sample pool')
        self.batch_pool[(self.cur_batch%self.pool_size) * self.batch_size:(self.cur_batch%self.pool_size + 1) * self.batch_size] = batch
        self.cur_batch += 1
        self.update_lambdas()
        if self.n == self.pool_size:
            self.logger.debug(f'empty pool')
            self.empty_pool()

    def update_lambdas(self):
        self.lambdas.requires_grad_(True)
        self.logger.debug(f'lambdas before update:{self.lambdas.detach()}')
        optimizer = torch.optim.Adam([self.lambdas], lr=self.lr)
        cnt = 0
        while True:
            for i in range(min(self.cur_batch, self.pool_size)):
                if cnt == self.update_iteration:
                    break
                vector = deepcopy(self.base_vector)
                for key, params in self.agg_pool().items():
                    vector[key] = vector[key] + params
                optimizer.zero_grad()
                y_hat = functional_call( self.model,vector,self.batch_pool[i * self.batch_size:(i + 1) * self.batch_size])
                loss = softmax_entropy(y_hat).mean(0)
                loss.backward()
                optimizer.step()
                cnt += 1
            if cnt == self.update_iteration:
                break
        self.logger.debug(f'lambdas after update:{self.lambdas.detach()}')
        self.lambdas.requires_grad_(False)

    @torch.no_grad()
    def infer(self, batch):
        vector = self.agg_pool()
        if vector:
            for key, params in self.base_vector.items():
                vector[key] = vector[key] + params
            y_hat = functional_call( self.model,vector,batch)
        else:
            y_hat = functional_call( self.model,self.base_vector,batch)
        return y_hat

    def forward(self, batch):
        self.step += 1
        vector = deepcopy(self.base_vector)
        y_hat =functional_call( self.model,{key:param.requires_grad_() for key,param in vector.items()},batch)
        loss = softmax_entropy(y_hat).mean(0)
        loss.backward()
        vector = {
            key: -params.grad * self.lr
            for key, params
            in vector.items()
        }
        self.update_pool(vector, batch,loss)
        y_hat = self.infer(batch)
        return y_hat
