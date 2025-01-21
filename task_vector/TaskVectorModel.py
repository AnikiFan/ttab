import math

import torch
import torch.nn as nn
from task_vector.utils import softmax_entropy
from copy import deepcopy
class TaskVectorModel(nn.Module):
    def __init__(self,model,pool_size,num_classes,img_size,max_sample):
        super(TaskVectorModel, self).__init__()
        self.model = model
        self.model.train()
        self.model.requires_grad = True
        self.pool_size = pool_size
        self.base_vector = model.state_dict()
        self.pool = [
            {key,torch.zeros_like(params,requires_grad=False)}
            for key,params
            in self.base_vector.items()
            for _ in range(self.pool_size)
        ]
        self.coefficients = torch.zeros(self.pool_size,requires_grad=True)
        self.scores = torch.zeros(self.pool_size)
        self.n = 0
        self.num_classes = num_classes
        self.init_c = 0.001
        self.threshold = 0.3 # 熵值小于threshold*log(num_classes)便被认为是优质的
        self.sample_pool = torch.zeros((max_sample,*img_size))
        self.max_sample = max_sample
        self.cur_smaple = 0

    @torch.no_grad()
    def score(self,vector,sample):
        new_vector = deepcopy(self.base_vector)
        for key,params in vector.items():
            new_vector[key] += params
        self.model.load_state_dict(new_vector,strict=False)
        y_hat = self.model(sample)
        score = softmax_entropy(y_hat)
        return score

    def agg_pool(self):
        if not self.n:
            return {}
        result = deepcopy(self.pool[0])
        for i in range(1,self.n):
            for key,params in self.pool[i].items():
                result[key] += self.coefficients[i]*params
        return result

    def empty_pool(self):
        vector = self.agg_pool()
        for key,params in vector.items():
            self.base_vector[key] += params.detach()
        self.n = 0

    def insert(self,vector):
        assert self.n < self.pool_size
        self.n += 1
        for key,params in vector.items():
            self.pool[self.n][key] = params
        self.coefficients[self.n] = self.init_c

    def update_pool(self,vector,sample):
        score = self.score(vector,sample)
        if score < self.threshold*math.log(self.num_classes):
            self.sample_pool[self.cur_smaple%self.max_sample] = sample
            self.cur_smaple += 1
        update = False
        for i in range(self.n):
            if self.scores[i]>score:
                update = True
                break
        if not update:
            return
        if self.n == self.pool_size:
            self.empty_pool()
        self.insert(vector)
        self.update_coefficients()

    def update_coefficients(self):
        optimizer = torch.optim.Adam(params=self.coefficients,lr=0.001)
        optimizer.zero_grad()
        y_hat = self.infer(self.sample_pool)
        loss = softmax_entropy(y_hat)
        loss.backward()
        optimizer.step()

    def infer(self,x):
        vector = deepcopy(self.base_vector)
        for key,params in self.agg_pool():
            vector[key] += params
        self.model.load_state_dict(vector,strict=False)
        y_hat = self.model(x)
        return y_hat

    def forward(self,x):
        self.model.load_state_dict(deepcopy(self.base_vector),strict=False)
        self.model.requires_grad = True
        y_hat = self.model(x)
        loss = softmax_entropy(y_hat)
        loss.backward()
        vector = {
            key:-params.grad
            for key,params
            in self.model.named_parameters()
        }
        self.update_pool(vector,x)
        y_hat = self.infer(x)
        return y_hat


