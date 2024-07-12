#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, gpu, **kwargs):

	if gpu == 0: print('Initialised Adam optimizer')

	return torch.optim.Adam(parameters, lr = lr, weight_decay = weight_decay, betas = (0.95, 0.999));