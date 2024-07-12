#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, gpu, weight_decay=1e-5, **kwargs):

	if gpu == 0: print('Initialised AdamW optimizer')

	return torch.optim.AdamW(parameters, lr = lr, weight_decay = weight_decay, betas = (0.9, 0.95));