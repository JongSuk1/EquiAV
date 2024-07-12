#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
	def __init__(self, gpu, **kwargs):
		super(LossFunction, self).__init__()
		self.criterion  = nn.BCEWithLogitsLoss().cuda()
		if gpu == 0:
			print('Initialised Cross Entropy Loss')

	def forward(self, x, label=None):

		nloss   = self.criterion(x, label)

		return nloss