#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class LossFunction(nn.Module):
	def __init__(self, gpu, **kwargs):
		super(LossFunction, self).__init__()
		self.labels = None
		self.last_local_batch_size = None
		self.tau = nn.Parameter(data=torch.tensor(0.07), requires_grad=True) if kwargs['tau_inv'] == 0 else kwargs['tau_inv']

		if gpu == 0:
			print('Initialised NTXent Loss')

	def forward(self, a_inv, v_inv):
		# tau = 0.05
		local_batch_size = a_inv.size(0)

		if local_batch_size != self.last_local_batch_size:
			self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=a_inv.device
            )
			self.last_local_batch_size = local_batch_size

        # normalized features
		a_inv = F.normalize(a_inv, dim=-1, p=2)
		v_inv = F.normalize(v_inv, dim=-1, p=2)

        # gather features from all GPUs
		a_inv_all, v_inv_all = \
			utils.all_gather_batch([a_inv, v_inv])

        # cosine similarity as logits
		logits_per_audio = a_inv @ v_inv_all.t() / self.tau
		logits_per_video = v_inv @ a_inv_all.t() / self.tau

		loss = (F.cross_entropy(logits_per_audio, self.labels) + \
            F.cross_entropy(logits_per_video, self.labels)) / 2

        # compute accuracy
		with torch.no_grad():
			pred = torch.argmax(logits_per_audio, dim=-1)
			correct = pred.eq(self.labels).sum()
			acc = 100 * correct / local_batch_size

		return {'loss': loss, 'inv_loss': loss, 'inv_acc': acc, 'acc': acc}
	
	
	