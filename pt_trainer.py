#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import wandb
import importlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from collections import OrderedDict
from utils import *

class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, label=None, t_a_list=None, t_v_list=None):
        return self.module(data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, label,t_a_list,t_v_list)

class EquiAV(nn.Module):
    def __init__(self, loss_inter, loss_intra, inter_scale, intra_a_scale, intra_v_scale, model, audio_pretrained_encoder=None, visual_pretrained_encoder=None, **kwargs):
        super(EquiAV, self).__init__();

        self.model = model
        self.gpu = kwargs['gpu']

        EquiAV_Model = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__M__ = EquiAV_Model(**kwargs);

        LossFunction_inv = importlib.import_module('loss.'+loss_inter).__getattribute__('LossFunction')
        self.__L_inter__ = LossFunction_inv(**kwargs);

        LossFunction_equi = importlib.import_module('loss.'+loss_intra).__getattribute__('LossFunction')
        self.__L_intra__ = LossFunction_equi(**kwargs);

        self.inter_scale = inter_scale
        self.intra_a_scale = intra_a_scale
        self.intra_v_scale = intra_v_scale

        self.cen_mean = kwargs['cen_mean']

        if audio_pretrained_encoder is not None:
            if os.path.exists(audio_pretrained_encoder):
                self.initialize(audio_pretrained_encoder,mode='audio')
            else:
                if self.gpu == 0: print(f"Model weight path for Audio <{audio_pretrained_encoder}> doesn't exist!")
        else:
            if self.gpu == 0:  print("Scratch Audio Backbone")
                
        if visual_pretrained_encoder is not None:
            if os.path.exists(visual_pretrained_encoder):
                self.initialize(visual_pretrained_encoder,mode='visual')
            else:
                if self.gpu == 0: print(f"Model weight path for Visual <{visual_pretrained_encoder}> doesn't exist!")
        else:
            if self.gpu == 0:  print("Scratch Visual Backbone")

    def initialize(self, weight_path, mode='visual'):
        weights = torch.load(weight_path, map_location=torch.device('cpu'))
        dst_params = {name: p for name, p in self.__M__.named_parameters()}

        for name, p in weights['model'].items():
            keyword = name.split('.')[0]

            if mode == 'audio':
                dst_name = name.replace(keyword, f"{keyword}_a")
            elif mode == 'visual':
                dst_name = name.replace(keyword, f"{keyword}_v")

            if dst_name in dst_params and p.shape == dst_params[dst_name].shape:
                # if self.gpu == 0: print(f"Successfully Loaded: {mode} - {dst_name}")
                with torch.no_grad():
                    dst_params[dst_name].copy_(p)
            elif dst_name not in dst_params:
                if self.gpu == 0: print(f"Missing Block: {mode} - {dst_name} is not in the our Model")
            elif p.shape != dst_params[dst_name].shape:
                if 'patch_embed_a' in dst_name:
                    with torch.no_grad():
                        dst_params[dst_name].copy_(p.repeat(1,3,1,1))
                    if self.gpu == 0: print(f"Parameters channel repeat: {mode} - {name}/ {p.repeat(1,3,1,1).shape} => {dst_name} / {dst_params[dst_name].shape}")
                else:
                    if self.gpu == 0: print(f"Wrong parameters: {mode} - {name}/ {p.shape} != {dst_name}/ {dst_params[dst_name].shape}")
            else:
                if self.gpu == 0: print(f"Missing Block: {mode} - {name} is not exist and wrong params")

    def forward(self, data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, label=None, t_a_list=None, t_v_list=None):
        if self.cen_mean:
            z_a_inv, z_v_inv, z_a_equi_t, z_v_equi_t, z_a_aug, z_v_aug = self.__M__.forward(data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, t_a_list, t_v_list)

        else: 
            z_a_inv, z_v_inv, z_a_equi_t, z_v_equi_t, z_a_aug, z_v_aug = self.__M__.forward(data_a, data_v, data_a_aug, data_v_aug, t_a, t_v)
        
        loss_dict_inter = self.__L_inter__.forward(z_a_inv, z_v_inv)
        loss_dict_intra_a = self.__L_intra__.forward(z_a_equi_t, z_a_aug)
        loss_dict_intra_v = self.__L_intra__.forward(z_v_equi_t, z_v_aug)

        loss_inter = loss_dict_inter['loss']
        acc_inter = loss_dict_inter['acc']
        loss_intra_a = loss_dict_intra_a['loss']
        acc_intra_a = loss_dict_intra_a['acc']
        loss_intra_v = loss_dict_intra_v['loss']
        acc_intra_v = loss_dict_intra_v['acc']

        nloss = self.inter_scale * loss_inter + self.intra_a_scale * loss_intra_a + self.intra_v_scale * loss_intra_v
        
        loss_dict = {'loss': nloss,
                    'loss_inter': loss_inter,
                    'loss_intra_a': loss_intra_a,
                    'loss_intra_v': loss_intra_v,
                    'acc_inv': acc_inter,
                    'acc_intra_a': acc_intra_a,
                    'acc_intra_v': acc_intra_v
                }
        
        return loss_dict


class ModelTrainer(nn.Module):

    def __init__(self, equiAV, gpu, optimizer, mixedprec, no_wandb=False, **kwargs):
        super(ModelTrainer, self).__init__()
 
        self.__model__  = equiAV
        self.gpu = gpu
        self.mixedprec = mixedprec

        self.max_epoch = kwargs['max_epoch']
        
        self.lr = kwargs['lr']
        self.scheduler = kwargs['scheduler']
        self.warmup_epoch = kwargs['warmup_epoch']
        self.ipe = kwargs['iteration_per_epoch']
        self.ipe_scale = 1.0
        self.cen_mean = kwargs["cen_mean"]

        trainables = [p for p in self.__model__.parameters() if p.requires_grad]

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), gpu=gpu, **kwargs)

        init_lr = self.__optimizer__.param_groups[0]['lr']

        if self.gpu == 0:
            print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in self.__model__.parameters()) / 1e6))
            print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
            print(f'Initial lr: {init_lr}')

        Scheduler = importlib.import_module('scheduler.'+self.scheduler).__getattribute__('Scheduler')
        self.__scheduler__ = Scheduler(self.__optimizer__, warmup_steps=int(self.warmup_epoch*self.ipe), ref_lr=self.lr, T_max=int(self.ipe_scale*self.max_epoch*self.ipe), **kwargs)

        self.scaler = GradScaler(init_scale=65536.0, growth_factor=2, backoff_factor=0.5, growth_interval=2000) if self.mixedprec else None

        # logging
        self.no_wandb = no_wandb
        self.print_freq = kwargs['print_freq']
        self.result_save_path = kwargs['result_save_path']

    def train_network(self, loader=None, eval_mode=False, epoch=-1):
        # Setting for the logging
        batch_time = AverageMeter('Time', ':6.2f')
        data_time = AverageMeter('Data', ':6.2f')
        mem = AverageMeter('Mem (GB)', ':6.1f')
        metric_names = ['loss', 'loss_inter', 'loss_intra_a', 'loss_intra_v', 'acc_inv', 'acc_intra_a', 'acc_intra_v']
        metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])

        progress = ProgressMeter(
            len(loader),
            # [batch_time, data_time, mem, *metrics.values(), *tau_metrics.values()],
            [batch_time, data_time, mem, *metrics.values()],
            prefix="Epoch: [{}]".format(epoch))

        # number of model parameters
        param_num = 0
        for p in self.__model__.parameters():
            param_num += p.numel()

        if eval_mode:
            self.__model__.eval();
        else:
            self.__model__.train();

        data_iter = 0
        end = time.time()
        for data in loader:
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cen_mean:
                data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, _, t_a_list, t_v_list = data
                t_a_list = t_a_list.cuda(self.gpu)
                t_v_list = t_v_list.cuda(self.gpu)  
            else:
                data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, _ = data

            # transform input to torch cuda tensor
            data_a = data_a.cuda(self.gpu)          # batch x target_length x num melbins
            data_v = data_v.cuda(self.gpu)          # batch x channel x width x height
            data_a_aug = data_a_aug.cuda(self.gpu)
            data_v_aug = data_v_aug.cuda(self.gpu)  # batch x channel x width x height
            t_a = t_a.cuda(self.gpu)
            t_v = t_v.cuda(self.gpu)

            # ==================== FORWARD PASS ====================
            with autocast(enabled=self.mixedprec):
                if self.cen_mean:
                    loss_dict = self.__model__(data_a, data_v, data_a_aug, data_v_aug, t_a, t_v, None, t_a_list, t_v_list)
                else:
                    loss_dict = self.__model__(data_a, data_v, data_a_aug, data_v_aug, t_a, t_v)

            if not eval_mode:
                _new_lr = self.__scheduler__.step()
                
                if self.mixedprec:
                    # mixed precision
                    self.scaler.scale(loss_dict['loss']).backward();

                    self.scaler.unscale_(self.__optimizer__)
                    torch.nn.utils.clip_grad_norm_(self.__model__.parameters(), 0.01)

                    self.scaler.step(self.__optimizer__);
                    self.scaler.update();       
                else:
                    # single precision
                    loss_dict['loss'].backward()
                    self.__optimizer__.step();
            
                self.zero_grad();
            
            for k in loss_dict:
                metrics[k].update(loss_dict[k], loader.batch_size)

            # measure elapsed time and memory
            batch_time.update(time.time() - end)
            end = time.time()
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # logging
            if data_iter % self.print_freq == 0:
                param_sum = 0
                for p in self.__model__.parameters():
                    param_sum += torch.pow(p.detach(),2).sum()
                param_avg = torch.sqrt(param_sum) / param_num
                
                if self.gpu == 0:
                    if not self.no_wandb and not eval_mode:
                        wandb.log({**{f'train_{k}': v.item() for k, v in loss_dict.items()},
                                    'scaler': self.scaler.get_scale() if self.mixedprec else 0,
                                    'lr': _new_lr,
                                    'param_avg': param_avg,
                        })
                    log_info = progress.display(data_iter)

                    with open(os.path.join(self.result_save_path, 'log.txt'), 'a') as f:
                        f.write('\t'.join(log_info) + '\n')
            
            data_iter += 1

        sys.stdout.write("\n");
        progress.synchronize()
        
        return {
            **{k: v.avg for k, v in metrics.items()}
        }


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path, epoch):
        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = origname.replace('__M__.','');
                if name not in self_state:
                    if self.gpu == 0: print("{} is not in the model.".format(origname));
                    continue;
                else:
                    if self.gpu == 0: print("{} is loaded in the model".format(name));
            else:
                if self.gpu == 0: print("{} is loaded in the model".format(name));

            if self_state[name].size() != loaded_state[origname].size():
                if self.gpu == 0: print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

        for _ in range((epoch-1)*self.ipe):
            self.__scheduler__.step()