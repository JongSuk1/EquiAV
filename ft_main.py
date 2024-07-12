#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import wandb
import argparse
import importlib
import numpy as np
import warnings
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import distutils

from ft_trainer import *
from utils import DistributedSamplerWrapper, WeightedRandomSampler

warnings.filterwarnings(action='ignore')

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "TrainArgs");

parser.add_argument('--gpu', type=str,   default='0',   help='gpu id to use');

## Data definition
parser.add_argument('--dataset',type=str, default="VGGSound", help='name of dataset definition');
parser.add_argument('--fold',type=str, default="1", help='name of dataset definition');
parser.add_argument("--bal", type=lambda x:bool(distutils.util.strtobool(x)),  default=False, help="weight sampling for class balance ex) 'bal'");

parser.add_argument("--num_mel_bins", type=int, default=128,    help="number of mel bins of spectrogram");

# dataset augmentations for finetuning
parser.add_argument("--mixup", type=float, default=0.5, help="how many (0-1) samples need to be mixup during training");
parser.add_argument("--noise", type=lambda x:bool(distutils.util.strtobool(x)),  default=False, help='if use balance sampling');
parser.add_argument('--ft_freqm', type=int, default=48, help='frequency mask max length');
parser.add_argument('--ft_timem', type=int, default=192, help='time mask max length');
parser.add_argument('--label_smooth', type=float, default=0.0, help='label smoothing');

## Data loader details
parser.add_argument('--batch_size', type=int,   default=16,    help='Batch size, number of speakers per batch');
parser.add_argument('--nDataLoaderThread', type=int, default=8,     help='Number of loader threads');
parser.add_argument('--checkloader', dest='checkloader', action='store_true', help='check the dataloders')

## Training details
parser.add_argument('--max_epoch', type=int,    default=50,          help='Maximum number of epochs');
parser.add_argument('--trainfunc_ft', type=str,    default="bceloss",   help='Finetuning loss function');

## Model definition
parser.add_argument('--model', type=str,   default="EquiAV_ft",   help='Name of model definition');
parser.add_argument('--inter_linear',     type=bool,  default=True,      help='Use the linear head for extracting invariant representation');
parser.add_argument('--head_type',     type=str,  default='linear', choices=['linear', 'mlp'],      help='Head type (linear or mlp)');
parser.add_argument('--head_dim',     type=int,  default=512,  help='Dimension for mlp hidden layer');
parser.add_argument('--aug_size_a', type=int,   default=21,         help='Dimension for data augmentation parameters');
parser.add_argument('--aug_size_v', type=int,   default=17,         help='Dimension for data augmentation parameters');
parser.add_argument("--drop_path", type=float, default=0.1,    help="drop_path value of the finetuning model");
parser.add_argument("--drop_out", type=float, default=0,    help="drop_out value of the finetuning model");

parser.add_argument('--freeze_base', type=bool,  default=False,       help='Freeze base network without MLP during training');
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model");
parser.add_argument('--data_aug',      type=lambda x:bool(distutils.util.strtobool(x)),  default=False,  help='Enable data_aug');


## Optimizer details
parser.add_argument('--optimizer', type=str,   default="adamw", help='sgd or adam');
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay in the optimizer');

## Learning rate details
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate');
parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights");
parser.add_argument("--start_lr", type=float, default=2e-7, help="start point of learning rate");
parser.add_argument("--final_lr", type=float, default=1e-6, help="final point of learning rate");

# Scheduler
parser.add_argument('--scheduler',      type=str,   default="warmupcos", help='Learning rate scheduler');
parser.add_argument('--warmup_epoch',      type=int,   default=3, help='warmup epoch for cosine lr scheduler');

## Load and save
parser.add_argument('--save_path', type=str, default="/exp", help='Path for model and logs');
parser.add_argument('--model_save_freq',     type=int, default=2, help='Frequency of saving model weight');

parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model weights for finetuning');

## Accelerate training
parser.add_argument('--port', type=str,default="8008", help='Port for distributed training, input as text');
parser.add_argument('--distributed',    type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='Enable distributed training');
parser.add_argument('--mixedprec',      type=lambda x:bool(distutils.util.strtobool(x)),  default=True,  help='Enable mixed precision training');

## Logging
parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging');
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity');
parser.add_argument('--wandb_name', type=str, default=None, help='wandb entity');
parser.add_argument('--wandb_project', type=str, default=None, help='wandb entity');
parser.add_argument('--print_freq', default=10, type=int, help='print frequency');

args = parser.parse_args();

args.train_list = f"/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/train.json"
args.verify_list = f"/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/test.json"
args.label_csv = f"/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/class_labels_indices.csv"

weight_file = f'/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/weights.csv' if args.bal else None

label_dim = {'AudioSet_2M':[527,'mAP'],
             'AudioSet_20K':[527,'mAP'],
             'VGGSound':[309,'acc']}

args.label_dim = label_dim[args.dataset][0]
args.main_metrics = label_dim[args.dataset][1]

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def verbose_print(gpu, print_phrase):
    if gpu == 0: print(print_phrase)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Write input hyperparameters to score file
    if args.gpu == 0:
        scorefile = open(args.result_save_path+"/scores.txt", "a+");
        scorefile.write('{} script executed\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")));
        print('\n=================args=================')
        scorefile.write('\n=================args=================\n')
        for items in vars(args):
            print(items, vars(args)[items]);
            scorefile.write('{}: {}\n'.format(items, vars(args)[items]));
        scorefile.flush()
        
    # Initialize wandb
    if args.gpu == 0 and not args.no_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, resume='allow', name=args.wandb_name)

    # Make an instance of the model
    verbose_print(args.gpu, '\n=================Define Model=================')
    model = EquiAV_ft(**vars(args));

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        
        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        model = WrappedModel(model).cuda(args.gpu)

    # Initialise dataset
    verbose_print(args.gpu, '\n=================Load Dataset=================')
    DataSet = importlib.import_module('datasets.AudioVisual').__getattribute__('MainDataset')

    norm_stats = {'AudioSet_2M':[-4.346, 4.332],'AudioSet_20K':[-4.346, 4.332],'VGGSound':[-4.956, 4.486]}   
    target_length = {'AudioSet_2M':1024,'AudioSet_20K':1024,'VGGSound':1024}
    
    audio_conf = {'target_length': target_length[args.dataset],
                        'im_res': 224, 'nmels': 128,
                        'ft_freqm': args.ft_freqm, 'ft_timem': args.ft_timem, 'mixup': args.mixup, 
                        'noise':True, 'label_smooth': args.label_smooth, 
                        'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1]}

    val_audio_conf = {'target_length': target_length[args.dataset],
                        'im_res': 224, 'nmels': 128,
                        'ft_freqm': 0, 'ft_timem': 0, 'mixup': 0, 
                        'noise':False, 'label_smooth': 0, 
                        'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
                        'mode': 'test'}
    
    train_dataset = DataSet(dataset_file_name=args.train_list, audio_conf=audio_conf, **vars(args))
    verbose_print(args.gpu, 'Load train dataset')
    val_dataset = DataSet(dataset_file_name=args.verify_list, audio_conf=val_audio_conf, **vars(args))
    verbose_print(args.gpu, 'Load test dataset')

    if args.distributed:
        if args.bal:
            verbose_print(args.gpu,'balanced sampler is being used in distributed setting')
            samples_weight = np.loadtxt(weight_file, delimiter=',')

            weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            train_sampler = DistributedSamplerWrapper(sampler=weighted_sampler, dataset=train_dataset, num_replicas=ngpus_per_node, rank=args.gpu)
        
        else:
            verbose_print(args.gpu,'balanced sampler is not used in distributed setting')
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=args.gpu, drop_last=True)
        
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, rank=args.gpu, drop_last=True)

    else:
        if args.bal:
            verbose_print(args.gpu,'balanced sampler is being used in not distributed setting')
            samples_weight = np.loadtxt(weight_file, delimiter=',')
            train_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        else:
            verbose_print(args.gpu,'balanced sampler is not used in not distributed setting')
            train_sampler = None

        val_sampler = None

    # Initialise data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        pin_memory=False,
        shuffle=(train_sampler is None),
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )
    
    args.iteration_per_epoch = len(train_loader)

    # Define the ModelTrainer
    verbose_print(args.gpu, '\n=================Parameter of the Model=================')
    trainer = ModelTrainer(model, **vars(args))

    # Start first epoch
    epoch = 1;

    # Load model parameters
    verbose_print(args.gpu, f'\n=================Load Model - pretrained model =================')
    if args.pretrained_model is not None:
        trainer.loadParameters(args.pretrained_model)
        verbose_print(args.gpu, f"Model {args.pretrained_model} loaded from previous state!");

    else: verbose_print(args.gpu, "Note you are finetuning a model without any finetuning.");

    best_metric, best_loss = -np.inf, np.inf

    # Run training
    for epoch in range(epoch,args.max_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        verbose_print(args.gpu, f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Train Epoch {epoch}');
        train_loss  = trainer.train_network(train_loader, evalmode=False, epoch=epoch);

        verbose_print(args.gpu, f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Eval Epoch {epoch}');
        with torch.no_grad():
            val_stats, val_loss = trainer.train_network(val_loader, evalmode=True, epoch=epoch);

            mAUC = np.mean([stat['auc'] for stat in val_stats])
            metric = np.mean([stat['AP'] for stat in val_stats]) if args.main_metrics == 'mAP' else val_stats[0]['acc']

            print(f"GPU: {args.gpu} / mAP: {metric} / val_loss: {val_loss}")
            
            if args.gpu == 0:
                print(f'==============EPOCH {epoch}/{args.max_epoch}=============\n')
                print(f"\n{args.main_metrics}: {metric:.6f}")
                print(f"AUC: {mAUC:.6f}")
                print(f"d_prime: {d_prime(mAUC):.6f}\n")

                print(f"train_loss: {train_loss:.6f}")
                print(f"val_loss: {val_loss:.6f}\n")

                print(time.strftime("%Y-%m-%d %H:%M:%S"), "{}: EP {:d}, TLOSS {:.5f}, VLOSS {:.5f}\n".format(args.save_path, epoch, train_loss, val_loss));
                
                scorefile.write("EP {:d}, TLOSS {:.5f}, VLOSS {:.5f}\n".format(epoch, train_loss, val_loss));
                scorefile.flush()

                print('validation finished')

                log_stats = {f'eval_{args.main_metrics}': metric,
                            'eval_mAUC': mAUC,
                            'd_prime': d_prime(mAUC),
                            'val_loss': val_loss,
                            'epoch': epoch}
                    
                if not args.no_wandb:
                    wandb.log(log_stats)

                if metric > best_metric:
                    best_metric = metric

                    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"Saving best metric model with Epoch {epoch} / {args.main_metrics} : {metric:.6f}")
                    trainer.saveParameters(args.model_save_path+f"/model_bestMetric_ft.pth");
                
                if best_loss > val_loss:
                    best_loss = val_loss
 
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"Saving best loss model with Epoch {epoch}")
                    trainer.saveParameters(args.model_save_path+f"/model_bestLoss_ft.pth");
                               
                if epoch % args.model_save_freq == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Saving model {:d}".format(epoch))
                    trainer.saveParameters(args.model_save_path+f"/model_{epoch}_ft.pth");       

    if args.gpu == 0:
        scorefile.close();


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    
    i_try = 3

    while os.path.exists(args.save_path):
        if not 'try' in args.save_path:
            args.save_path += '_try2'
        else:
            new_path = args.save_path.replace(args.save_path.split('_')[-1], f'try{i_try}')
            args.save_path = new_path
            i_try += 1

    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    # args.distributed = False
    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)

if __name__ == '__main__':
    main()