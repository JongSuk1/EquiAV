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

from pt_trainer import *

warnings.filterwarnings(action='ignore')

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "TrainArgs");

parser.add_argument('--gpu', type=str,   default='0',   help='gpu id to use');

## Data definition
parser.add_argument('--dataset',          type=str, default="AudioSet_20K", help='name of dataset definition');
parser.add_argument("--num_mel_bins", type=int, default=128,    help="number of mel bins of spectrogram");

# dataset augmentations for equivariant learning
parser.add_argument('--aug_size_a',       type=int,   default=24,   choices=[7, 12, 21, 23, 24, 26],
                    help='Dimension for data augmentation parameters / 7(SA+TS), 12(SA+TS+RRC), 21(SA+TS+RRC+CJ), 23(SA+TS+RRC+CJ+GB), 24(SA+TS+RRC+CJ+GB+HF), 24(SA+TS+RRC+CJ+GB+HF + VJ)');
parser.add_argument('--aug_size_v',       type=int,   default=18,   choices=[16, 18, 21],       
                    help='Dimension for data augmentation parameters / 16(RRC+CJ+GB), 18(RRC+CJ+GB+HF+GS), 21(RRC+CJ+GB+HF+GS+FR+VF)');

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training");
parser.add_argument('--freqm', type=int, default=48, help='frequency mask max length');
parser.add_argument('--timem', type=int, default=192, help='time mask max length');

## Data loader details
parser.add_argument('--batch_size',     type=int,   default=4,    help='Batch size, number of speakers per batch');
parser.add_argument('--nDataLoaderThread', type=int, default=8,     help='Number of loader threads');

## Training details
parser.add_argument('--max_epoch',       type=int,    default=50,          help='Maximum number of epochs');
parser.add_argument('--loss_inter',   type=str,    default="ntxent",   help='Invariant loss function');
parser.add_argument('--loss_intra',  type=str,    default="simclr",   help='Equivariant loss function');
parser.add_argument('--tau_inv',  type=float,    default=0.07,   help='temperature for invariant loss function');
parser.add_argument('--tau_equi',  type=float,    default=0.07,   help='temperature for equivariant loss function');

# Weight of Loss elements
parser.add_argument('--inter_scale',       type=float,  default=1./3.,         help='Scale for multimodal invariant embeddings');
parser.add_argument('--intra_a_scale',    type=float,  default=1./3.,         help='Scale for equivariant audio loss');
parser.add_argument('--intra_v_scale',    type=float,  default=1./3.,         help='Scale for equivariant visual loss');

## Model definition
parser.add_argument('--model',              type=str,   default="pt_EquiAV",   help='Name of model definition');
parser.add_argument('--inter_linear',     type=lambda x:bool(distutils.util.strtobool(x)),  default=False,      help='Use the linear head for extracting invariant representation');
parser.add_argument('--cen_mean',     type=lambda x:bool(distutils.util.strtobool(x)),  default=True,      help='Use the cen_mean');
parser.add_argument('--cen_num',    type=int,   default=16,      help='number of augmented for using JS mean');

parser.add_argument('--proj_hidden_dim',    type=int,   default=2048,      help='channel dimension of hidden layer of projection head');
parser.add_argument('--inter_proj_out_dim',   type=int,   default=512,      help='out channel dimension of invariant projection head');
parser.add_argument('--intra_proj_out_dim',  type=int,   default=512,      help='out channel dimension of equivariant projection head');
parser.add_argument('--aug_emb_dim',        type=int,   default=256,      help='out channel dimension of augmentation encoder');

## Optimizer details
parser.add_argument('--optimizer',      type=str,   default="adamw", help='sgd or adam');
parser.add_argument('--weight_decay',   type=float, default=1e-5,      help='Weight decay in the optimizer');

## Learning rate details
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate');
parser.add_argument("--start_lr", type=float, default=2e-7, help="start point of learning rate");
parser.add_argument("--final_lr", type=float, default=1e-6, help="final point of learning rate");

# Scheduler
parser.add_argument('--scheduler',      type=str,   default="warmupcos", help='Learning rate scheduler');
parser.add_argument('--warmup_epoch',      type=int,   default=2, help='warmup epoch for cosine lr scheduler');

## Load and save
parser.add_argument('--audio_pretrained_encoder', type=str, default='./pretrained_weights/mae_pretrain_vit_base.pth', help='pretrained model weights for visual encoders');
parser.add_argument('--visual_pretrained_encoder', type=str, default='./pretrained_weights/mae_pretrain_vit_base.pth', help='pretrained model weights for visual encoders');
parser.add_argument('--save_path',     type=str, default="/exp", help='Path for model and logs');
parser.add_argument('--model_save_freq',     type=int, default=1, help='Frequency of saving model weight');

parser.add_argument('--continue_model', type=str, default=None, help='load the model that want to train continuously');

## Accelerate training
parser.add_argument('--port',           type=str,   default="8008", help='Port for distributed training, input as text');
parser.add_argument('--distributed',    type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='Enable distributed training');
parser.add_argument('--mixedprec',      type=lambda x:bool(distutils.util.strtobool(x)),  default=True,  help='Enable mixed precision training');

## Logging
parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging');
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity');
parser.add_argument('--wandb_project', type=str, default=None, help='wandb entity');
parser.add_argument('--wandb_name', type=str, default=None, help='wandb entity');
parser.add_argument('--print_freq', default=10, type=int, help='print frequency');

args = parser.parse_args();

args.train_list = f"/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/train.json"
args.verify_list = f"/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/test.json"
args.label_csv = f"/home/lhk/workspace/ESSL/EquiAV/datasets/dataprep/{args.dataset}/class_labels_indices.csv"

label_dim = {'AudioSet_2M': 527,'AudioSet_20K':527,'VGGSound':309}
args.label_dim = label_dim[args.dataset]

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def verbose_print(gpu, print_phrase):
    if gpu == 0: print(print_phrase)

## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

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
    model = EquiAV(**vars(args));

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
                        'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 
                        'noise':True, 'label_smooth': 0, 
                        'mean':norm_stats[args.dataset][0],'std':norm_stats[args.dataset][1],
                        'cen_mean':args.cen_mean, 'cen_num':args.cen_num,
                        'aug_size_a': args.aug_size_a, 'aug_size_v': args.aug_size_v}
    
    val_audio_conf = {'target_length': target_length[args.dataset],
                        'im_res': 224, 'nmels': 128,
                        'freqm': 0, 'timem': 0, 'mixup': 0, 
                        'noise':False, 'label_smooth': 0, 
                        'mean':norm_stats[args.dataset][0],'std':norm_stats[args.dataset][1],
                        'cen_mean':args.cen_mean, 'cen_num':args.cen_num,
                        'aug_size_a': args.aug_size_a, 'aug_size_v': args.aug_size_v,
                        'mode': 'test'}

    train_dataset = DataSet(dataset_file_name=args.train_list, audio_conf=audio_conf, **vars(args))
    val_dataset = DataSet(dataset_file_name=args.verify_list, audio_conf=val_audio_conf, **vars(args))
    
    verbose_print(args.gpu, 'Load train / validation dataset')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=args.gpu, drop_last=True)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset,   rank=args.gpu, drop_last=True)

    else:
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
        drop_last=True,
    )

    args.iteration_per_epoch = len(train_loader)

    # Define the ModelTrainer
    verbose_print(args.gpu, '\n=================Parameter of the Model=================')
    trainer = ModelTrainer(model, **vars(args))
    
    # Start first epoch
    epoch = 1;

    # Load model parameters
    verbose_print(args.gpu, f'\n=================Load Model - continue model=================')
    if args.continue_model is not None:
        epoch = int(os.path.splitext(os.path.basename(args.continue_model))[0][6:])+1

        trainer.loadParameters(args.continue_model, epoch=epoch);
        verbose_print(args.gpu, f"Model {args.continue_model} loaded from previous state!");

    else: verbose_print(args.gpu, "Note you are pretraining a model from epoch 1.");


    best_loss, best_loss_val = np.inf, np.inf

    # Run training
    for epoch in range(epoch, args.max_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        verbose_print(args.gpu, f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} Train Epoch {epoch}');
        train_stats = trainer.train_network(train_loader, epoch=epoch);
        
        verbose_print(args.gpu, f'{time.strftime(f"%Y-%m-{epoch} %H:%M:%S")} Eval Epoch');
        
        with torch.no_grad():
            val_stats = trainer.train_network(val_loader, eval_mode=True, epoch=epoch);

        if args.gpu == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "{}: EP {:d}, TLOSS {:.5f}, VLOSS {:.5f}\n".format(args.save_path, epoch, train_stats['loss'], val_stats['loss']));
            
            scorefile.write("EP {:d}, TLOSS {:.5f}, VLOSS {:.5f}\n".format(epoch, train_stats['loss'], val_stats['loss']));
            scorefile.flush()

            log_stats = {**{f'test_{k}': v for k, v in val_stats.items()}, 'epoch': epoch}

            if not args.no_wandb:
                wandb.log(log_stats)
                
            trainer.saveParameters(args.model_save_path+f"/model_lastest.pth");

            if best_loss > train_stats['loss']:
                best_loss = train_stats['loss']

                # Save model
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"Saving best train model with Epoch {epoch}")
                trainer.saveParameters(args.model_save_path+f"/model_best_train_pretrained.pth");

            if best_loss_val > val_stats['loss']:
                best_loss_val = val_stats['loss']

                # Save model
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"Saving best val model with Epoch {epoch}")
                trainer.saveParameters(args.model_save_path+f"/model_best_val_pretrained.pth");


            if epoch % args.model_save_freq == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Saving model {:d}".format(epoch))
                trainer.saveParameters(args.model_save_path+f"/model_{epoch}.pth");      

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