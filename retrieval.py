# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
from numpy import dot
from numpy.linalg import norm
from models.pt_EquiAV import MainModel
from datasets.AudioVisual import MainDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadParameters(model, path):
    self_state = model.module.state_dict()
    loaded_state = torch.load(path)
    # loaded_state = torch.load(path, map_location=device)
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = origname.replace('__M__.','')
            if name not in self_state:
                print("{} is not in the model.".format(origname))
                continue
            else:
                print("{} is loaded in the model".format(name))
        else:
            print("{} is loaded in the model".format(name))

        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
            continue

        self_state[name].copy_(param)

# get mean
def get_sim_mat(a, b):
    B = a.shape[0]
    sim_mat = np.empty([B, B])
    for i in range(B):
        for j in range(B):
            sim_mat[i, j] = dot(a[i, :], b[j, :]) / (norm(a[i, :]) * norm(b[j, :]))
    return sim_mat

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

# direction: 'audio' means audio->visual retrieval, 'video' means visual->audio retrieval
def get_retrieval_result(audio_model, val_loader, direction='audio'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        for i, (a_input, v_input ,_,_,_,_,_ ) in enumerate(val_loader):
            audio_input, video_input = a_input.to(device), v_input.to(device)

            with autocast():
                audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                # # mean pool all patches
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)
    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    if direction == 'audio':
        # audio->visual retrieva
        sim_mat = get_sim_mat(A_a_feat, A_v_feat)
    elif direction == 'video':
        # visual->audio retrieval
        sim_mat = get_sim_mat(A_v_feat, A_a_feat)
    result = compute_metrics(sim_mat)

    r1 = result['R1']
    r5 = result['R5']
    r10 = result['R10']
    mr = result['MR']

    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

    return r1, r5, r10, mr

def eval_retrieval(model, data_list, audio_conf, label_csv, direction, batch_size=48):
    print(model)
    print(data_list)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data_list
    args.label_csv = label_csv
    args.loss_fn = torch.nn.BCELoss()

    audio_model = MainModel()

    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    loadParameters(audio_model, model)

    audio_model.eval()

    ret_data = MainDataset(dataset_file_name=data_list, label_csv=label_csv, audio_conf=audio_conf)
    val_loader = torch.utils.data.DataLoader(ret_data, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    r1, r5, r10, mr = get_retrieval_result(audio_model, val_loader, direction)
    r1, r5, r10 = round(r1,3),round(r5,3),round(r10,3)

    return r1, r5, r10, mr

#TODO
model = ''

res = []
res.append([model])
# # for audioset
for direction in ['video', 'audio']:
    #TODO
    data_list = '' # AudioSet retrieval json file path
    label_csv = '' # AudioSet label csv file path

    dataset = 'audioset'

    audio_conf = {'target_length': 1024, 'nmels': 128, 'label_smooth': 0, 'im_res': 224,'mean':-4.346,'std': 4.332, 'mode': 'test','frame_use':10}
    
    r1, r5, r10, mr = eval_retrieval(model, data_list=data_list, audio_conf=audio_conf, label_csv=label_csv, direction=direction, batch_size=50)
    if direction == 'video':
        res.append([dataset, 'video->audio', r1, r5, r10, mr])

    elif direction == 'audio':
        res.append([dataset, 'audio->video', r1, r5, r10, mr])
        
np.savetxt(f'./retrieval_result.csv', res, delimiter=',', fmt='%s')