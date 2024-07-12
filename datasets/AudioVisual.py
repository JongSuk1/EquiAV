# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import random
import numpy as np

import torch
import torchaudio
import torch.nn.functional
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
from torch import Tensor
from decord import gpu

# change python list to numpy array to avoid memory leak.
def pro_data(data_json):
    for i in range(len(data_json)):
        data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]
    data_np = np.array(data_json, dtype=str)
    return data_np

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

    
def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(axis_length * p))
    

def mask_along_axis(
    specgram: Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> Tensor:
    """Apply a mask along ``axis``.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Mask will be applied from indices ``[v_0, v_0 + v)``,
    where ``v`` is sampled from ``uniform(0, max_v)`` and
    ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``, with
    ``max_v = mask_param`` when ``p = 1.0`` and
    ``max_v = min(mask_param, floor(specgrams.size(axis) * p))``
    otherwise.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram `(channel, freq, time)`
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrogram of dimensions `(channel, freq, time)`
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram, 0, 0

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(0, specgram.shape[axis], device=specgram.device, dtype=specgram.dtype)
    mask = (mask >= mask_start) & (mask < mask_end)
    if axis == 1:
        mask = mask.unsqueeze(-1)

    assert mask_end - mask_start < mask_param

    specgram = specgram.masked_fill(mask, mask_value)

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram, mask_start, mask_end

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

## Main Dataset Code

class MainDataset(Dataset):
    def __init__(self, dataset_file_name, label_csv, audio_conf, **kwargs):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_file_name
        """
        self.cen_mean = audio_conf.get('cen_mean', False)
        self.cen_num = audio_conf.get('cen_num', 16)

        self.datapath = dataset_file_name

        with open(dataset_file_name, 'r') as f:
            data_json = json.load(f)

        self.data = pro_data(data_json['data']) # list to numpy to avoid memory leaks

        self.num_samples = self.data.shape[0]

        self.label_smooth = audio_conf.get('label_smooth', 0.0)
        self.melbins = audio_conf.get('nmels', 128)
        
        # for augmentation
        self.freqm = audio_conf.get('freqm', 0)
        self.timem = audio_conf.get('timem', 0)
        self.ft_freqm = audio_conf.get('ft_freqm', 0)
        self.ft_timem = audio_conf.get('ft_timem', 0)
        self.mixup = audio_conf.get('mixup', 0)
        self.noise = audio_conf.get('noise', False)

        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = audio_conf.get('mean', 0)
        self.norm_std = audio_conf.get('std', 0)
        
        self.skip_norm = audio_conf.get('skip_norm') if audio_conf.get('skip_norm') else False

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)        

        self.target_length = audio_conf.get('target_length')

        self.mode = audio_conf.get('mode', 'train')

        self.frame_use = audio_conf.get('frame_use', -1)
        self.total_frame = audio_conf.get('total_frame', 10)

        self.im_res = audio_conf.get('im_res', 224)

        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

        ## Augmentation Config for Equivariant Learning
        self.aug_size_a = audio_conf.get('aug_size_a', 0)
        self.aug_size_v = audio_conf.get('aug_size_v', 0)

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)

            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            if gpu == 0: print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_img(self, video_id, video_path):
        if self.mode == 'test':
            frame_idx = int((self.total_frame) / 2) if self.frame_use == -1 else self.frame_use

        else:
            frame_idx = random.randint(1, 10)-1

        return f'{video_path}/frame_{str(frame_idx)}/{video_id}.jpg'

    def get_raw_item(self, index):

        datum = self.data[index]
        datum = self.decode_data(datum)
        label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)

        if random.random() < self.mixup:
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda)
            except Exception as e:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
                print(e)

            try:
                image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), self.randselect_img(mix_datum['video_id'], mix_datum['video_path']), mix_lambda)
            except Exception as e:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading image')
                print(e)
            
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)
            

        else:
            try:
                fbank = self._wav2fbank(datum['wav'], None, 0)
            except Exception as e:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
                print(e)

            try:
                image = self.get_image(self.randselect_img(datum['video_id'], datum['video_path']), None, 0)
            except Exception as e:
                print(self.randselect_img(datum['video_id'], datum['video_path']))
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                raw_image = None
                print('there is an error in loading image')
                print(e)

            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        raw_image = Image.open(self.randselect_img(datum['video_id'], datum['video_path']))

        ##################### Augmented For Downstream tasks #######################
        # SpecAug, not do for eval set
        ft_freqm = torchaudio.transforms.FrequencyMasking(self.ft_freqm)
        ft_timem = torchaudio.transforms.TimeMasking(self.ft_timem)
        
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = ft_freqm(fbank)
        if self.timem != 0:
            fbank = ft_timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        ##########################################################################

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        else:
            pass

        if self.noise == True:
            # fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
        ##############################################################################

        return fbank, image, raw_image, label_indices

    def sample_t_a(self):
        # initialize transform parameters
        # aug_size_a -> 7(SA+TS), 12(SA+TS+RRC), 21(SA+TS+RRC+CJ), 23(SA+TS+RRC+CJ+GB), 24(SA+TS+RRC+CJ+GB+HF)
        t_a_list = torch.tensor([])
        for idx in range(self.cen_num):
            transform_nums = torch.zeros(self.aug_size_a) # default transform, colorjitter, grayscale, blur, flip

            min_scale = 0.08

            tm_value = torch.rand(1) * self.timem
            tm_min_value = torch.rand(1) * (1024 - tm_value)
            tm_start, tm_end = (tm_min_value.long()).squeeze(), (tm_min_value.long() + tm_value.long()).squeeze()

            fm_value = torch.rand(1) * self.freqm
            fm_min_value = torch.rand(1) * (128 - fm_value)
            fm_start, fm_end = (fm_min_value.long()).squeeze(), (fm_min_value.long() + fm_value.long()).squeeze()

            # SpecAug
            # time masking
            if self.aug_size_a >= 7:
                if random.uniform(0, 1) < 0.8:
                    transform_nums[0] = 1 # SpecAug on/off
                    transform_nums[1] = tm_start
                    transform_nums[2] = tm_end
                    transform_nums[3] = fm_start
                    transform_nums[4] = fm_end
                    # Time Shifting
                    transform_nums[5] = 1 # Time Shifting on/off
                    ts_nu = np.random.randint(-1024, 1024)
                    transform_nums[6] = ts_nu / (1024)

            # Vision Augmentation
            # Random Resized Crop
            if self.aug_size_a >= 12:
                transform_nums[7] = 1 # RRC on/off
                img = torch.rand(3,1024,128)
                i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(min_scale, 1.0), ratio=(1.0 / 10.0, 1.5 / 8.0))
                _, num_t, num_f = img.shape 
                transform_nums[8] = i / num_f # top left
                transform_nums[9] = j / num_t
                transform_nums[10] = h / num_f
                transform_nums[11] = w / num_t   
                
            # Color jitter
            if self.aug_size_a >= 21:
                if random.uniform(0, 1) < 0.8:
                    fn_idx = torch.randperm(4)
                    transform_nums[12] = 1 # CJ on/off
                    transform_nums[13:17] = fn_idx
                    _, b, c, s, h = T.ColorJitter.get_params([0.6,1.4],[0.6,1.4],[0.6,1.4],[-0.1,0.1])
                    for fn_id in fn_idx:
                        if fn_id == 0:
                            transform_nums[17] = b-1
                        elif fn_id == 1:
                            transform_nums[18] = c-1
                        elif fn_id == 2:
                            transform_nums[19] = s-1
                        elif fn_id == 3:
                            transform_nums[20] = h
                else:
                    transform_nums[13:17] = torch.arange(4)

            # Gaussian Blur
            if self.aug_size_a >= 23:
                if random.uniform(0, 1) < 0.5:
                    transform_nums[21] = 1 # GB on/off
                    sigma = random.uniform(.1, 2.)
                    transform_nums[22] = sigma

            # Horizontal Flip
            if self.aug_size_a >= 24:
                if random.uniform(0, 1) < 0.5:
                    transform_nums[23] = 1 # HF on/off    

            # Volume Jittering
            if self.aug_size_a >= 26:
                if random.uniform(0, 1) < 0.5:
                    transform_nums[24] = 1 # HF on/off    
                    transform_nums[25] = random.uniform(1.-self.vol, 1.+self.vol)

            transform_nums = transform_nums.unsqueeze(0)
            if idx == 0:
                t_a_list = transform_nums.float()
            else:
                t_a_list = torch.cat([t_a_list,transform_nums.float()],dim=0)

        return t_a_list


    def sample_t_v(self):
        # initialize transform parameters
        # aug_size_v -> 18(RRC+CJ+GB+HF+GS), 21(RRC+CJ+GB+HF+GS+FR+VF)
        t_v_list = torch.tensor([])

        for idx in range(self.cen_num):
            transform_nums = torch.zeros(self.aug_size_v) # default transform, colorjitter, grayscale, blur, flip

            min_scale = 0.08

            if self.aug_size_v >= 16:
                transform_nums[0] = 1 # RRC on/off
                img = torch.rand(3,224,224)

                i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(min_scale, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
                width, height = 224, 224
                transform_nums[1] = i / height # top left
                transform_nums[2] = j / width
                transform_nums[3] = h / height
                transform_nums[4] = w / width

                if random.uniform(0, 1) < 0.8:
                    fn_idx = torch.randperm(4)
                    transform_nums[5] = 1 # CJ on/off
                    transform_nums[6:10] = fn_idx
                    _, b, c, s, h = T.ColorJitter.get_params([0.6,1.4],[0.6,1.4],[0.6,1.4],[-0.1,0.1])
                    for fn_id in fn_idx:
                        if fn_id == 0:
                            transform_nums[10] = b-1
                        elif fn_id == 1:
                            transform_nums[11] = c-1
                        elif fn_id == 2:
                            transform_nums[12] = s-1
                        elif fn_id == 3:
                            transform_nums[13] = h
                else:
                    transform_nums[6:10] = torch.arange(4)

                # Gaussian Blur
                if random.uniform(0, 1) < 0.5:
                    transform_nums[14] = 1 # GB on/off
                    sigma = random.uniform(.1, 2.)
                    transform_nums[15] = sigma

            if self.aug_size_v >= 18:
                # if not simple:
                # Horizontal Flip
                if random.uniform(0, 1) < 0.5:
                    transform_nums[16] = 1 # HF on/off

                # Grayscale
                if random.uniform(0, 1) < 0.2:
                    transform_nums[17] = 1 # GS on/off

            if self.aug_size_v >= 21:
                # Four-fold Rotation
                if random.uniform(0, 1) < 0.5:
                    transform_nums[18] = 1 # FR on/off
                    angle_idx = random.choice((0, 1, 2, 3))
                    transform_nums[19] = angle_idx

                # Vertical Flip
                if random.uniform(0, 1) < 0.5:
                    transform_nums[20] = 1 # VF on/off
            
        
            transform_nums = transform_nums.unsqueeze(0)

            if idx == 0:
                t_v_list = transform_nums.float()
            else:
                t_v_list = torch.cat([t_v_list,transform_nums.float()],dim=0)

        return t_v_list

    def _get_transform_a(self, fbank):
        # initialize transform parameters
        # aug_size_a -> 7(SA+TS), 12(SA+TS+RRC), 21(SA+TS+RRC+CJ), 23(SA+TS+RRC+CJ+GB), 24(SA+TS+RRC+CJ+GB+HF)
        transform_nums = torch.zeros(self.aug_size_a) # default transform, colorjitter, grayscale, blur, flip
        fbank = fbank.unsqueeze(0)  # 1 T F
        # transform_nums[12:16] = torch.arange(4)
        fbank = fbank.repeat(3, 1, 1)

        min_scale = 0.08

        # SpecAug
        # time masking
        if self.aug_size_a >= 7:
            if random.uniform(0, 1) < 0.8:
                transform_nums[0] = 1 # SpecAug on/off
                fbank, tm_start, tm_end = mask_along_axis(fbank, self.timem, mask_value=0.0, axis=1)
                # frequency masking
                fbank, fm_start, fm_end = mask_along_axis(fbank, self.freqm, mask_value=0.0, axis=2)
                transform_nums[1] = tm_start
                transform_nums[2] = tm_end
                transform_nums[3] = fm_start
                transform_nums[4] = fm_end
                
                # Time Shifting
                transform_nums[5] = 1 # Time Shifting on/off
                ts_nu = np.random.randint(-self.target_length, self.target_length)
                transform_nums[6] = ts_nu / (self.target_length)

                fbank = torch.roll(fbank, ts_nu, 0)    

        # Vision Augmentation
        # Random Resized Crop
        if self.aug_size_a >= 12:
            transform_nums[7] = 1 # RRC on/off
            i, j, h, w = T.RandomResizedCrop.get_params(fbank, scale=(min_scale, 1.0), ratio=(1.0 / 10.0, 1.5 / 8.0))
            _, num_t, num_f = fbank.shape 
            transform_nums[8] = i / num_f # top left
            transform_nums[9] = j / num_t
            transform_nums[10] = h / num_f
            transform_nums[11] = w / num_t   
            fbank = F.resized_crop(fbank, i, j, h, w, size=(num_t, num_f))
            
        # Color jitter
        if self.aug_size_a >= 21:
            if random.uniform(0, 1) < 0.8:
                fn_idx = torch.randperm(4)
                transform_nums[12] = 1 # CJ on/off
                transform_nums[13:17] = fn_idx
                _, b, c, s, h = T.ColorJitter.get_params([0.6,1.4],[0.6,1.4],[0.6,1.4],[-0.1,0.1])
                for fn_id in fn_idx:
                    if fn_id == 0:
                        transform_nums[17] = b-1
                        fbank = F.adjust_brightness(fbank, b)
                    elif fn_id == 1:
                        transform_nums[18] = c-1
                        fbank = F.adjust_contrast(fbank, c)
                    elif fn_id == 2:
                        transform_nums[19] = s-1
                        fbank = F.adjust_saturation(fbank, s)
                    elif fn_id == 3:
                        transform_nums[20] = h
                        fbank = F.adjust_hue(fbank, h)
            else:
                transform_nums[13:17] = torch.arange(4)

        # Gaussian Blur
        if self.aug_size_a >= 23:
            if random.uniform(0, 1) < 0.5:
                transform_nums[21] = 1 # GB on/off
                sigma = random.uniform(.1, 2.)
                transform_nums[22] = sigma
                # fbank = fbank.filter(ImageFilter.GaussianBlur(radius=sigma))
                ks = math.ceil(sigma) * 2 + 1
                fbank = F.gaussian_blur(fbank, kernel_size=(ks, ks), sigma=sigma)

        # Horizontal Flip
        if self.aug_size_a >= 24:
            if random.uniform(0, 1) < 0.5:
                transform_nums[23] = 1 # HF on/off
                fbank = F.hflip(fbank)

        return transform_nums.float(), fbank

    def _get_transform_v(self, img):
        # initialize transform parameters
        # aug_size_v -> 18(RRC+CJ+GB+HF+GS), 21(RRC+CJ+GB+HF+GS+FR+VF)
        transform_nums = torch.zeros(self.aug_size_v) # default transform, colorjitter, grayscale, blur, flip

        min_scale = 0.08

        if self.aug_size_v >= 16:
            transform_nums[0] = 1 # RRC on/off
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(min_scale, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
            width, height = F.get_image_size(img)
            transform_nums[1] = i / height # top left
            transform_nums[2] = j / width
            transform_nums[3] = h / height
            transform_nums[4] = w / width
            img = F.resized_crop(img, i, j, h, w, size=(224,224))

            if random.uniform(0, 1) < 0.8:
                fn_idx = torch.randperm(4)
                transform_nums[5] = 1 # CJ on/off
                transform_nums[6:10] = fn_idx
                _, b, c, s, h = T.ColorJitter.get_params([0.6,1.4],[0.6,1.4],[0.6,1.4],[-0.1,0.1])
                for fn_id in fn_idx:
                    if fn_id == 0:
                        transform_nums[10] = b-1
                        img = F.adjust_brightness(img, b)
                    elif fn_id == 1:
                        transform_nums[11] = c-1
                        img = F.adjust_contrast(img, c)
                    elif fn_id == 2:
                        transform_nums[12] = s-1
                        img = F.adjust_saturation(img, s)
                    elif fn_id == 3:
                        transform_nums[13] = h
                        img = F.adjust_hue(img, h)
            else:
                transform_nums[6:10] = torch.arange(4)

            # Gaussian Blur
            if random.uniform(0, 1) < 0.5:
                transform_nums[14] = 1 # GB on/off
                sigma = random.uniform(.1, 2.)
                transform_nums[15] = sigma
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        if self.aug_size_v >= 18:
            # Horizontal Flip
            if random.uniform(0, 1) < 0.5:
                transform_nums[16] = 1 # HF on/off
                img = F.hflip(img)

            # Grayscale
            if random.uniform(0, 1) < 0.2:
                transform_nums[17] = 1 # GS on/off
                num_output_channels = F.get_image_num_channels(img)
                img = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)

        
        if self.aug_size_v >= 21:
            # Four-fold Rotation
            if random.uniform(0, 1) < 0.5:
                transform_nums[18] = 1 # FR on/off
                angle_idx = random.choice((0, 1, 2, 3))
                transform_nums[19] = angle_idx
                img = F.rotate(img, angle_idx * 90.0)
            
            # Vertical Flip
            if random.uniform(0, 1) < 0.5:
                transform_nums[20] = 1 # VF on/off
                img = F.vflip(img)

        img = F.to_tensor(img)
        img = F.normalize(img, [0.485, 0.456, 0.406] , [0.229, 0.224, 0.225])
        
        return transform_nums.float(), img

    def __getitem__(self, index):
        fbank, image, raw_img, label_indices = self.get_raw_item(index)

        t_a, fbank_aug = self._get_transform_a(fbank)
        t_v, image_aug = self._get_transform_v(raw_img)

        fbank = fbank.unsqueeze(0).repeat(3, 1, 1)

        if self.cen_mean:
            t_a_list = self.sample_t_a()
            t_v_list = self.sample_t_v()

            return fbank, image, fbank_aug, image_aug, t_a, t_v, label_indices, t_a_list, t_v_list
        else:

            return fbank, image, fbank_aug, image_aug, t_a, t_v, label_indices

  
    def __len__(self):
        return self.num_samples