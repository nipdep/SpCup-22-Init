# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os
import wave
import math
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import torchaudio.functional as F
import random
import librosa
# from audiomentations import Mp3Compression, AddGaussianSNR, Compose

def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

# _SAMPLE_DIR = "_assets"
# SAMPLE_NOISE_PATH = os.path.join(_SAMPLE_DIR, "bg.wav")
SAMPLE_NOISE_PATH = './_assets/bg.wav'
def get_noise_sample(*, resample=None):
    return _get_sample(SAMPLE_NOISE_PATH, resample=resample)

def add_SNR(signal, sr, snr_db, snr_noise):
    noise = torch.zeros(signal.shape).type(torch.float32)
    if snr_noise.shape[1]> signal.shape[1]:
        snr_noise = snr_noise[:, :signal.shape[1]]
    noise[:, :snr_noise.shape[1]] = snr_noise
    signal = signal = signal.type(torch.float32)
    speech_power = signal.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = 20 #math.exp(snr_db / 10)
    scale = snr * noise_power / speech_power
    return (scale * signal + noise) / 2

# noise, _ = get_noise_sample()

def get_sample(path, noise, is_train, is_dest):
    effects = [] # [["remix", "1"]]
    # if random.choice([True, False]):
    #     effects.append(["lowpass", "-1", "300"])
    # if random.choice([True, False]):
    #     effects.append(["rate", "8000"])
    
    if is_train:
        # if random.choice([True, False]):
        #     effects.append(["speed", "0.9"])
        if random.choice([True, False]):
            effects.append(["pitch", "-10"])
        # if random.choice([True, False]):
        #     effects.append(["echo", "0.8", "0.88", "6", "0.4"])
        if random.choice([True, False]):
            effects.append(["dither", "-a"])
        if random.choice([True, False]):
            effects.append(["stretch", "1.1"])
        # if random.choice([True, False]):
        #     effects.append(["gain", "-B"])
    if is_dest:
        if random.choice([True, False]):
            effects.append(["reverb", "-w", "0.25", "0.9"])

    effects.append(["norm"])

    ad_signal, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects, normalize=False)
    if is_dest: 
        if ad_signal.shape[0] == 2:
            ad_signal = ad_signal[1, :][None, :]
        noise, _ = get_noise_sample(resample=sr)
        if random.choice([True, False]):
            ad_signal = F.apply_codec(ad_signal, sr, format= "mp3", compression=-4.5)
        if random.choice([True, False]):
            ad_signal = add_SNR(ad_signal, sr, 3, noise)

        ad_signal = ad_signal.type(torch.float32)
        ad_signal /= 21
    return ad_signal, sr

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:]-coeff*signal[:-1])


class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, is_train=False, is_dest=False):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print(
            '---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(
            self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get(
            'skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print(
                'now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(
                self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        self.noise_lvl = self.audio_conf.get('noise_level')
        if self.noise == True:
            print('now use noise augmentation')

        # self.index_dict = make_index_dict(label_csv)
        self.label_num = 6  # len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))
        noise, _ = get_noise_sample()
        self.noises = noise
        self.is_train = is_train
        self.is_dest = is_dest

    def _wav2fbank(self, filename):

        waveform, sr = get_sample(filename, self.noises, self.is_train, self.is_dest)
        waveform = waveform.type(torch.float32)
        # waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank, 0


    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        fbank, mix_lambda = self._wav2fbank(datum['wav'])
        # for label_str in datum['labels'].split(','):
        #     label_indices[int(self.index_dict[label_str])] = 1.0
        label_indices[datum['labels']] += 1.0

        label_indices = torch.FloatTensor([datum['labels']])

        # SpecAug, not do for eval set
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # if self.freqm != 0:
        #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        # if not self.skip_norm:
        #     fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # # skip normalization the input if you are trying to get the normalization stats.
        # else:
        #     pass

        # if self.noise == True:
        #     fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * (self.noise_lvl**0.5)  # Add gaussian noise with configured noise level
        #     fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)


        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)


class AudioTestDataset(Dataset):
    def __init__(self, dataset_df, audio_conf):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        self.data = [i[0] for i in list(dataset_df.loc[:, ['track']].values)]
        self.audio_conf = audio_conf
        print(
            '---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(
            self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get(
            'skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print(
                'now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(
                self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        self.noise_lvl = self.audio_conf.get('noise_level')
        if self.noise == True:
            print('now use noise augmentation')

        # self.index_dict = make_index_dict(label_csv)
        self.label_num = 6  # len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

    def _wav2fbank(self, filename):
        waveform, sr = get_sample(filename, "", False, False)
        waveform = waveform.type(torch.float32)
        # waveform = waveform - waveform.mean()


        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank, 0


    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        datum = self.data[index]
        # label_indices = np.zeros(self.label_num)
        fbank, mix_lambda = self._wav2fbank(datum)


        # SpecAug, not do for eval set
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # if self.freqm != 0:
        #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        # if not self.skip_norm:
        #     fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # # skip normalization the input if you are trying to get the normalization stats.
        # else:
        #     pass

        # if self.noise == True:
        #     fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * (self.noise_lvl**0.5)  # Add gaussian noise with configured noise level
        #     fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank #, label_indices

    def __len__(self):
        return len(self.data)
