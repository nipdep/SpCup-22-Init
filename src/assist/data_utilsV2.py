import os
import math
import random

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio.functional as F
import torchaudio

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

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

    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / speech_power
    return (scale * signal + noise) / 2

# noise, _ = get_noise_sample()

def get_sample(path, noise):
    effects = [["remix", "1"]]
    if random.choice([True, False]):
        effects.append(["lowpass", "-1", "300"])
    if random.choice([True, False]):
        effects.append(["rate", "8000"])
    if random.choice([True, False]):
        effects.append(["speed", "0.8"])
    if random.choice([True, False]):
        effects.append(["reverb", "-w"])

    ad_signal, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects, normalize=False)
    if ad_signal.shape[0] == 2:
        ad_signal = ad_signal[1, :][None, :]
    noise, _ = get_noise_sample(resample=sr)
    if random.choice([True, False]):
        ad_signal = F.apply_codec(ad_signal, sr, format= "mp3", compression=-9)
    if random.choice([True, False]):
        ad_signal = add_SNR(ad_signal, sr, 3, noise)
    return ad_signal, sr

def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        noise, _ = get_noise_sample()
        self.noise = noise
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = get_sample(f"{self.base_dir}/{key}", self.noise)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[index]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
