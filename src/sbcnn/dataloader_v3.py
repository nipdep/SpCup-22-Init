import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from audiomentations import AddGaussianSNR
from pedalboard import Pedalboard, Reverb
class SpectTestDataset:

    def __init__(self):
        self.AUTOTUNE = tf.data.AUTOTUNE
        # self.add_noise = add_noise
        # self.reverb = reverb
    def decode_audio(self, audio_binary):
        # Decode WAV-encoded audio files to `float32` tensors, normalized
        # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
        audio, _ = tf.audio.decode_wav(contents=audio_binary)
        # Since all the data is single channel (mono), drop the `channels`
        # axis from the array.
        return tf.squeeze(audio, axis=-1)

    def get_waveform(self, file_path):
        cpath = self.DATA_PATH + os.sep + file_path
        audio_binary = tf.io.read_file(cpath)
        waveform = self.decode_audio(audio_binary)
        
        return waveform

    def waveform_mapper_v1(self, ds):
        return ds.map(
            # map_func=lambda x: tf.py_function(func=self.get_waveform, inp=[x], Tout=(tf.float32, tf.int64)),
            # map_func=lambda x,y: (tf.py_function(self.get_waveform, [x], tf.float32), y),
            map_func=lambda x: (self.get_waveform(x)),
            num_parallel_calls=self.AUTOTUNE)

    def get_spectrogram(self, waveform):
        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = self.sptr_len
        waveform = waveform[:input_len]

        zero_padding = tf.zeros(
            [input_len] - tf.shape(waveform),
            dtype=tf.float32)    
        
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Concatenate the waveform with `zero_padding`, which ensures all audio
        # clips are of the same length.
        equal_length = tf.concat([waveform, zero_padding], 0)
        # Convert the waveform to a spectrogram via a STFT.
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        # Obtain the magnitude of the STFT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def spectrogram_mapper(self, ds):
        return ds.map(
            # map_func=lambda x: tf.py_function(func=self.get_spectrogram, inp=[x], Tout=(tf.float32, tf.int64)),
            map_func=lambda x: (tf.py_function(self.get_spectrogram, [x], tf.float32)),
            # map_func=lambda x, y: (self.get_spectrogram(x), y),
            num_parallel_calls=self.AUTOTUNE)

    def load_dataset(self, path):
        label_df = pd.read_csv(path)
        X = label_df['track'].values
        X = tf.convert_to_tensor(X)

        primary_ds = tf.data.Dataset.from_tensor_slices(X)
        # prm_val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        return primary_ds

    def call(self, data_path, label_path, sptr_len=16000, BUFFER_SIZE=32000, BATCH_SIZE=32, is_cache=True, is_prefetch=True):
        self.sptr_len = sptr_len
        self.DATA_PATH = data_path
        test_set= self.load_dataset(label_path)

        test_ds = self.waveform_mapper_v1(
               test_set)
        test_ds = self.spectrogram_mapper(test_ds)

        test_dataset = test_ds.batch(
             BATCH_SIZE, drop_remainder=False)

        if is_cache:
            test_dataset = test_dataset.cache()

        if is_prefetch:
            test_dataset = test_dataset.prefetch(
                self.AUTOTUNE)

        return test_dataset, test_set
