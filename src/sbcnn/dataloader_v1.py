import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split


class SpectDataset:

    def __init__(self, add_noise=False):
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.add_noise = add_noise

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
            map_func=lambda x, y: (self.get_waveform(x), y),
            num_parallel_calls=self.AUTOTUNE)

    def waveform_mapper_v2(self, ds):
        n_label = 5
        return ds.map(
            # map_func=lambda x: tf.py_function(func=self.get_waveform, inp=[x], Tout=(tf.float32, tf.int64)),
            # map_func=lambda x,y: (tf.py_function(self.get_waveform, [x], tf.float32), y),
            map_func=lambda x, y: (
                self.get_waveform(x), tf.one_hot(y, n_label)),
            num_parallel_calls=self.AUTOTUNE)

    def get_spectrogram(self, waveform):
        # Zero-padding for an audio waveform with less than 16,000 samples.
        input_len = self.sptr_len
        waveform = waveform[:input_len]
        if self.add_noise:
            noise = tf.random.normal(tf.shape(waveform), 0, 0.1)
            waveform = tf.math.add(waveform, noise)
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
            # map_func=lambda x,y: (tf.py_function(self.get_spectrogram, [x], tf.float32), y),
            map_func=lambda x, y: (self.get_spectrogram(x), y),
            num_parallel_calls=self.AUTOTUNE)

    def load_dataset(self, path, split_ratio):
        label_df = pd.read_csv(path)
        X, y = label_df['track'].values, label_df['algorithm'].values
        # stratified split dataset into train-validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=split_ratio, shuffle=True)

        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)
        X_test = tf.convert_to_tensor(X_test)
        y_test = tf.convert_to_tensor(y_test)

        primary_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        prm_val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        return primary_ds, prm_val_ds

    def call(self, data_path, label_path, sptr_len=16000, BUFFER_SIZE=32000, BATCH_SIZE=32, split_raio=0.2, is_cache=True, is_prefetch=True, one_hot=False):
        self.sptr_len = sptr_len
        self.DATA_PATH = data_path
        train_ds, val_ds = self.load_dataset(label_path, split_raio)
        if not(one_hot):
            train_ds, val_ds = self.waveform_mapper_v1(
                train_ds), self.waveform_mapper_v1(val_ds)
        else:
            train_ds, val_ds = self.waveform_mapper_v2(
                train_ds), self.waveform_mapper_v2(val_ds)
        train_ds, val_ds = self.spectrogram_mapper(
            train_ds), self.spectrogram_mapper(val_ds)

        train_dataset = train_ds.shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE, drop_remainder=False)
        val_dataset = val_ds.batch(BATCH_SIZE, drop_remainder=False)

        if is_cache:
            train_dataset, val_dataset = train_dataset.cache(), val_dataset.cache()

        if is_prefetch:
            train_dataset, val_dataset = train_dataset.prefetch(
                self.AUTOTUNE), val_dataset.prefetch(self.AUTOTUNE)

        return train_dataset, val_dataset
