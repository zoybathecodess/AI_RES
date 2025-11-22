# src/models/nn/lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models




def build_lstm(input_shape, units=64, dropout=0.2, seed=42):
tf.random.set_seed(seed)
np.random.seed(seed)
inp = layers.Input(shape=input_shape)
x = layers.LSTM(units, return_sequences=False)(inp)
x = layers.Dropout(dropout)(x)
out = layers.Dense(1)(x)
model = models.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
return model
