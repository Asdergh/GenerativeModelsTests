import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import json as js
import os

from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Dropout, Embedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model

class RNNGN:

    def __init__(self, total_words, embedding_size, 
                model_lstm_units, model_lstm_activations, model_lstm_dropout_rates,
                model_dense_units, model_dense_activations, model_dense_dropout_rates) -> None:

        self.total_words = total_words
        self.embedding_size = embedding_size

        self.model_lstm_units = model_lstm_units
        self.model_lstm_activations = model_lstm_activations
        self.model_lstm_dropout_rates = model_lstm_dropout_rates

        self.model_dense_units = model_dense_units
        self.model_dense_activations = model_dense_activations
        self.model_dense_dropout_rates = model_dense_dropout_rates

        self.lstm_layers_n = len(self.model_lstm_units)
        self.dense_layers_n = len(self.model_dense_units)
    
    def _build_model_(self):

        input_layer = Input(shape=(None, ))
        embedding_layer = Embedding(input_dim=self.total_words, output_dim=self.embedding_size)(input_layer)
        lstm_layer = embedding_layer

        for layer_n in range(self.lstm_layers_n):

            lstm_layer = LSTM(units=self.model_lstm_units[layer_n], return_sequences=True)(lstm_layer)
            lstm_layer = Bidirectional(lstm_layer)
            lstm_layer = Dropout(self.model_lstm_dropout_rates)(lstm_layer)
        
        dense_layer = lstm_layer
        for layer_n in range(self.dense_layers_n):

            dense_layer = Dense(units=self.model_dense_units[layer_n])(dense_layer)
            dense_layer = Dropout(rate=self.)
