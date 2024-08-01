import numpy as np
import json as js
import random as rd
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import os

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal

class GAN:


    def __init__(self, input_dim 
                 , descriminator_conv_filters
                 , descriminator_conv_strides
                 , descriminator_conv_kernel_size
                 , descriminator_batch_normalization
                 , descriminator_dropout_rate
                 , descriminator_activation_foo
                 , descriminator_learning_rate
                 , generator_init_dense_shape
                 , generator_upsample
                 , generator_conv_t_filters
                 , generator_conv_t_strides
                 , generator_conv_t_kernel_size
                 , generator_batch_normalization
                 , generator_dropout_rate
                 , generator_learning_rate
                 , generator_activation_foo
                 , optimizer
                 , hiden_dim) -> None:

        self.input_dim = input_dim
        self.optimizer = optimizer
        self.hiden_dim = hiden_dim
        
        self.descriminator_conv_filters = descriminator_conv_filters
        self.descriminator_conv_strides = descriminator_conv_strides
        self.descriminator_conv_kernel_size = descriminator_conv_kernel_size
        self.descriminator_batch_normalization_momentum = descriminator_batch_normalization
        self.descriminator_dropout_rate = descriminator_dropout_rate
        self.descriminator_activation_foo = descriminator_activation_foo
        self.descriminator_learning_rate = descriminator_learning_rate

        self.generator_init_dense_shape = generator_init_dense_shape
        self.generator_upsample = generator_upsample
        self.generator_conv_t_filters = generator_conv_t_filters
        self.generator_conv_t_strides = generator_conv_t_strides
        self.generator_conv_t_kernel_size = generator_conv_t_kernel_size
        self.generator_batch_normalization_momentum = generator_batch_normalization
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        self.generator_activation_foo = generator_activation_foo

        self.weights_init = RandomNormal(mean=.0, stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.n_descriminator_layers = len(self.descriminator_conv_filters)
        self.n_generator_layers = len(self.generator_conv_t_filters)
        self.epoch = 0

        self._build_descriminator()
        self._build_generator()
        self._build_gan_model()
    
    def set_trainable(self, model, value):
        model.trainale = value
        for layer in model.layers:
            layer.trainable = value

    def _get_activation(self, activation):

        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)

        else:
            layer = Activation(activation)

        return layer
            
    def _get_optimizer(self, lr):

        if self.optimizer == "rms_prop":
            optimizer = RMSprop(learning_rate=lr)
        
        elif self.optimizer == "Adam":
            optimizer = Adam(learning_rate=lr)
        
        else:
            optimizer = Adam(learning_rate=lr)
        
        return optimizer


    def _build_descriminator(self):

        descriminator_input_layer = Input(shape=self.input_dim,  name="DesctiminatorInputLayer")
        descriminator_layer = descriminator_input_layer
        print(descriminator_layer.shape)
        for layer_number in range(self.n_descriminator_layers):
            
            print(descriminator_layer.shape)
            descriminator_layer = Conv2D(filters=self.descriminator_conv_filters[layer_number]
                                         , strides=self.descriminator_conv_strides[layer_number]
                                         , kernel_size=self.descriminator_conv_kernel_size[layer_number]
                                         , padding="same"
                                         , name=f"DescriminatorConv2DLayer_{layer_number}"
                                         , kernel_initializer=self.weights_init)(descriminator_layer)
            
            

            if self.descriminator_batch_normalization_momentum and layer_number > 0:
                descriminator_layer = BatchNormalization(momentum=self.descriminator_batch_normalization_momentum)(descriminator_layer)
            
            descriminator_layer = Activation(activation=self.descriminator_activation_foo)(descriminator_layer)
            if self.descriminator_dropout_rate:
                descriminator_layer = Dropout(rate=self.descriminator_dropout_rate)(descriminator_layer)
            
        descriminator_layer = Flatten()(descriminator_layer)
        descriminator_output = Dense(1, activation="sigmoid")(descriminator_layer)
        self.descriminator_model = Model(descriminator_input_layer, descriminator_output)
    

    def _build_generator(self):

        generator_input_layer = Input(shape=(self.hiden_dim, ), name="GeneratorInputLayer")
        generator_layer = generator_input_layer
        generator_layer = Dense(units=np.prod(self.generator_init_dense_shape), kernel_initializer=self.weights_init)(generator_layer)

        if self.generator_batch_normalization_momentum:
            generator_layer = BatchNormalization(momentum=self.generator_batch_normalization_momentum)(generator_layer)

        generator_layer = self._get_activation(self.generator_activation_foo)(generator_layer)
        if self.generator_dropout_rate:
            generator_layer = Dropout(self.generator_dropout_rate)(generator_layer)
        
        generator_layer = Reshape(self.generator_init_dense_shape)(generator_layer)
        for layer_number in range(self.n_generator_layers):

            if self.generator_upsample[layer_number] == 2:

                generator_layer = UpSampling2D()(generator_layer)
                generator_layer = Conv2D(filters=self.generator_conv_t_filters[layer_number]
                                         , strides=self.generator_conv_t_strides[layer_number]
                                         , kernel_size=self.generator_conv_t_kernel_size[layer_number]
                                         , name=f"GeneratorConv2DLayer_{layer_number}"
                                         , padding="same"
                                         , kernel_initializer=self.weights_init)(generator_layer)
            
            else:

                generator_layer = Conv2DTranspose(filters=self.generator_conv_t_filters[layer_number]
                                         , strides=self.generator_conv_t_strides[layer_number]
                                         , kernel_size=self.generator_conv_t_kernel_size[layer_number]
                                         , name=f"GeneratorConv2DTransposeLayer_{layer_number}"
                                         , padding="same"
                                         , kernel_initializer=self.weights_init)(generator_layer)
            
            if layer_number < self.n_generator_layers - 1:

                if self.generator_batch_normalization_momentum:
                    generator_layer = BatchNormalization(momentum=self.generator_batch_normalization_momentum)(generator_layer)
                generator_layer = Activation(activation=self.generator_activation_foo)(generator_layer)
                
            
            else:
                generator_layer = Activation(activation="tanh")(generator_layer)
        
        generator_output = generator_layer
        self.generator_model = Model(generator_input_layer, generator_output)
    
    def _build_gan_model(self):
        
        self.descriminator_model.compile(optimizer=self._get_optimizer(lr=self.descriminator_learning_rate)
                                         , loss="binary_crossentropy"
                                         , metrics=["accuracy"])
        
        self.set_trainable(self.descriminator_model, False)
        model_input = Input(shape=(self.hiden_dim, ), name="ModelInputLayer")
        model_output = self.descriminator_model(self.generator_model(model_input))
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=self._get_optimizer(lr=self.generator_learning_rate)
            , loss="binary_crossentropy"
            , metrics=["accuracy"]
        )

        self.set_trainable(self.descriminator_model, True)
    
    def _train_descriminator(self, train_tensor, batch_size):

        valid_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        random_index = np.random.randint(0, train_tensor.shape[0], batch_size)
        true_sample = train_tensor[random_index]

        noise = np.random.normal(0, 1, (batch_size, self.hiden_dim))
        gen_sample = self.generator_model.predict(noise)

        desc_loss_real, desc_acc_real = self.descriminator_model.train_on_batch(true_sample, valid_labels)
        desc_loss_fake, desc_acc_fake = self.descriminator_model.train_on_batch(gen_sample, fake_labels)

        desc_loss = 0.5 * (desc_loss_real + desc_loss_fake)
        desc_acc = 0.5 * (desc_acc_real + desc_acc_fake)

        return [desc_loss, desc_acc, desc_loss_fake, desc_acc_fake, desc_loss_real, desc_acc_real]

    def _train_generator(self, batch_size):

        valid_labels = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.hiden_dim))
        return self.model.train_on_batch(noise, valid_labels)


    def train_model(self
                    , train_tensor
                    , batch_size
                    , batches_per_print
                    , epochs
                    , run_folder):

        for epoch in range(epochs):

            desc_train_history = self._train_descriminator(train_tensor=train_tensor, batch_size=batch_size)
            gen_train_history = self._train_generator(batch_size=batch_size)
            
            self.d_losses.append(desc_train_history)
            self.g_losses.append(gen_train_history)
            
            if (epoch % batches_per_print) == 0:

                curent_weights_folder = os.path.join(run_folder, "weights")
                if not os.path.exists(curent_weights_folder):
                    os.mkdir(curent_weights_folder)
                curent_weights_file = os.path.join(curent_weights_folder, ".weights.h5")

                self._save_images(run_folder=run_folder, samples_number=25)
                self.model.save_weights(curent_weights_file)
                self._save_model(run_folder=run_folder)
            
            self.epoch += 1
    
    def _save_model(self, run_folder):

        self.descriminator_model.save(os.path.join(run_folder, "descriminator_model.h5"))
        self.generator_model.save(os.path.join(run_folder, "generator_model.h5"))
        self.model.save(os.path.join(run_folder, "model.h5"))

    def load_weights(self, run_folder):

        self.model.load_weights(run_folder)
    
    def _save_images(self, run_folder, samples_number):

        samples_number_sq = int(np.sqrt(samples_number))
        noise = np.random.normal(0, 1, (samples_number, self.hiden_dim))
        gen_images = self.generator_model.predict(noise)

        fig, axis = plt.subplots(nrows=samples_number_sq, ncols=samples_number_sq)
        sample_number = 0
        for sample_i in range(samples_number_sq):
            for sample_j in range(samples_number_sq):

                axis[sample_i, sample_j].imshow(gen_images[sample_number, :, :, :], cmap="mako")
                sample_number += 1
        
        fig.savefig(os.path.join(run_folder, f"epoch_number{self.epoch}_generation.png"))
        plt.close()



        

                
        