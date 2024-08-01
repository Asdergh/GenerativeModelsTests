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

class WGAN:


    def __init__(self, input_dim 
                 , critic_conv_filters
                 , critic_conv_strides
                 , critic_conv_kernel_size
                 , critic_batch_normalization
                 , critic_dropout_rate
                 , critic_activation_foo
                 , critic_learning_rate
                 , critic_trashhold_rate
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
        
        self.critic_conv_filters = critic_conv_filters
        self.critic_conv_strides = critic_conv_strides
        self.critic_conv_kernel_size = critic_conv_kernel_size
        self.critic_batch_normalization_momentum = critic_batch_normalization
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_activation_foo = critic_activation_foo
        self.critic_learning_rate = critic_learning_rate
        self.critic_trashhold_rate = critic_trashhold_rate

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

        self.n_critic_layers = len(self.critic_conv_filters)
        self.n_generator_layers = len(self.generator_conv_t_filters)
        self.epoch = 0

        self._build_critic()
        self._build_generator()
        self._build_wgan_model()
    
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


    def _wasserstain(self, true_sample, gen_sample):
        return -K.mean(true_sample * gen_sample)
    
    def _build_critic(self):

        critic_input_layer = Input(shape=self.input_dim,  name="DesctiminatorInputLayer")
        critic_layer = critic_input_layer
        for layer_number in range(self.n_critic_layers):
            
            print(critic_layer.shape)
            critic_layer = Conv2D(filters=self.critic_conv_filters[layer_number]
                                         , strides=self.critic_conv_strides[layer_number]
                                         , kernel_size=self.critic_conv_kernel_size[layer_number]
                                         , padding="same"
                                         , name=f"CriticConv2DLayer_{layer_number}"
                                         , kernel_initializer=self.weights_init)(critic_layer)
            
            

            if self.critic_batch_normalization_momentum and layer_number > 0:
                critic_layer = BatchNormalization(momentum=self.critic_batch_normalization_momentum)(critic_layer)
            
            critic_layer = Activation(activation=self.critic_activation_foo)(critic_layer)
            if self.critic_dropout_rate:
                critic_layer = Dropout(rate=self.critic_dropout_rate)(critic_layer)
            
        critic_layer = Flatten()(critic_layer)
        critic_output = Dense(1, activation=None, kernel_initializer=self.weights_init, name="CriticOutputLayer")(critic_layer)
        self.critic_model = Model(critic_input_layer, critic_output)
    

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
    
    def _build_wgan_model(self):
        
        self.critic_model.compile(optimizer=self._get_optimizer(lr=self.critic_learning_rate)
                                         , loss=self._wasserstain
                                         , metrics=["accuracy"])
        
        self.set_trainable(self.critic_model, False)
        model_input = Input(shape=(self.hiden_dim, ), name="ModelInputLayer")
        model_output = self.critic_model(self.generator_model(model_input))
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=self._get_optimizer(lr=self.generator_learning_rate)
            , loss=self._wasserstain
            , metrics=["accuracy"]
        )

        self.set_trainable(self.critic_model, True)
    
    def _train_critic(self, train_tensor, batch_size, trashhold_rate):

        valid_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        random_index = np.random.randint(0, train_tensor.shape[0], batch_size)
        true_sample = train_tensor[random_index]

        noise = np.random.normal(0, 1, (batch_size, self.hiden_dim))
        gen_sample = self.generator_model.predict(noise)

        desc_loss_real, desc_acc_real = self.critic_model.train_on_batch(true_sample, valid_labels)
        desc_loss_fake, desc_acc_fake = self.critic_model.train_on_batch(gen_sample, fake_labels)

        desc_loss = 0.5 * (desc_loss_real + desc_loss_fake)
        desc_acc = 0.5 * (desc_acc_real + desc_acc_fake)

        if self.critic_trashhold_rate is not None:

            for layer in self.critic_model.layers:

                weights = layer.get_weights()
                weights = [np.clip(weights_tensor, -trashhold_rate, trashhold_rate) for weights_tensor in weights]
                layer.set_weights(weights)

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

            desc_train_history = self._train_critic(train_tensor=train_tensor
                                                    , batch_size=batch_size
                                                    , trashhold_rate=self.critic_trashhold_rate)
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

        self.critic_model.save(os.path.join(run_folder, "critic_model.h5"))
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



        

                
        