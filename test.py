import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json as js


from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

class ConvGAN:

    def __init__(self, input_shape, hiden_dim, 
                dis_conv_depth, dis_dense_depth, dis_conv_activation,
                dis_dense_activation, dis_out_activation, dis_norm_momentum, 
                gen_conv_depth, gen_conv_activation, gen_out_activation, 
                gen_dropout_rate, gen_norm_momentum, dis_dropout_rate,
                dis_learning_rate, run_folder) -> None:

        
        self.run_folder = run_folder
        self.input_shape = input_shape
        self.hiden_dim = hiden_dim

        self.gen_conv_depth = gen_conv_depth
        self.gen_conv_activation = gen_conv_activation
        self.gen_out_activation = gen_out_activation
        self.gen_dropout_rate = gen_dropout_rate
        self.gen_norm_momentum = gen_norm_momentum

        self.dis_conv_depth = dis_conv_depth
        self.dis_dense_depth = dis_dense_depth
        self.dis_conv_activation = dis_conv_activation
        self.dis_dense_activation = dis_dense_activation
        self.dis_out_activation = dis_out_activation
        self.dis_norm_momentum = dis_norm_momentum
        self.dis_dropout_rate = dis_dropout_rate
        self.dis_learning_rate = dis_learning_rate

        self.dis_loss = []
        self.gen_loss = []
        self.epoch_iterator = 0

        self._build_discriminator_()
        self._build_generator_()
        self._build_model_()
    
    def _build_generator_(self):

        conv_filters = (2 ** self.gen_conv_depth)

        gen_input_layer = Input(shape=(self.hiden_dim, ))
        gen_rec_layer = Dense(units=np.prod(self.saved_shape))(gen_input_layer)
        gen_rec_layer = Reshape(target_shape=self.saved_shape)(gen_rec_layer)

        gen_conv_layer = gen_rec_layer
        for _ in range(self.gen_conv_depth):

            gen_conv_layer = Conv2DTranspose(filters=conv_filters, kernel_size=(3, 3), strides=2, padding="same")(gen_conv_layer)
            gen_conv_layer = Activation(self.gen_conv_activation)(gen_conv_layer)
            gen_conv_layer = Dropout(rate=self.gen_dropout_rate)(gen_conv_layer)
            gen_conv_layer = BatchNormalization(momentum=self.gen_norm_momentum)(gen_conv_layer)

            conv_filters //= 2
        
        gen_out_layer = Conv2D(filters=self.input_shape[-1], kernel_size=(3, 3), strides=1, padding="same")(gen_conv_layer)
        gen_out_layer = Activation(self.gen_out_activation)(gen_out_layer)

        self.generator = Model(gen_input_layer, gen_out_layer)
    
    def _build_discriminator_(self):

        dis_input_layer = Input(shape=self.input_shape)
        dis_conv_layer = dis_input_layer
        conv_filters = (2 ** self.gen_conv_depth)

        for _ in range(self.dis_conv_depth):
            
            dis_conv_layer = Conv2D(filters=conv_filters, kernel_size=(3, 3), strides=2, padding="same")(dis_conv_layer)
            dis_conv_layer = Activation(self.dis_conv_activation)(dis_conv_layer)
            dis_conv_layer = Dropout(rate=self.dis_dropout_rate)(dis_conv_layer)
            dis_conv_layer = BatchNormalization(momentum=self.dis_norm_momentum)(dis_conv_layer)
        
        self.saved_shape = dis_conv_layer.shape[1:]
        dis_dense_layer = Flatten()(dis_conv_layer)
        dense_units = (2 ** self.dis_dense_depth)

        for _ in range(self.dis_dense_depth):

            dis_dense_layer = Dense(units=dense_units, activation=self.dis_dense_activation)(dis_dense_layer)
            dis_dense_layer = Dropout(rate=self.dis_dropout_rate)(dis_dense_layer)
        
        dis_out_layer = Dense(units=1, activation=self.dis_out_activation)(dis_dense_layer)
        self.discriminator = Model(dis_input_layer, dis_out_layer)
    
    def _build_model_(self):

        self.discriminator.compile(optimizer=RMSprop(learning_rate=0.01), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
        self.discriminator.trainabel = False

        model_input_layer = Input(shape=(self.hiden_dim, ))
        model_output_layer = self.discriminator(self.generator(model_input_layer))
        self.model = Model(model_input_layer, model_output_layer)
        self.model.compile(loss=BinaryCrossentropy(), optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
        
        self.discriminator.trinable = True
    
    def _train_discriminator_(self, train_tensor, batch_size):

        noise = np.random.normal(0.1, 0.2, (batch_size, self.hiden_dim))
        random_idx = np.random.randint(0, train_tensor.shape[0] - 1, batch_size)
        
        valid_labels = np.ones(batch_size)
        fake_labels = np.zeros(batch_size)

        true_samples = train_tensor[random_idx]
        gen_samples = self.generator.predict(noise)
        
        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_samples, valid_labels)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_samples, fake_labels)

        d_avg_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_avg_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_avg_loss, d_avg_acc, d_loss_real, d_acc_real, d_loss_fake, d_acc_fake]

    def _train_generator_(self, train_tensor, batch_size):

        valid_labels = np.ones(batch_size)
        noise = np.random.normal(0.1, 0.2, (batch_size, self.hiden_dim))
        return self.model.train_on_batch(noise, valid_labels)

    def _save_samples(self, samples_number, cmap):

        fig, axis = plt.subplots()
        noise = np.random.normal(0.1, 0.2, (samples_number, self.hiden_dim))

        samples_number_sq = int(np.sqrt(samples_number))
        gen_samples = self.generator.predict(noise)
        show_tensor = np.zeros(shape=(samples_number_sq * self.input_shape[0], 
                                       samples_number_sq * self.input_shape[1], 
                                       self.input_shape[2]))

        sample_number = 0
        for i in range(samples_number_sq):
            for j in range(samples_number_sq):

                show_tensor[i * self.input_shape[0]: (i + 1) * self.input_shape[0],
                            j * self.input_shape[1]: (j + 1) * self.input_shape[1], :] = gen_samples[sample_number]
                sample_number += 1

        axis.imshow(show_tensor, cmap=cmap)
        gen_samples = os.path.join(self.run_folder, "gen_samples")
        if not os.path.exists(gen_samples):
            os.mkdir(gen_samples)
        
        curent_gen_samples = os.path.join(gen_samples, f"gen_samples_{self.epoch_iterator}.png")
        fig.savefig(curent_gen_samples)
    
    def train(self, train_tensor, epochs, batch_size, epoch_per_save, cmap):

        for epoch in range(epochs):

            self.dis_loss.append(self._train_discriminator_(train_tensor, batch_size))
            self.gen_loss.append(self._train_generator_(train_tensor, batch_size))
            
            if (epoch % epoch_per_save) == 0:
                self._save_samples(samples_number=25, cmap=cmap)

            self.epoch_iterator += 1
        

        



        