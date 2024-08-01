import numpy as np
import tensorflow as tf
import pickle as pk
import json as js
import os


class AutoEncoder:

    def __init__(self, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides
                 ,decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides
                 ,hiden_dim, input_dim, var_autoencoder
                 , batch_normalization=False, dropout_layers=False
                 , dim_version="2D") -> None:
        
        self.input_dim = input_dim
        self.dim_version = dim_version
        self.var_autoencoder = var_autoencoder
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.hiden_dim = hiden_dim
        self.batch_normalization = batch_normalization
        self.dropout_layers = dropout_layers

        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.n_layers_decoder = len(self.decoder_conv_t_filters)
        self._build()


    def _build(self):

        encoder_input_layer = tf.keras.Input(shape=self.input_dim
                                             , name="encoder_input_layer")
        
        encoder_layer = encoder_input_layer
        for layer_number in range(self.n_layers_encoder):

            if self.dim_version == "2D":

                encoder_layer = tf.keras.layers.Conv2D(filters=self.encoder_conv_filters[layer_number]
                                                    , kernel_size=self.encoder_conv_kernel_size[layer_number]
                                                    , strides=self.encoder_conv_strides[layer_number]
                                                    , name=f"EncoderConv2DLayer_{layer_number}"
                                                    , padding="same"
                                                    , activation="relu")(encoder_layer)
            
            elif self.dim_version == "3D":

                encoder_layer = tf.keras.layers.Conv3D(filters=self.encoder_conv_filters[layer_number]
                                                    , kernel_size=self.encoder_conv_kernel_size[layer_number]
                                                    , strides=self.encoder_conv_strides[layer_number]
                                                    , name=f"EncoderConv2DLayer_{layer_number}"
                                                    , padding="same"
                                                    , activation="relu")(encoder_layer)
            
            else:

                encoder_layer = tf.keras.layers.Conv2D(filters=self.encoder_conv_filters[layer_number]
                                                    , kernel_size=self.encoder_conv_kernel_size[layer_number]
                                                    , strides=self.encoder_conv_strides[layer_number]
                                                    , name=f"EncoderConv2DLayer_{layer_number}"
                                                    , padding="same"
                                                    , activation="relu")(encoder_layer)

            encoder_layer = tf.keras.layers.LeakyReLU()(encoder_layer)

            if self.batch_normalization:
                encoder_layer = tf.keras.layers.BatchNormalization()(encoder_layer)
            
            if self.dropout_layers:
                encoder_layer = tf.keras.layers.Dropout(rate=0.25)(encoder_layer)
        

        shape_befor_flatting = tf.keras.backend.int_shape(encoder_layer)[1:]
        encoder_layer = tf.keras.layers.Flatten()(encoder_layer)
        if self.var_autoencoder:

            def sampling(args):

                mu, log_var = args
                epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0., stddev=1.)
                result_point = mu + tf.keras.backend.exp(log_var / 2.0) * epsilon

                return result_point
            
            self.mu = tf.keras.layers.Dense(self.hiden_dim, name="MuLayer")(encoder_layer)
            self.log_var = tf.keras.layers.Dense(self.hiden_dim, name="LogVarLayer")(encoder_layer)
            self.encoder_mu_log_var = tf.keras.Model(encoder_input_layer, (self.mu, self.log_var))

            encoder_output = tf.keras.layers.Lambda(sampling, name="EncoderOutput")([self.mu, self.log_var])
            self.encoder_model = tf.keras.Model(encoder_input_layer, encoder_output)

        else:

            encoder_output = tf.keras.layers.Dense(self.hiden_dim, name="EncoderOutput")(encoder_layer)
            self.encoder_model = tf.keras.Model(encoder_input_layer, encoder_output)

        decoder_input_layer = tf.keras.Input(shape=(self.hiden_dim, ), name="DecoderInput")
        decoder_layer = tf.keras.layers.Dense(np.product(shape_befor_flatting))(decoder_input_layer)
        decoder_layer = tf.keras.layers.Reshape(shape_befor_flatting)(decoder_layer)


        for layer_number in range(self.n_layers_decoder):
            
            if self.dim_version == "2D":

                decoder_layer = tf.keras.layers.Conv2DTranspose(filters=self.decoder_conv_t_filters[layer_number]
                                                    , kernel_size=self.decoder_conv_t_kernel_size[layer_number]
                                                    , strides=self.decoder_conv_t_strides[layer_number]
                                                    , name=f"DecoderConv2DTransposeLayer_{layer_number}"
                                                    , padding="same")(decoder_layer)
            
            elif self.dim_version == "3D":

                decoder_layer = tf.keras.layers.Conv3DTranspose(filters=self.decoder_conv_t_filters[layer_number]
                                                    , kernel_size=self.decoder_conv_t_kernel_size[layer_number]
                                                    , strides=self.decoder_conv_t_strides[layer_number]
                                                    , name=f"DecoderConv2DTransposeLayer_{layer_number}"
                                                    , padding="same")(decoder_layer)

            else:

                decoder_layer = tf.keras.layers.Conv2DTranspose(filters=self.decoder_conv_t_filters[layer_number]
                                                    , kernel_size=self.decoder_conv_t_kernel_size[layer_number]
                                                    , strides=self.decoder_conv_t_strides[layer_number]
                                                    , name=f"DecoderConv2DTransposeLayer_{layer_number}"
                                                    , padding="same")(decoder_layer)
                
            if layer_number < self.n_layers_decoder - 1:

                decoder_layer = tf.keras.layers.LeakyReLU()(decoder_layer)

                if self.batch_normalization:
                    decoder_layer = tf.keras.layers.BatchNormalization()(decoder_layer)
                
                if self.dropout_layers:
                    decoder_layer = tf.keras.layers.Dropout(rate=0.25)(decoder_layer)
            
            else:
                decoder_layer = tf.keras.layers.Activation("sigmoid")(decoder_layer)
        
        decoder_output = decoder_layer
        self.decoder_model = tf.keras.Model(decoder_input_layer, decoder_output)

        model_input = encoder_input_layer
        model_output = self.decoder_model(encoder_output)
        self.model = tf.keras.Model(model_input, model_output)
    
    
    
    def compile(self, learning_rate):
        
        def _vae_r_loss( true_sample, pred_sample):
            return tf.keras.backend.mean(tf.keras.backend.square(true_sample - pred_sample))

        def _vae_kl_loss( true_sample, pred_sample):
            return -0.5 * tf.keras.backend.sum(1 + self.log_var - tf.keras.backend.square(self.mu) -
                                            tf.keras.backend.exp(self.log_var), axis=1)
        
        def _vae_loss(true_sample, pred_sample):

            r_loss = _vae_r_loss(true_sample, pred_sample)
            kl_loss = _vae_kl_loss(true_sample, pred_sample)

            return r_loss + kl_loss
        
        if self.var_autoencoder:
            
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                               , loss=_vae_loss
                               , metrics=[_vae_r_loss, _vae_kl_loss])
            
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                           , loss=tf.keras.losses.MeanSquaredError())

    
    def save_params_pkl(self, folder):

        if not os.path.exists(folder):
            os.mkdir(folder)
    
        params_log_path = os.path.join(folder, "params.pkl")
        with open(params_log_path, "wb") as pkl_file:

            pk.dump([self.input_dim
                , self.encoder_conv_filters
                , self.encoder_conv_kernel_size
                , self.encoder_conv_strides
                , self.decoder_conv_t_filters
                , self.decoder_conv_t_kernel_size
                , self.decoder_conv_t_strides
                , self.hiden_dim
                , self.batch_normalization
                , self.dropout_layers], pkl_file)


    def train(self, train_data, batch_size, epochs, shuffle, run_folder):
        
        weights_folder_path = os.path.join(run_folder, "weights")
        if not os.path.exists(weights_folder_path):
            os.mkdir(weights_folder_path)
        weights_file_path = os.path.join(weights_folder_path, ".weights.h5")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file_path, save_weights_only=True, verbose=1)
        self.model.fit(train_data
                       , train_data
                       , batch_size=batch_size
                       , shuffle=shuffle
                       , epochs=epochs
                       , callbacks=[checkpoint])
    
                

            

            




        