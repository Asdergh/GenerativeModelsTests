import numpy as np
import matplotlib.pyplot as plt
import random as rd
import json as js
import cv2
import os

from collections import deque
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Normalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Add
from tensorflow import pad

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


class CycleGAN:

    def __init__(self, discriminator_filters_number
                 , generator_filters_number
                 , learning_rate
                 , discriminator_optimizer
                 , discriminator_loss_function
                 , discriminator_metrics
                 , discriminator_activation_function
                 , generator_model_type
                 , generator_activation_function
                 , entire_model_optimizer
                 , entire_model_losses
                 , entire_model_metrics
                 , input_shape
                 , buffer_max_lenght
                 , lambda_validation
                 , lambda_reconstraction
                 , lambda_id):
        
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        #self.path = int(self.input_shape[0] / (2 ** 3))
        self.path = 6
        self.discriminator_patch_size = (self.path, self.path, 1)
        self.buffer_max_lenght = buffer_max_lenght

        self.discriminator_filters_n = discriminator_filters_number
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_loss_function = discriminator_loss_function
        self.discriminator_metrics = discriminator_metrics
        self.discriminator_activation_function = discriminator_activation_function

        self.generator_model_type = generator_model_type
        self.generator_filters_n = generator_filters_number
        self.generator_activation_function = generator_activation_function

        self.model_optimizer = entire_model_optimizer
        self.model_losses = entire_model_losses
        self.model_metrics = entire_model_metrics

        self.buffer_A = deque(maxlen=self.buffer_max_lenght)
        self.buffer_B = deque(maxlen=self.buffer_max_lenght)

        self.weights_init = RandomNormal(mean=0., stddev=0.02)
        self.lambda_validation = lambda_validation
        self.lambda_reconstraction = lambda_reconstraction
        self.lambda_id = lambda_id

        self.discriminator_losses = []
        self.generator_losses = []
        self.compile_model()

        self.d_losses = []
        self.g_losses = []

    
    def get_optimizer(self, optimizer, learning_rate):

        if optimizer == "Adam":
            return Adam(learning_rate=learning_rate)
            
        elif optimizer == "rms_prop":
            return RMSprop(learning_rate=learning_rate)
            
        elif optimizer == "sgd":
            return SGD(learning_rate=learning_rate)

        else:
            return Adam(learning_rate=learning_rate)
        
        

    def _build_generator_unet(self, layers_number):
        
        down_sampling_layers_buffer = []
        up_sampling_layers_buffer = []

        def get_downsampling_layer(input_layer, filters, kernel_size=4):
            
            down_layer = Conv2D(filters=filters
                                , kernel_size=kernel_size
                                , padding="same"
                                , strides=2)(input_layer)
            
            down_layer = Normalization(axis=-1, mean=None, variance=None, invert=False)(down_layer)
            down_layer = Activation("relu")(down_layer)
            return down_layer
        
        def get_upsampling_layer(input_layer, skip_layer, filters, kernel_size=4):

            upsample_layer = UpSampling2D(size=2)(input_layer)
            upsample_layer = Conv2D(filters=filters
                                    , padding="same"
                                    , kernel_size=kernel_size
                                    , strides=1)(upsample_layer)
            
            upsample_layer = Activation("relu")(upsample_layer)
            upsample_layer = Concatenate()([upsample_layer, skip_layer])
            return upsample_layer

        input_layer = Input(shape=self.input_shape)
        down_layer = input_layer
        filters_n_expander = 1

        for down_layer_number in range(layers_number // 2):
            
            if down_layer_number == 0:
                down_layer = get_downsampling_layer(down_layer, self.generator_filters_n)
            
            else:
                down_layer = get_downsampling_layer(down_layer, self.generator_filters_n * filters_n_expander)

            filters_n_expander *= 2
            down_sampling_layers_buffer.append(down_layer)
        
        for up_layer_number in range(layers_number // 2):

            filters_n_expander //= 2
            if (up_layer_number != 0):

                if (up_layer_number == 1):
                    up_layer = get_upsampling_layer(input_layer=down_sampling_layers_buffer[-up_layer_number]
                                                    , skip_layer=down_sampling_layers_buffer[-(up_layer_number + 1)]
                                                    , filters=self.generator_filters_n * filters_n_expander)
            
                else:
                    up_layer = get_upsampling_layer(input_layer=up_layer
                                                    , skip_layer=down_sampling_layers_buffer[-(up_layer_number + 1)]
                                                    , filters=self.generator_filters_n * filters_n_expander)
                
                up_sampling_layers_buffer.append(up_layer)
        
        up_layer = UpSampling2D(size=2)(up_layer)
        up_sampling_layers_buffer.append(up_layer)
        output_layer = Conv2D(filters=self.input_shape[-1]
                              , strides=1
                              , padding="same"
                              , kernel_size=4
                              , activation="tanh")(up_sampling_layers_buffer[-1])
        
        return Model(input_layer, output_layer)
    
    def _build_generator_resnet(self, layer_number_rec):
        
        layers_buffer = []
        def get_conv_layer(input_layer, filters, final):

            layer = ReflectionPadding2D(padding=(3, 3))(input_layer)
            layer = Conv2D(filters=filters
                           , padding="valid"
                           , kernel_size=(7, 7)
                           , strides=1
                           , kernel_initializer=self.weights_init)(layer)

            if final:

                layer = Activation("tanh")(layer)
            
            else:

                layer = Normalization(axis=-1)(layer)
                layer = Activation("relu")(layer)
            
            return layer

        def get_downsampling_layer(input_layer, filters):

            layer = Conv2D(filters=filters
                           , kernel_size=(3, 3)
                           , strides=2
                           , padding="same"
                           , kernel_initializer=self.weights_init)(input_layer)
            
            layer = Normalization(axis=-1)(layer)
            layer = Activation("relu")(layer)

            return layer

        def get_residual_layer(input_layer, filters):

            short_cut = input_layer
            layer = ReflectionPadding2D(padding=(1, 1))(input_layer)
            layer = Conv2D(filters=filters
                           , kernel_size=(3, 3)
                           , strides=1
                           , padding="valid"
                           , kernel_initializer=self.weights_init)(layer)
            layer = Normalization(axis=-1)(layer)
            layer = Activation("relu")(layer)

            layer = ReflectionPadding2D(padding=(1, 1))(layer)
            layer = Conv2D(filters=filters
                           , kernel_size=(3, 3)
                           , padding="valid"
                           , strides=1
                           , kernel_initializer=self.weights_init)(layer)
            layer = Normalization(axis=-1)(layer)

            return Add()([short_cut, layer])
        
        def get_upsampling_layer(input_layer, filters):

            layer = Conv2DTranspose(filters=filters
                           , kernel_size=(3, 3)
                           , strides=2
                           , padding="same"
                           , kernel_initializer=self.weights_init)(input_layer)
            layer = Normalization(axis=-1)(layer)
            layer = Activation("relu")(layer)

            return layer

        filters_n_expander = 1
        input_layer = Input(shape=self.input_shape)

        filters_n_expander *= 2
        conv_layer = get_conv_layer(input_layer=input_layer, filters=self.generator_filters_n, final=False)
        layers_buffer.append(conv_layer)

        downsample_layer = get_downsampling_layer(input_layer=layers_buffer[-1], filters=self.generator_filters_n * filters_n_expander)
        layers_buffer.append(downsample_layer)

        filters_n_expander *= 2
        downsample_layer = get_downsampling_layer(input_layer=layers_buffer[-1], filters=self.generator_filters_n * filters_n_expander)
        layers_buffer.append(downsample_layer)

        for _ in range(layer_number_rec):

            rec_layer = get_residual_layer(input_layer=layers_buffer[-1], filters=self.generator_filters_n * filters_n_expander)
            layers_buffer.append(rec_layer)
        
        filters_n_expander = int(filters_n_expander / 2)
        upsample_layer = get_upsampling_layer(input_layer=layers_buffer[-1], filters=self.generator_filters_n * filters_n_expander)
        layers_buffer.append(upsample_layer)

        upsample_layer = get_upsampling_layer(input_layer=layers_buffer[-1], filters=self.generator_filters_n * filters_n_expander)
        layers_buffer.append(upsample_layer)

        output_layer = get_conv_layer(input_layer=upsample_layer, filters=self.input_shape[-1], final=True)
        layers_buffer.append(output_layer)

        return Model(input_layer, output_layer)
    

            

    def _build_discriminator(self, layers_number):

        self.disc_layers_buffer = []
        def get_disc_layer(input_layer, filters, strides, kernel_size=4, norm=True):

            discriminator_layer = Conv2D(filters=filters
                                         , kernel_size=kernel_size
                                         , strides=strides
                                         , activation="leaky_relu"
                                         , kernel_initializer=self.weights_init)(input_layer)
            
            if norm:
                discriminator_layer = Normalization(axis=-1, mean=None, variance=None, invert=False)(discriminator_layer)
            
            return discriminator_layer
        
        disc_input_layer = Input(shape=self.input_shape)
        disc_layer = disc_input_layer
        filters_n_expander = 1

        for disc_layer_number in range(layers_number // 2):

            if disc_layer_number == 0:
                disc_layer = get_disc_layer(disc_layer, self.discriminator_filters_n, norm=False, strides=2)
            
            else:
                disc_layer = get_disc_layer(disc_layer, self.discriminator_filters_n * filters_n_expander, strides=2)
            
            filters_n_expander *= 2
            self.disc_layers_buffer.append(disc_layer)
        
        disc_output_layer = Conv2D(filters=1
                                  , strides=1
                                  , kernel_size=4
                                  , kernel_initializer=self.weights_init
                                  , padding="same")(disc_layer)
        
        return Model(disc_input_layer, disc_output_layer)
    
    def compile_model(self):

        self.discriminator_A = self._build_discriminator(layers_number=8)
        self.discriminator_B = self._build_discriminator(layers_number=8)
        
        self.discriminator_A.compile(
            optimizer=self.get_optimizer(optimizer=self.discriminator_optimizer, learning_rate=self.learning_rate)
            , loss=self.discriminator_loss_function
            , metrics=self.discriminator_metrics
        )
        self.discriminator_B.compile(
            optimizer=self.get_optimizer(optimizer=self.discriminator_optimizer, learning_rate=self.learning_rate)
            , loss=self.discriminator_loss_function
            , metrics=self.discriminator_metrics
        )

        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False

        if self.generator_model_type == "unet":

            self.generator_A2B = self._build_generator_unet(layers_number=8)
            self.generator_B2A = self._build_generator_unet(layers_number=8)
        
        elif self.generator_model_type == "resnet":

            self.generator_A2B = self._build_generator_resnet(layer_number_rec=9)
            self.generator_B2A = self._build_generator_resnet(layer_number_rec=9)

        else:
            raise ValueError("wrong model type")
        
        self.input_image_A = Input(shape=self.input_shape)
        self.input_image_B = Input(shape=self.input_shape)
        
        self.fake_A2B_image = self.generator_A2B(self.input_image_A)
        self.fake_B2A_image = self.generator_B2A(self.input_image_B)

        self.reconstracted_image_A = self.generator_B2A(self.fake_A2B_image)
        self.reconstracted_image_B = self.generator_A2B(self.fake_B2A_image)
        
        self.image_A_id = self.generator_B2A(self.input_image_A)
        self.image_B_id = self.generator_A2B(self.input_image_B)

        self.valid_A2B = self.discriminator_B(self.fake_A2B_image)
        self.valid_B2A = self.discriminator_A(self.fake_B2A_image)

        self.model = Model(inputs=[self.input_image_A, self.input_image_B]
                           , outputs=[self.valid_A2B, self.valid_B2A
                                     , self.reconstracted_image_A, self.reconstracted_image_B
                                     , self.image_A_id, self.image_B_id])
        
        self.model.compile(
            optimizer=self.model_optimizer
            , loss=self.model_losses
            , loss_weights=[self.lambda_validation, self.lambda_validation
                            , self.lambda_reconstraction, self.lambda_reconstraction
                            , self.lambda_id, self.lambda_id]
        )

        self.discriminator_A.trainable = True
        self.discriminator_B.trainable = True
    
    def _train_discriminator(self, train_images, train_styles, batch_size):

        valid_labels = np.ones((batch_size, ) + self.discriminator_patch_size)
        fake_labels = np.zeros((batch_size, ) + self.discriminator_patch_size)
        
        random_idx_A = np.random.randint(0, train_images.shape[0], batch_size)
        random_idx_B = np.random.randint(0, train_styles.shape[0], batch_size)
        
        images_A = train_images[random_idx_A]
        images_B = train_styles[random_idx_B]

        fake_images_A = self.generator_B2A.predict(images_B)
        fake_images_B = self.generator_A2B.predict(images_A)

        discriminator_A_real_loss, discriminator_A_real_acc = self.discriminator_A.train_on_batch(images_A, valid_labels)
        discriminator_A_fake_loss, discriminator_A_fake_acc = self.discriminator_A.train_on_batch(fake_images_A, fake_labels)
        discriminator_B_real_loss, discriminator_B_real_acc = self.discriminator_B.train_on_batch(images_B, valid_labels)
        discriminator_B_fake_loss, discriminator_B_fake_acc = self.discriminator_B.train_on_batch(fake_images_B, fake_labels)

        discriminator_mean_acc = np.mean([discriminator_A_real_acc, discriminator_A_fake_acc
                                          , discriminator_B_real_acc, discriminator_B_fake_acc])
        
        discriminator_mean_loss = np.mean([discriminator_A_real_loss, discriminator_A_fake_loss
                                          , discriminator_B_real_loss, discriminator_B_fake_loss])
        
        return [discriminator_mean_loss, discriminator_mean_acc
                , discriminator_A_real_loss, discriminator_A_real_acc, discriminator_A_fake_loss, discriminator_A_fake_acc
                , discriminator_B_real_loss, discriminator_B_real_acc, discriminator_B_fake_loss, discriminator_B_fake_acc]

    def _train_generator(self, train_images, train_styles, batch_size):

        valid_labels = np.ones((batch_size, ) + self.discriminator_patch_size)

        random_idx_A = np.random.randint(0, train_images.shape[0], batch_size)
        random_idx_B = np.random.randint(0, train_styles.shape[0], batch_size)
        
        images_A = train_images[random_idx_A]
        images_B = train_styles[random_idx_B]

        return self.model.train_on_batch([images_A, images_B]
                                         , [valid_labels, valid_labels
                                            , images_A, images_B
                                            , images_A, images_B])


    def save_generated_samples(self, run_folder, images_tensor, styles_tensor, epoch, samples_number):

        samples_number_sq = int(np.sqrt(samples_number))

        generated_samples_folder = os.path.join(run_folder, "Generated_samples")
        if not os.path.exists(generated_samples_folder):
            os.mkdir(generated_samples_folder)
        generated_folder = os.path.join(generated_samples_folder, f"GenegeratedImagesOn_{epoch}_epoch")
        
        fake_A = self.generator_B2A.predict(styles_tensor)
        fake_B = self.generator_A2B.predict(images_tensor)
        
        rec_A = self.generator_B2A.predict(fake_B)
        rec_B = self.generator_B2A.predict(fake_A)

        gen_tensor = np.zeros(shape=((6 * self.input_shape[0]),
                                     (samples_number * self.input_shape[1]),
                                     3))
        
        images_buffer = [images_tensor, styles_tensor, fake_A, fake_B, rec_A, rec_B]
        for (tensor_number, image_tensor) in enumerate(images_buffer):
            for (image_number, image) in enumerate(image_tensor):
                
                try:

                    gen_tensor[tensor_number * self.input_shape[0]: (tensor_number + 1) * self.input_shape[0],
                            image_number * self.input_shape[1]: (image_number + 1) * self.input_shape[1],
                            :] = image
                
                except BaseException:
                    pass

        plt.style.use("dark_background")
        fig, axis = plt.subplots()
        axis.imshow(gen_tensor)
        fig.savefig(generated_folder)


    def train_model(self, epochs, epoch_per_save
                    , train_images, train_styles
                    , batch_size, run_folder=None):
        
        train_images = train_images
        train_styles = train_styles

        models_set = [("discriminator_A.keras", self.discriminator_A)
                        , ("discriminator_B.keras", self.discriminator_B)
                        , ("generator_A2B.keras", self.generator_A2B)
                        , ("generator_B2A.keras", self.generator_B2A)
                        , ("model.keras", self.model)]
        
        for epoch in range(epochs):

            discriminator_history = self._train_discriminator(train_images=train_images
                                                              , train_styles=train_styles
                                                              , batch_size=batch_size)
            
            generator_history = self._train_generator(train_images=train_images
                                                      , train_styles=train_styles
                                                      , batch_size=batch_size)
            
            self.d_losses.append(discriminator_history)
            self.g_losses.append(generator_history)

            if (epoch % epoch_per_save) == 0:

                self.save_generated_samples(run_folder=run_folder, images_tensor=train_images,
                                             styles_tensor=train_styles, epoch=epoch,
                                               samples_number=25)


        
        models_folder = os.path.join(run_folder, "models")
        if not os.path.exists(models_folder):
            os.mkdir(models_folder)

        for (model_path, model) in models_set:

            curent_model_path = os.path.join(run_folder, model_path)
            model.save(curent_model_path)
                
            
                
            

if __name__ == "__main__":

    net = CycleGAN(
    discriminator_filters_number=64
    , generator_filters_number=32
    , generator_model_type="res"
    , learning_rate=0.01
    , entire_model_losses=["mse", "mse"
                           , "mse", "mse"
                           , "mse", "mse"]
    , entire_model_optimizer=Adam(0.0002, 0.5)
    , entire_model_metrics=None
    , lambda_validation=1
    , lambda_reconstraction=10
    , lambda_id=5
    , discriminator_optimizer="adam"
    , discriminator_loss_function="mse"
    , discriminator_metrics=["accuracy"]
    , discriminator_activation_function="relu"
    , generator_activation_function="relu"
    , input_shape=(128, 128, 3)
    , buffer_max_lenght=50
    )

    net.discriminator_A.summary()
    print("next model" + 56 * "-")
    net.discriminator_B.summary()
    print("next model" + 56 * "-")
    net.generator_A2B.summary()
    print("next model" + 56 * "-")
    net.generator_A2B.summary()
    print("next model" + 56 * "-")
    net.generator_B2A.summary()
            
            

            


        



                
                    

            







            


        
        