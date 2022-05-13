from pickletools import optimize
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
layers = tf.keras.layers


class GAN(tf.keras.Model):
    def __init__(self, input_dim=256, input_channels=1, ):
        super(GAN, self).__init__()
        
        self.generator = Generator()
        self.discriminator = Discriminator()

        # Input shape
        self.model_input = self.generator.input

        # The generator model_input and generates img
        img = self.generator(self.model_input)

        # For the combined model we will only train the generator
       # self.discriminator.trainable = False

        # The discriminator takes the generated images as input and determines validity
        validity = self.discriminator([img, self.model_input])

        # The assembled model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.gan_model = tf.keras.models.Model(self.model_input, [validity, img])


  

class Encoder(tf.keras.Model):
    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(2)
            ])

    def call(self, x):
        return self.encoder(x)



class Decoder(tf.keras.Model):
    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.decoder = tf.keras.Sequential(
            [
            layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same'),
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            ]
        )

    def call(self, x):
        return self.decoder(x)


class Center(tf.keras.Model):
    def __init__(self, filters):
        super(Center, self).__init__()
        self.center = tf.keras.Sequential(
            [
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')   
            ]
        )

    def call(self, x):
        return self.center(x)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
         
        self.generator = self.create_generator((256,256,1))
        # self.encoder_block1 = Encoder(16)
        # self.encoder_block2 = Encoder(32)
        # self.encoder_block3 = Encoder(64)
        # self.encoder_block4 = Encoder(128)
        # self.encoder_block5 = Encoder(256)
        # self.encoder_block6 = Encoder(512)
        # self.center_block = Center(1024)
        # self.decoder_block1 = Decoder(512)
        # self.decoder_block2 = Decoder(256)
        # self.decoder_block3 = Decoder(128)
        # self.decoder_block4 = Decoder(64)
        # self.decoder_block5 = Decoder(32)
        # self.decoder_block6 = Decoder(16)
        # self.output_layer = layers.Conv2D(1, kernel_size=1, padding="same", activation = "tanh")    
        # self.generator.compile(
        #     optimizer='adam', 
        #     loss='mean_absolute_error', 
        #     metrics=['mean_absolute_error']
        #     ) 

    def create_generator(self, input_shape):
        #self.input_layer = 
        self.encoder_block1 = tf.keras.Sequential([
            layers.Input(shape=(256,256,1)),
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(256,256,1)),
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(2)
        ])
        # self.encoder_block1 = Encoder(16)
        self.encoder_block2 = Encoder(32)
        self.encoder_block3 = Encoder(64)
        self.encoder_block4 = Encoder(128)
        self.encoder_block5 = Encoder(256)
        self.encoder_block6 = Encoder(512)
        self.center_block = Center(1024)
        self.decoder_block1 = Decoder(512)
        self.decoder_block2 = Decoder(256)
        self.decoder_block3 = Decoder(128)
        self.decoder_block4 = Decoder(64)
        self.decoder_block5 = Decoder(32)
        self.decoder_block6 = Decoder(16)
        self.output_layer = layers.Conv2D(1, kernel_size=1, padding="same", activation = "tanh") #check activation here

    def call(self, input):
        input = layers.Input(shape=(256,256,1))
        x1 = self.encoder_block1(input)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        x5 = self.encoder_block5(x4)
        x6 = self.encoder_block6(x5)
        x7 = self.center_block(x6)
        x8 = self.decoder_block1(layers.concatenate([x7, x6]))
        x9 = self.decoder_block2(layers.concatenate([x8, x5]))
        x10 = self.decoder_block3(layers.concatenate([x9, x4]))
        x11 = self.decoder_block4(layers.concatenate([x10, x3]))
        x12 = self.decoder_block5(layers.concatenate([x11, x2]))
        x13 = self.decoder_block6(layers.concatenate([x12, x1]))
        output = self.output_layer(x13)
        return output


class Discriminator(tf.keras.Model):
    def __init__(self, input_dim=256, input_channels=1):
        super(Discriminator, self).__init__()

        self.discriminator = self.create_discriminator((256,256,2))
        self.disc_loss = 0
        self.discriminator.compile(
            optimizer='adam', 
            loss='mean_absolute_error', 
            metrics=['mean_absolute_error'])
            # )   # oklart om dessa saker behöver vara här

    def create_discriminator(self, input_shape):
        # kanske lägga till alpha=0.2 i varje leaky-relu
        discriminator = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(4,4), strides=2, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(32, kernel_size=(4,4), strides=2, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(64, kernel_size=(4,4), strides=2, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(128, kernel_size=(4,4), strides=2, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(256, kernel_size=(4,4), strides=2, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(512, kernel_size=(4,4), strides=2, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Flatten(),       # la till detta eftersom de hade det i deeptrack modellen, ger nu 1 tal som output
            layers.Dense(1, activation='sigmoid') # Ska denna finnas? Fattar ej.
        ])
        return discriminator
    
    def call(self, x):
        return self.discriminator(x)


gen = Generator()
# print(gan.generator.summary())
gen.build((256,256,1))
print(gen.summary())

disc = Discriminator()
disc.build((1, 256,256,2))
print(disc.summary())


gan = GAN()