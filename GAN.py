import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
layers = tf.keras.layers


class GAN(tf.keras.Model):
    def __init__(self, input_dim=64, input_channels=1):
        super(GAN, self).__init__()
        
        self.generator = self.create_generator(input_dim, input_channels)    

        self.generator.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")   # TODO CHANGE

        self.discriminator = self.create_discriminator((256,256,1))

     

    def call(self, inputs, training=None, mask=None): # TODO
        return self.generator(inputs)


    def create_generator(self, input_dim, input_channels):
        filters = [16, 32, 64, 128, 256, 512]
        skip_vec = []
        input = tf.keras.Input(shape=(input_dim, input_dim, input_channels))

        x = input    
        for n_filters in filters:
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            skip_vec.append(x)
            x = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(1024, kernel_size=3, strides=1, padding='same', activation='relu')(x) # TODO RESNET?
        x = layers.Conv2D(1024, kernel_size=3, strides=1, padding='same', activation='relu')(x)

        for ind, n_filters in enumerate(filters[::-1]):
            x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
            x = layers.concatenate([x, skip_vec[-(ind+1)]])
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)


        output = layers.Conv2D(16, kernel_size=1, padding="same", activation = "softmax")(x) #check activation here
        
        model = tf.keras.Model(input, output)
        return model
   
    def create_discriminator(self, input_shape):
        discriminator = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(4,4), strides=2),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(32, kernel_size=(4,4), strides=2),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(64, kernel_size=(4,4), strides=2),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(128, kernel_size=(4,4), strides=2),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(256, kernel_size=(4,4), strides=2),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(512, kernel_size=(4,4), strides=2),
            tfa.layers.InstanceNormalization(),
            layers.LeakyReLU(),
        ])
        return discriminator


mod = GAN()
print(mod.generator.summary())

print(mod.discriminator.summary())




