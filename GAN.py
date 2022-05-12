import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import matplotlib.pyplot as plt
layers = tf.keras.layers


class GAN(tf.keras.Model):
    def __init__(self, input_dim=256, input_channels=1):
        super(GAN, self).__init__()
        
        self.generator = self.create_generator(input_dim, input_channels)    

        self.generator.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")   # TODO CHANGE

        self.discriminator = self.create_discriminator((256,256,1))


    def training_step(self, data):
        # inspirerat av https://github.com/softmatterlab/DeepTrack-2.0/blob/develop/deeptrack/models/gans/cgan.py

        # data borde här innehålla en batch av masks (x) or riktiga bilder/raw images (y)
        x, y = data
        generated_images = self.generator(x)    # batch av fake bilder
        batch_size = len(x)
        valid = tf.ones(batch_size)
        fake = tf.zeros(batch_size)
        # först tränar vi discriminator sen hela paketet
        with tf.GradientTape() as tape:     # används för att typ hålla koll på gradients enkelt och träna de som ska tränas.
            pass                            # ibland ska vikterna för discriminatorn låsas tror jag.
            # predicted_real_images = self.discriminator([y, x])    # ska vara 1/0 vektor för predictions av real images
            # predicted_fake_images = self.discriminator([y, generated_images])     # samma fast för fake images
            # beräkna loss på något sätt: loss(predicted_real_images - valid) + loss(predicted_fake_images - fake)

        # nu kan vi använda tape att beräkna gradients
        # gradients = tape.gradient(loss, trainiable weights av discriminator)
        # använd discriminatorns optimizer för att uppdatera vikter mha gradients
        
        # implementera liknade fast för hela paketet, behövs:
        #   - från totala modellen behöver vi kunna få 1/0 av disc och den genererade bilden från gen
        #   - kopierar upp genererade bilderna massa gånger, varför?
        #   - kopierar upp riktiga bilderna massa gånger, varför?
        # använder sen dessa kopior av massa bilder för att beräkna den totala förlusten av hela paketet
        # sen gradients mha av tape och optimizer steget


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


        output = layers.Conv2D(16, kernel_size=1, padding="same", activation = "tanh")(x) #check activation here
        
        model = tf.keras.Model(input, output)
        return model
   
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


# mod = GAN()
# print(mod.generator.summary())

# print(mod.discriminator.summary())

# im = Image.open("cropped_raw/im_00_0.tif")
# im = np.asarray(im)
# im = np.reshape(im, (1, 256, 256, 1))

# im = mod.discriminator(im).numpy()
# print(im.shape)
# print(im)
#im = Image.fromarray(im)
#im.show()




