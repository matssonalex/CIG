from pickletools import optimize
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
layers = tf.keras.layers
#from keras.utils import plot_model

class GAN(tf.keras.Model):
    def __init__(self, input_dim=256, input_channels=1, ):
        super(GAN, self).__init__()
        
        self.generator = self.create_generator(input_dim, input_channels)    

        self.generator.compile(
            optimizer='adam', 
            loss='mean_absolute_error', 
            metrics=['mean_absolute_error']
            ) 

        self.discriminator = self.create_discriminator((256,256,2))
        self.disc_loss = 0
        self.discriminator.compile(
            optimizer='adam', 
            loss='mean_absolute_error', 
            metrics=['mean_absolute_error']
            )   # oklart om dessa saker behöver vara här


        # slå ihop model till en
        input_shape = self.generator.input
        image = self.generator(input_shape)
        # self.discriminator.trainable = False


        validity = self.discriminator(layers.concatenate([image, input_shape]))

        self.full_model = tf.keras.Model(self.generator.input, [validity, image])
        self.gan_loss = 0

        self.full_model.compile(
            optimizer='adam',
            loss='mean_absolute_error',
            metrics=['mean_absolute_error']
        )

    def cutom_loss(self, pred_fake, generated_imgs, true_imgs):
        gamma = 0.8
        # är det mean absolute error mellan raw image och generead image av generator?
        loss = gamma * mae(true_imgs, generated_imgs) + (1 - pred_fake)**2
        return loss


    def train_step(self, data):
        # inspirerat av https://github.com/softmatterlab/DeepTrack-2.0/blob/develop/deeptrack/models/gans/cgan.py
        #self.discriminator.traniable = True
        # data borde här innehålla en batch av masks (x) or riktiga bilder/raw images (y)
        x_batch, y_batch = data
        batch_size = 10
        
        x_batch = tf.reshape(x_batch, (batch_size, 256, 256, 1))    # borde kanske göras utanför?
        y_batch = tf.reshape(y_batch, (batch_size, 256, 256, 1))
        
        generated_images = self.generator(x_batch)    # batch av fake bilder
        generated_images = tf.reshape(generated_images, (batch_size, 256, 256, 1))
        
       
        with tf.GradientTape() as tape:     # används för att typ hålla koll på gradients enkelt och träna de som ska tränas.
            pred_real_images = self.discriminator(layers.concatenate([y_batch, x_batch]))
            pred_fake_images = self.discriminator(layers.concatenate([generated_images, x_batch]))

            self.disc_loss = pred_fake_images**2 + (1 - pred_real_images)**2

        #print(self.disc_loss)    
        gradients = tape.gradient(self.disc_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_weights)
            )
       
        
        # self.generator.build((256,256,1))
        with tf.GradientTape() as tape:
            # this doesn't work. Have tried a lot of different approaches...

            # output_images = self.generator(x_batch)
            # output_validity, output_images = self.full_model(x_batch)  # if batch is 10 images, then output will be 10 numbers and 10 generated images
            output_validity, output_images = self.full_model(x_batch)
            raw_reshaped = tf.reshape(y_batch, (batch_size, 256*256))
            out_img_reshaped = tf.reshape(output_images, (batch_size, 256*256))
            raw_reshaped = raw_reshaped / 1.0
            out_img_reshaped = raw_reshaped / 1.0

            mean_abs_err = tf.convert_to_tensor([mae(out_img_reshaped[i], raw_reshaped[i]) for i in range(batch_size)], dtype=np.float32)

            # loss = tf.ones(10,1)    
            self.gan_loss = 0.8 * tf.reshape(mean_abs_err, (10,1)) + (1 - pred_fake_images)**2
 
        gradients = tape.gradient(self.gan_loss, self.full_model.trainable_weights)
        self.full_model.optimizer.apply_gradients(
            zip(gradients, self.full_model.trainable_weights)
            )
            

    def call(self, input,*args, **kwargs):
        #return self.generator.call(*args, **kwargs)
        gen_img = self.generator.call(input)
        out = self.discriminator.call(layers.concatenate([input, gen_img]))
        return out, gen_img
        #print(np.shape(input))
        #print(np.shape(gen_img))
       # disc_input = layers.concatenate([tf.reshape(input, (256,256,1)), gen_img])
        #out = self.discriminator(disc_input)
        #eturn out
        

    # def call(self, inputs, training=None, mask=None): # TODO
    #     gen_img = self.generator(input)
    #     output = self.discriminator([input, gen_img])
    #     return gen_img, output


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


        output = layers.Conv2D(1, kernel_size=1, padding="same", activation = "tanh")(x) #check activation here
        
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





