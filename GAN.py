import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
layers = tf.keras.layers
#from keras.utils import plot_model

class GAN(tf.keras.Model):
    def __init__(self, input_dim=256, input_channels=1):
        super(GAN, self).__init__()
        
        self.generator = self.create_generator(input_dim, input_channels)    
        self.discriminator = self.create_discriminator((256,256,2))
        
       
    def compile(self, d_optimizer, g_optimizer, loss_fn_d, loss_fn_g):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn_d = loss_fn_d
        self.loss_fn_g = loss_fn_g    


    def train_step(self, data):
        # inspirerat av https://github.com/softmatterlab/DeepTrack-2.0/blob/develop/deeptrack/models/gans/cgan.py
        # data borde här innehålla en batch av masks (x) or riktiga bilder/raw images (y)
        x_batch, y_batch = data

        batch_size = np.shape(x_batch)[0]
        if batch_size is None:
            x_batch = tf.reshape(x_batch, (1, 256, 256, 1))    # borde kanske göras utanför?
            y_batch = tf.reshape(y_batch, (1, 256, 256, 1))

        x_batch = tf.reshape(x_batch, (batch_size, 256, 256, 1))    # borde kanske göras utanför?
        y_batch = tf.reshape(y_batch, (batch_size, 256, 256, 1))
        
        # self.generator.trainable = False
        generated_images = self.generator(x_batch)    # batch av fake bilder
        generated_images = tf.reshape(generated_images, (batch_size, 256, 256, 1))
        
        # Här tränas discriminator, 
        self.discriminator.trainable = True
        with tf.GradientTape() as tape:     # används för att typ hålla koll på gradients enkelt och träna de som ska tränas.
            pred_real_images = self.discriminator(layers.concatenate([y_batch, x_batch]))
            pred_fake_images = self.discriminator(layers.concatenate([generated_images, x_batch]))
            
            d_loss = self.loss_fn_d(pred_real_images, pred_fake_images) # mse change
            # d_loss = self.generator.compiled_loss(pred_real_images, pred_fake_images)
        
        self.generator.trainable = False    # check
        gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_weights)
            )
       
        # train generator betydligt fler felkällor här tror jag. 
        self.generator.trainable = True
        self.discriminator.trainable = False
        
        with tf.GradientTape() as tape:
            generated_images = self.generator(x_batch)
            pred_real_images = self.discriminator(layers.concatenate([y_batch, x_batch]))
            pred_fake_images = self.discriminator(layers.concatenate([generated_images, x_batch]))

            raw_reshaped = tf.reshape(y_batch, (batch_size, 256*256))
            out_img_reshaped = tf.reshape(generated_images, (batch_size, 256*256))
            raw_reshaped = raw_reshaped / 1.0
            out_img_reshaped = raw_reshaped / 1.0

            g_loss = self.loss_fn_g(raw_reshaped, out_img_reshaped, pred_fake_images, pred_real_images)   # mae between first 2, mse CHaNGE
            # g_loss = self.loss_fn_g([y_batch, generated_images], [pred_fake_images, pred_real_images])


        gradients = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_weights)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}
            

    def call(self, input):
        input = tf.expand_dims(input, -1)
        input_shape = np.shape(input)
        if len(input_shape) == 3:
            input = tf.reshape(input, (1, 256, 256, 1))
        else:
            input = tf.reshape(input, (input_shape[0], 256, 256, 1))
        
        gen_img = self.generator(input)
        out = self.discriminator(layers.concatenate([input, gen_img]))
        return out, gen_img     # validity between 0-1 and the generated image
       

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
            layers.Conv2D(1, kernel_size=(4,4), strides=1, padding='same', activation='linear')
            # layers.Linear()
            # layers.Flatten(),       # la till detta eftersom de hade det i deeptrack modellen, ger nu 1 tal som output
            # layers.Dense(1, activation='sigmoid')  # return a matrix with a 1-filter conv2d. Linear activation
        ])
        return discriminator





