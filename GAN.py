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


        validity = self.discriminator(layers.concatenate([image, input_shape]))

        self.full_model = tf.keras.Model(self.generator.input, [validity, image])
        self.gan_loss = 0

        self.full_model.compile(
            optimizer='adam',
            loss='mean_absolute_error'
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
        batch_size = len(x_batch)
        
        x_batch = tf.reshape(x_batch, (batch_size, 256, 256, 1))    # borde kanske göras utanför?
        y_batch = tf.reshape(y_batch, (batch_size, 256, 256, 1))
        
        generated_images = self.generator(x_batch)    # batch av fake bilder
        generated_images = np.reshape(generated_images, (batch_size, 256, 256, 1))
        
        valid = tf.ones(batch_size)
        #fake = tf.zeros(batch_size)
        # först tränar vi discriminator sen hela paketet
        with tf.GradientTape() as tape:     # används för att typ hålla koll på gradients enkelt och träna de som ska tränas.
            pred_real_images = self.discriminator(layers.concatenate([y_batch, x_batch]))
            pred_fake_images = self.discriminator(layers.concatenate([generated_images, x_batch]))

            self.disc_loss = pred_fake_images**2 + (1 - pred_real_images)**2

        #print(self.disc_loss)    
        gradients = tape.gradient(self.disc_loss, self.discriminator.trainable_weights)
        self.discriminator.optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_weights)
            )
        
        
        
        with tf.GradientTape() as tape:
            output_validity, output_images = self.full_model(x_batch)   # if batch is 10 images, then output will be 10 numbers and 10 generated images
            
            raw_reshaped = tf.reshape(y_batch, (batch_size, 256*256))
            out_img_reshaped = tf.reshape(output_images, (batch_size, 256*256))
            raw_reshaped = raw_reshaped / 1.0
            out_img_reshaped = raw_reshaped / 1.0

            MAE = np.array([mae(out_img_reshaped[i], raw_reshaped[i]) for i in range(batch_size)])
            #MAE = np.array([(1/65536) * sum(abs(out_img_reshaped[i,:] - raw_reshaped[i,:])) for i in range(batch_size)])
            self.gan_loss = 0.8 * np.reshape(MAE, (10,1)) + (1 - pred_fake_images)**2
        
        #self.discriminator.traniable = False
        gradients = tape.gradient(self.gan_loss, self.full_model.trainable_weights)
        self.full_model.optimizer.apply_gradients(
            zip(gradients, self.full_model.trainable_weights)
            )
            

        
        #print(self.discriminator.trainable_weights[0][0][0])    
            # ibland ska vikterna för discriminatorn låsas tror jag.
            # predicted_real_images = self.discriminator([y, x])    # ska vara 1/0 vektor för predictions av real images
            # predicted_fake_images = self.discriminator([generated_images, x])     # samma fast för fake images
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

    def call(self, input, training=None, mask=None):
        #gen_img = self.generator(input)
        #print(np.shape(input))
        #print(np.shape(gen_img))
       # disc_input = layers.concatenate([tf.reshape(input, (256,256,1)), gen_img])
        #out = self.discriminator(disc_input)
        #eturn out
        pass

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




