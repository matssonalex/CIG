import numpy as np
import tensorflow as tf
layers = tf.keras.layers


class GAN(tf.keras.Model):
    def __init__(self, input_dim=64, input_channels=3):
        super(GAN, self).__init__()
        
        self.generator = self.create_generator(input_dim, input_channels)    
        
        #self.generator.compile()
        self.generator.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")   # TODO CHANGE
     

    # def double_conv_block(self, x, n_filters):
    #     x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    #     x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    #     return x


    # def downsample_block(self, x, n_filters):
    #     skip_vec = self.double_conv_block(x, n_filters)
    #     x = layers.MaxPool2D(2)(skip_vec)
    #     # x = layers.Dropout(0.3)(x)
        
    #     return x, skip_vec

    # def upsample_block(self, x, skip_vec, n_filters):
    #     x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
    #     x = layers.concatenate([x, skip_vec])
    #     # x = layers.Dropout(0.3)(x)
    #     x = self.double_conv_block(x, n_filters)

    #     return x

    def create_generator(self, input_dim, input_channels):
        filters = [16, 32, 64, 128, 256, 512]
        skip_vec = []
        input = tf.keras.Input(shape=(input_dim, input_dim, 1))

        

        x = input    
        for n_filters in filters:
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            #x = self.double_conv_block(x, n_filters)
            skip_vec.append(x)
            x = layers.MaxPool2D(2)(x)

        x = layers.Conv2D(1024, kernel_size=3, strides=1, padding='same', activation='relu')(x) # TODO RESNET?
        x = layers.Conv2D(1024, kernel_size=3, strides=1, padding='same', activation='relu')(x)

        for ind, n_filters in enumerate(filters[::-1]):
            x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
            x = layers.concatenate([x, skip_vec[-(ind+1)]])
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            x = layers.Conv2D(n_filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)

            # x = self.double_conv_block(x, n_filters)
            # self.skip_vec.append(x)
            # x = layers.MaxPool2D(2)(x) 
        #print(np.shape(input))
        # x, skip_vec1 = self.downsample_block(input, 16),
        # x, skip_vec2 = self.downsample_block(x, 32),
        # x, skip_vec3 = self.downsample_block(x, 64),
        # x, skip_vec4 = self.downsample_block(x, 128),
        # x, skip_vec5 = self.downsample_block(x, 256),
        # x, skip_vec6 = self.downsample_block(x, 512),

        # x = self.double_conv_block(x, 1024),

        # x = self.upsample_block(x, skip_vec6, 512),
        # x = self.upsample_block(x, skip_vec5, 256),
        # x = self.upsample_block(x, skip_vec4, 128),
        # x = self.upsample_block(x, skip_vec3, 64),
        # x = self.upsample_block(x, skip_vec2, 32),
        # x = self.upsample_block(x, skip_vec1, 16)

        output = layers.Conv2D(16, kernel_size=1, padding="same", activation = "relu")(x) #check activation here
        
        model = tf.keras.Model(input, output)
        return model
   

mod = GAN()
print(mod.generator.summary())

