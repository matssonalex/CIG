from re import X
import tensorflow as tf
layers = tf.keras.layers


class GAN(tf.keras.Model):
    def __init__(self, input_dim=64, input_channels=3):
        super(GAN, self).__init__()

        self.generator = self.create_generator(input_dim, input_channels)    


    def double_conv_block(self, x, n_filters):
        x = layers.Conv2D(n_filters, (3,3), stride=1, padding='same', activation='relu')(x)
        x = layers.Conv2D(n_filters, (3,3), stride=1, padding='same', activation='relu')(x)

        return x


    def downsample_block(self, x, n_filters):
        skip_vec = self.double_conv_block(x, n_filters)
        x = layers.MaxPool2D(2)(skip_vec)
        x = layers.Dropout(0.3)(x)
        
        return x, skip_vec

    def upsample_block(self, x, skip_vec, n_filters):
        x = layers.Conv2DTranspose(n_filters, (3,3), stride=2, padding='same')(x)
        x = layers.concatenate([x, skip_vec])
        x = layers.Dropout(0.3)(x)
        x = self.double_conv_block(x, n_filters)

        return x

    def create_generator(self, input_dim, input_channels):
        
        input = tf.keras.Input(shape=(input_dim, input_dim, input_channels)),
        x, skip_vec1 = self.downsample_block(input, 16),
        x, skip_vec2 = self.downsample_block(x, 32),
        x, skip_vec3 = self.downsample_block(x, 64),
        x, skip_vec4 = self.downsample_block(x, 128),
        x, skip_vec5 = self.downsample_block(x, 256),
        x, skip_vec6 = self.downsample_block(x, 512),

        x = self.double_conv_block(x, 1024),

        x = self.upsample_block(x, skip_vec6, 512),
        x = self.upsample_block(x, skip_vec5, 256),
        x = self.upsample_block(x, skip_vec4, 128),
        x = self.upsample_block(x, skip_vec3, 64),
        x = self.upsample_block(x, skip_vec2, 32),
        x = self.upsample_block(x, skip_vec1, 16)

        output = layers.Conv2D(16, 1, padding="same", activation = "relu") #check activation here
        model = tf.keras.Model(input, output)
        return model
   
    
    




mod = GAN()
print(mod.summary())