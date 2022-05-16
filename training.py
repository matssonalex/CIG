from GAN import GAN
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import loader   # this will create "label_images.npy" and "raw_images.py"
from sklearn.metrics import mean_absolute_error as mae

def train_GAN(model, x, y, batch_size, n_epochs):
#     # hittils endast discriminator som tränas när man kör model.training_step(

    model.fit(x[:100], y[:100], batch_size=batch_size, epochs=n_epochs)


def test_GAN(model, x, y, ind):
    model.trainable = False
    

    validity, image = model(x[ind])
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(np.reshape(x[ind], (256,256)))
    plt.title('Mask')
    plt.subplot(1,3,2)
    plt.imshow(np.reshape(image, (256,256)))
    plt.title('Generated')
    plt.subplot(1,3,3)
    plt.imshow(np.reshape(y[ind], (256,256)))
    plt.title('Ground truth')
    



def loss_fn_gen(z_label, z_output, pred_fake):
    gamma = 0.8
    mean_abs_err = tf.keras.losses.mean_absolute_error(z_label, z_output)
    return gamma*tf.reshape(mean_abs_err, (np.shape(z_label)[0],1)) + (1 - pred_fake)**2


def loss_fn_disc(pred_real, pred_fake):
    return pred_fake**2 + (1 - pred_real)**2


if __name__ == '__main__':
    gan = GAN()
    gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5),
    loss_fn_g=loss_fn_gen,
    loss_fn_d=loss_fn_disc
    )
    
    # get training data
    label_img = np.load('label_images.npy')
    raw_img = np.load('raw_images.npy')

    ind = np.random.randint(960)

    test_GAN(gan, label_img, raw_img, ind)
    train_GAN(gan, label_img, raw_img, 10, 50)
    test_GAN(gan, label_img, raw_img, ind)
    print('hello')
    plt.show()



    