from cProfile import label
from GAN import GAN
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import loader   # this will create "label_images.npy" and "raw_images.py"
from sklearn.metrics import mean_absolute_error as mae

def train_GAN(model, batch_size, n_epochs):
    # hittils endast discriminator som tränas när man kör model.training_step()
    label_img = np.load('label_images.npy')
    raw_img = np.load('raw_images.npy')

    n_images = len(label_img)
    for i in range(n_epochs):
        #for j in range(int(n_images/batch_size))

        indexes = np.random.randint(0, n_images, (batch_size))
        data = [label_img[indexes], raw_img[indexes]]
        loss = model.train_step(data)
        d_loss = loss['d_loss']
        g_loss = loss['g_loss']
        # print('hej')
        #d_loss
        print(f'discriminator loss: {d_loss.numpy()}')
        print(f'generator loss: {g_loss.numpy()}')

        # test_picture = label_img[0]
        #a, b = model.predict(test_picture)
        # plt.imshow(tf.reshape(b[0], (256,256)))
        # plt.show()


def loss_fn_gen(z_label, z_output, pred_fake):
    gamma = 0.8
    mean_abs_err = tf.convert_to_tensor([mae(z_output[i], z_label[i]) for i in range(10)], dtype=np.float32)

    return gamma*tf.reshape(mean_abs_err, (10,1)) + (1 - pred_fake**2)


def loss_fn_disc(pred_real, pred_fake):
    return pred_fake**2 + (1 - pred_real)**2


if __name__ == '__main__':
    gan = GAN()
    gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn_g=loss_fn_gen,
    loss_fn_d=loss_fn_disc
)
    train_GAN(gan, 10, 10)


    