from GAN import GAN
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import loader   # this will create "label_images.npy" and "raw_images.py"

def train_GAN(model, batch_size, n_epochs):
    # hittils endast discriminator som tränas när man kör model.training_step()
    label_img = np.load('label_images.npy')
    raw_img = np.load('raw_images.npy')

    n_images = len(label_img)
    for i in range(n_epochs):
        #for j in range(int(n_images/batch_size))
        #model.predict(np.reshape(label_img[0:10])
        indexes = np.random.randint(0, n_images, (batch_size))
        data = [label_img[indexes], raw_img[indexes]]
        #model.full_model.fit(data)
        model.train_step(data)
        print(f'total discriminator loss: {sum(model.disc_loss)[0]}')
        print(f'total GAN loss: {sum(model.gan_loss)[0]}')

def loss_fn():
    pass

if __name__ == '__main__':
    gan = GAN()
    train_GAN(gan, 10, 10)

    