from GAN import GAN
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
import loader   # this will create "label_images.npy" and "raw_images.py"
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_GAN(model, x, y, batch_size, n_epochs):
#     # hittils endast discriminator som tränas när man kör model.training_step(

    model.fit(x, y, batch_size=batch_size, epochs=n_epochs)


def test_GAN(model, x, y, ind):
    model.trainable = False
    

    validity, image = model(x[ind])
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(x[ind])
    plt.title('Mask')
    plt.subplot(1,3,2)
    plt.imshow(np.reshape(image, (256,256)))
    plt.title('Generated')
    plt.subplot(1,3,3)
    plt.imshow(y[ind])
    plt.title('Ground truth')
    


def loss_fn_gen(z_label, z_output, pred_fake, pred_real):
    gamma = 0.8
    # mean_abs_err = tf.keras.losses.mean_absolute_error(z_label, z_output)
    # a = gamma*tf.reshape(mean_abs_err, (np.shape(z_label)[0],1)) + (1 - pred_fake)**2
    # c = []
    # for i in range(10):
    #     temp = abs(z_label[i]


    # b = gamma * tf.keras.losses.mean_absolute_error(z_label, z_output) + tf.keras.losses.mean_squared_error(pred_fake, pred_real)
    mean_abs_err = tf.math.reduce_mean(abs(z_label - z_output), axis=1)
    a = (pred_fake - pred_real)**2
    a = tf.math.reduce_mean(a, axis=1)
    a = tf.math.reduce_mean(a, axis=1)
    b = gamma * tf.reshape(mean_abs_err, (10,1)) + a
    # b = tf.reshape(b, (10,1))
    return b


def loss_fn_disc(pred_real, pred_fake):
    # a = tf.keras.losses.mean_squared_error(pred_real, pred_fake)
    # mean_sq_err = tf.keras.loss.mean
    # b = pred_fake**2 + (1 - pred_real)**2
    return (pred_real - pred_fake)**2


def preprocess_data(label_img, raw_img):
    norm_label_img = []
    for label in label_img:
        min_label = min(label.flatten())
        max_label = max(label.flatten())
        temp = 2 * ((label - min_label) / (max_label - min_label)) - 1
        norm_label_img.append(temp)

    norm_raw_img = []
    for raw in raw_img:
        min_raw = min(raw.flatten())
        max_raw = max(raw.flatten())
        temp = 2 * ((raw - min_raw) / (max_raw - min_raw)) - 1
        norm_raw_img.append(temp)

    return np.array(norm_label_img), np.array(norm_raw_img)


if __name__ == '__main__':
    gan = GAN()
    gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5), #check with different lr
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5), #check with different lr
    loss_fn_g=loss_fn_gen,
    loss_fn_d=loss_fn_disc
    )
    
    # get training data
    label_img = np.load('combined_images.npy')
    raw_img = np.load('raw_images.npy')

    # label_img = label_img/255
    # raw_img = raw_img/255

    norm_label_img, norm_raw_img = preprocess_data(label_img, raw_img)


    ind = np.random.randint(320)

    test_GAN(gan, norm_label_img, norm_raw_img, ind)
    train_GAN(gan, norm_label_img, norm_raw_img, 10, 10)
    test_GAN(gan, norm_label_img, norm_raw_img, ind)
    print('hello')
    plt.show()



    