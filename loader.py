from PIL import Image
import numpy as np

nr_of_cropped = 16
nr_of_images = 20
format = '.png'
label_matrix = np.zeros([960, 256, 256])
raw_matrix = np.zeros([960, 256, 256])

def loader(nr_of_images, nr_of_cropped, rout, counter, format, matrix):
    for i in range(nr_of_images):
        if i < 10:
            nr = '0' + f'{i}'
        else:
            nr = f'{i}'
        for j in range(nr_of_cropped):
            path_label = rout + nr + '_' + f'{j}' + format
            img_label = Image.open(f'{path_label}')
            np_label = np.asarray(img_label)
            matrix[counter, :, :] = np_label
            counter += 1
# For the labels
rout = 'cropped_labels/im_'
loader(nr_of_images, nr_of_cropped, rout, 0, format, label_matrix)
rout = 'flipped_labels/flipped_im_'
loader(nr_of_images, nr_of_cropped, rout, 320, format, label_matrix)
rout = 'noisy_labels/noisy_im_'
loader(nr_of_images, nr_of_cropped, rout, 640, format, label_matrix)

# For the raw files
format = '.tif'
rout = 'cropped_raw/im_'
loader(nr_of_images, nr_of_cropped, rout, 0, format, raw_matrix)
rout = 'flipped_raw/flipped_im_'
loader(nr_of_images, nr_of_cropped, rout, 320, format, raw_matrix)
rout = 'cropped_raw/im_'
loader(nr_of_images, nr_of_cropped, rout, 640, format, raw_matrix)

np.save('label_images.npy', label_matrix)
np.save('raw_images.npy', raw_matrix)