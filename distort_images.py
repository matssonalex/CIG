from PIL import Image
import numpy as np

def flip(image):
    np_image = np.asarray(image)
    np_image = np.flip(np_image)
    np_image = np.rot90(np_image)
    for i in range(np_image.shape[1]):
        np_image[:, i] = np.roll(np_image[:,i], i) # scewing
    flipped_image = Image.fromarray(np_image)
    return flipped_image

def noiser(image):
    np_image = np.asarray(image)
    row, col = np_image.shape
    mean = 0
    STD = 1
    var = STD**2
    gauss = np.random.normal(mean, var, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = np_image + gauss
    noisy = noisy.astype(np.uint8)
    noisy_image = Image.fromarray(noisy)
    return noisy_image

# Loop through the maps and import images
counter = 0
for i in range(20):
    if i < 10:
        nr = '0' + f'{i}'
    else:
        nr = f'{i}'
    for part in range(16):
        labels_import = 'images/flipped_labels/flipped_im_' + nr + '_' + f'{part}' + '.png'
        #raw_import = 'images/cropped_raw/im_' + nr + '_' + f'{part}' + '.tif'
        img_labels = Image.open(f'{labels_import}')
        #img_raw = Image.open(f'{raw_import}')
        noised_img_labels = noiser(img_labels) # choose distortion function and the dir output below
        #noised_img_raw= flip(img_raw)
        labels_txt = 'images/flipped_noisy_labels/' + 'flipped_noisy_im_'+ nr + '_' + f'{part}' + '.png'
        #raw_txt = 'images/noisy_raw/' + 'noisy_im_'+ nr + '_' + f'{part}' + '.tif'
        noised_img_labels.save(labels_txt)
        #noised_img_raw.save(raw_txt)

