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

def noiser(image1, image2):

    # np_image1 = np.asarray(image1)
    # np_image2 = np.asarray(image2)
    image1.paste(image2, (0,0), image2) #add together
    np_image = np.asarray(image1)
    # np_image = np_image1 + np_image2
    # row, col = np_image.shape
    # mean = 0
    # std = 0.01
    # var = std**2
    # gauss = np.random.normal(mean, var, (row, col))
    # gauss = gauss.reshape(row, col)
    # noisy = np_image + gauss
    #noisy = 2 * (noisy - min(noisy.flatten())) / (max(noisy.flatten()) - min(noisy.flatten())) - 1

    #noisy = np.linalg.norm()
    noisy = np_image.astype(np.uint8)
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
        membrane_import = 'images/cropped_membranes/im_' + nr + '_' + f'{part}' + '.png'
        mitochondria_import = 'images/cropped_mitochondria/im_' + nr + '_' + f'{part}' + '.png'
        img_membrane = Image.open(f'{membrane_import}')
        img_mitochondria = Image.open(f'{mitochondria_import}')
        combined_img = noiser(img_membrane, img_mitochondria) # choose distortion function and the dir output below
        #noised_img_raw= flip(img_raw)
        labels_txt = 'images/combined/' + 'im_'+ nr + '_' + f'{part}' + '.png'
        #raw_txt = 'images/noisy_raw/' + 'noisy_im_'+ nr + '_' + f'{part}' + '.tif'
        combined_img.save(labels_txt)
        #noised_img_raw.save(raw_txt)
