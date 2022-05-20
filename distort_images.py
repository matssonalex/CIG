from PIL import Image
import numpy as np

noisy_matrix = np.zeros([320, 256, 256])
flip_noisy_matrix = np.zeros([320, 256, 256])
raw_matrix = np.zeros([320, 256, 256])
raw_flip_matrix = np.zeros([320, 256, 256])
combined_matrix = np.zeros([640, 256, 256])
combined_raw_matrix = np.zeros([640, 256, 256])


def flip(image):
    np_image = np.asarray(image)
    np_image = np.flip(np_image)
    np_image = np.rot90(np_image)
    for i in range(np_image.shape[1]):
        np_image[:, i] = np.roll(np_image[:,i], i) # scewing
    return np_image

def noise_and_concat(image1, image2):

    np_image1 = np.asarray(image1)
    if np_image1.dtype == 'bool':
        np_image1 = (np_image1.astype(int)) # convert to values between (0, 1) from (False, True)
    else:
        np_image1 = np_image1/np.max(np_image1) # convert to values between (0, 1) from (0, 255)
    np_image2 = np.asarray(image2)
    if np_image2.dtype == 'bool':
        np_image2 = np_image2.astype(int) # convert to values between (0, 1) from (False, True)
    else:
        np_image2 = np_image2/np.max(np_image2) # convert to values between (0, 1) from (0, 255)
    row, col = np_image2.shape
    mean = 0
    var = 0.1
    gauss = np.random.normal(mean, var, (row, col))
    np_image2 = np_image2 + gauss # add noise to one picture
    np_image1 = (np_image1-np.min(np_image1))/(np.max(np_image1)-np.min(np_image1))*(-1) + 0  # set background to -1, membrane to 0
    np_image2 = (np_image2-np.min(np_image2))/(np.max(np_image2)-np.min(np_image2)) # set mitochondria to 1
    np_image_combined = np_image1 + np_image2
    np_image_combined = (np_image_combined-np.min(np_image_combined))/(np.max(np_image_combined)-np.min(np_image_combined)) # set mitochondria to 1
    np_image_combined = np_image_combined*255
    image_combined = Image.fromarray(np_image_combined.astype(np.uint8))
    return image_combined

# Loop through the maps and import images
counter = 0
for i in range(20):
    if i < 10:
        nr = '0' + f'{i}'
    else:
        nr = f'{i}'
    for part in range(16):
        ### create the masks
        membrane_import = 'images/cropped_membranes/im_' + nr + '_' + f'{part}' + '.png'
        mitochondria_import = 'images/cropped_mitochondria/im_' + nr + '_' + f'{part}' + '.png'
        img_membrane = Image.open(f'{membrane_import}')
        img_mitochondria = Image.open(f'{mitochondria_import}')
        flipped_img_membrane = flip(img_membrane) # flip and shear the image
        flipped_img_mitochondria = flip(img_mitochondria) # flip and shear the image
        #noisy_matrix[counter, :, :] = noise_and_concat(img_membrane, img_mitochondria) # concat two images and add noise
        #flip_noisy_matrix[counter, :, :] = noise_and_concat(flipped_img_membrane, flipped_img_mitochondria) # concat two images and add noise
        pic = noise_and_concat(img_membrane, img_mitochondria)
        pic2 = noise_and_concat(flipped_img_membrane, flipped_img_mitochondria)
        pic_label = 'images/combined/im_' + nr + '_' + f'{part}' + '.png'
        pic2_label = 'images/flipped_combined/im_' + nr + '_' + f'{part}' + '.png'
        pic.save(pic_label)
        pic2.save(pic2_label)
        combined_matrix[counter, :, :] = noisy_matrix[counter, :, :]
        combined_matrix[counter + 320, :, :] = flip_noisy_matrix[counter, :, :]
        
        #### flip and shear the RAW as well
        raw_import = 'images/cropped_raw/' + 'im_'+ nr + '_' + f'{part}' + '.tif'
        img_raw = Image.open(f'{raw_import}')
        np_img = np.asarray(img_raw)
        flipped_raw = flip(img_raw) # flip and shear the image
        np_img = 2*((np_img-np.min(np_img))/(np.max(np_img)-np.min(np_img))) -1
        flipped_raw = 2*((flipped_raw-np.min(flipped_raw))/(np.max(flipped_raw)-np.min(flipped_raw))) -1
        raw_matrix[counter, :, :] = np_img
        raw_flip_matrix[counter, :, :] = flipped_raw
        combined_raw_matrix[counter, :, :] = np_img
        combined_raw_matrix[counter + 320, :, :] = flipped_raw
        #noised_img_raw.save(raw_txt)
        counter += 1
### save the different images/msks
#np.save('noisy.npy', noisy_matrix)
#np.save('flip_noisy_images.npy', flip_noisy_matrix)
#np.save('raw_flip.npy', raw_flip_matrix)
#np.save('raw_matrix.npy', raw_matrix)
#np.save('combined_matrix.npy', combined_matrix)
# np.save('combined_raw_matrix.npy', combined_raw_matrix)
