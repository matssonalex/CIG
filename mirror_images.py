import imageio
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def Verticalize(path):
    original_image = Image.open(path)
    vertical_image = original_image.transpose(method=Image.FLIP_TOP_BOTTOM)

    return vertical_image

def Horizontalize(path):
    original_image = Image.open(path)
    horizontal_image = original_image.transpose(method=Image.FLIP_LEFT_RIGHT)

    return horizontal_image

path = 'groundtruth-drosophila-vnc/stack1/labels'

for i in range(20):
    print(i)
    if i < 10:
        path = 'groundtruth-drosophila-vnc/stack1/labels/labels0000000' + str(i) + ".png"
    else:
        path = 'groundtruth-drosophila-vnc/stack1/labels/labels000000' + str(i) + ".png"

    horizontal_image = Horizontalize(path)
    vertical_image = Verticalize(path)
    horizontal_image.save("horizontal" + str(i) + ".png")
    vertical_image.save("vertical" + str(i) + ".png")
