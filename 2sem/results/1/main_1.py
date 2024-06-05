import numpy as np
from PIL import Image
import os

def rgb_to_grayscale(image_array):
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image.astype(np.uint8)

def convert_to_grayscale(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.bmp'):
            img = Image.open(os.path.join(input_folder, filename))
            img_array = np.array(img)
            gray_array = rgb_to_grayscale(img_array)
            gray_img = Image.fromarray(gray_array, 'L')
            gray_img.save(os.path.join(output_folder, 'grey_' + filename))

if __name__ == '__main__':
    input_folder_1 = '1/input'
    output_folder_1 = '1/output_1'
    os.makedirs(output_folder_1, exist_ok=True)
    convert_to_grayscale(input_folder_1, output_folder_1)
