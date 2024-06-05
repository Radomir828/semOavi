import numpy as np
from PIL import Image
import os

def adaptive_threshold(image_array, s=16, t=20):
    integral_image = np.cumsum(np.cumsum(image_array.astype(np.int64), axis=0), axis=1)
    out = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            x1 = max(i - s // 2, 0)
            x2 = min(i + s // 2, image_array.shape[0] - 1)
            y1 = max(j - s // 2, 0)
            y2 = min(j + s // 2, image_array.shape[1] - 1)
            count = (x2 - x1) * (y2 - y1)
            
            sum_ = integral_image[x2, y2]
            if x1 > 0:
                sum_ -= integral_image[x1 - 1, y2]
            if y1 > 0:
                sum_ -= integral_image[x2, y1 - 1]
            if x1 > 0 and y1 > 0:
                sum_ += integral_image[x1 - 1, y1 - 1]
            
            if image_array[i, j] * count < sum_ * (100 - t) / 100:
                out[i, j] = 0
            else:
                out[i, j] = 255

    return out

def binarize_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.bmp'):
            img = Image.open(os.path.join(input_folder, filename))
            img_array = np.array(img)
            binarized_array = adaptive_threshold(img_array)
            binarized_img = Image.fromarray(binarized_array, 'L')
            binarized_img.save(os.path.join(output_folder, 'binarized_' + filename))

if __name__ == '__main__':
    input_folder_2_4 = '1/output_1'
    output_folder_2_4 = '2.4/output_2.4'
    os.makedirs(output_folder_2_4, exist_ok=True)
    binarize_images(input_folder_2_4, output_folder_2_4)
