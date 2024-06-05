import os
import numpy as np
from PIL import Image

def apply_aperture(img, new_image, x, y, size, threshold):
    size //= 2
    left = max(y - size, 0)
    right = min(y + size, img.shape[1])
    low = max(x - size, 0)
    above = min(x + size, img.shape[0])
    
    aperture = img[low:above, left:right]
    ones = (aperture == 255).sum()

    if ones >= threshold:
        new_image[x, y] = 255

def rank_filter(img, size, threshold):
    if size % 2 == 0:
        raise Exception("Only odd size of aperture is supported")

    new_img = np.zeros(shape=img.shape, dtype=np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            apply_aperture(img, new_img, x, y, size, threshold)
            
    return new_img

def difference_image(img1, img2):
    return np.abs(img1 - img2).astype(np.uint8)

def main():
    input_folder = 'input'
    output_folder = 'output'
    diff_folder = 'difference'
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(diff_folder, exist_ok=True)
    
    image_names = ["img1.bmp", "img2.bmp", "img3.bmp", "face1.bmp", "face2.bmp", "house.bmp"]
    
    for image_name in image_names:
        image_path = os.path.join(input_folder, image_name)
        img = Image.open(image_path).convert('L')
        img_arr = np.array(img)
        
        # Применение рангового фильтра
        img_filtered_array = rank_filter(img_arr, size=5, threshold=10)
        img_filtered = Image.fromarray(img_filtered_array, 'L')
        img_filtered.save(os.path.join(output_folder, f'filtered_{image_name}'))
        
        # Вычисление разностного изображения
        diff_img_array = difference_image(img_arr, img_filtered_array)
        diff_img = Image.fromarray(diff_img_array, 'L')
        diff_img.save(os.path.join(diff_folder, f'difference_{image_name}'))

if __name__ == '__main__':
    main()
