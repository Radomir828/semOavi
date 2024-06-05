import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from to_semitone import to_semitone
from haralik import haralik, CON, LUN
from contrast import histogram_equalization

def create_directories(base_path):
    subdirs = ['contrasted', 'haralik', 'haralik_contrasted', 'histograms', 'semitone']
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

def main():
    base_path = 'output'
    create_directories(base_path)
    image_path = 'images/image.png'
    
    semitone_img = to_semitone(image_path)
    semitone_img.save(os.path.join(base_path, 'semitone', 'image.png'))

    semi = np.array(Image.open(os.path.join(base_path, 'semitone', 'image.png')).convert('L'))

    transformed = histogram_equalization(semi)
    transformed_img = Image.fromarray(transformed.astype(np.uint8), 'L')
    transformed_img.save(os.path.join(base_path, 'contrasted', 'image.png'))

    figure, axis = plt.subplots(2, 1)
    axis[0].hist(x=semi.flatten(), bins=np.arange(0, 255))
    axis[0].title.set_text('Исходное изображение')

    axis[1].hist(x=transformed.flatten(), bins=np.arange(0, 255))
    axis[1].title.set_text('Преобразованное изображение')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'histograms', 'image.png'))

    matrix = haralik(semi.astype(np.uint8))
    result = Image.fromarray(matrix.astype(np.uint8), 'L')
    result.save(os.path.join(base_path, 'haralik', 'image.png'))

    t_matrix = haralik(transformed.astype(np.uint8))
    t_result = Image.fromarray(t_matrix.astype(np.uint8), 'L')
    t_result.save(os.path.join(base_path, 'haralik_contrasted', 'image.png'))

    con_value = CON(matrix)
    con_contrasted_value = CON(t_matrix)
    lun_value = LUN(matrix)
    lun_contrasted_value = LUN(t_matrix)

    with open(os.path.join(base_path, 'results.txt'), 'w') as f:
        f.write(f"CON: {con_value}\n")
        f.write(f"CON (contrasted): {con_contrasted_value}\n")
        f.write(f"LUN: {lun_value}\n")
        f.write(f"LUN (contrasted): {lun_contrasted_value}\n")

if __name__ == '__main__':
    main()
