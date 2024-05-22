from PIL import Image
import numpy as np

def knn_resampling(old_image, scale):
    height, width = old_image.shape[:2]
    new_width = round(scale * width)
    new_height = round(scale * height)

    new_image = np.zeros((new_height, new_width, old_image.shape[2]), dtype=old_image.dtype)

    for x in range(new_width):
        for y in range(new_height):
            src_x = min(int(round(float(x) / float(new_width) * float(width))), width - 1)
            src_y = min(int(round(float(y) / float(new_height) * float(height))), height - 1)

            new_image[y, x] = old_image[src_y, src_x]

    return new_image


if __name__ == '__main__':

    image_name = 'muar.png'
    img_src = Image.open('input/' + image_name).convert('RGB')
    img_src_arr = np.array(img_src)

    M = 2  # Фактор растяжения
    N = 3  # Фактор сжатия
    K = M / N  # Фактор передискретизации

    img_stretch = knn_resampling(img_src_arr, M)
    Image.fromarray(img_stretch.astype(np.uint8), 'RGB').save('output/muar_stretch.png')

    img_compress = knn_resampling(img_src_arr, 1 / N)
    Image.fromarray(img_compress.astype(np.uint8), 'RGB').save('output/muar_compress.png')

    img_resample_two_pass = knn_resampling(knn_resampling(img_src_arr, M), 1 / N)
    Image.fromarray(img_resample_two_pass.astype(np.uint8), 'RGB').save('output/muar_resample.png')

    img_resample_one_pass = knn_resampling(img_src_arr, K)
    Image.fromarray(img_resample_one_pass.astype(np.uint8), 'RGB').save('output/muar_resample_single.png')

