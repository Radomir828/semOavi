import os
import numpy as np
from PIL import Image

def to_grayscale(image_array):
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image.astype(np.uint8)

def gradient_matrices(image_array):
    # Вычисление градиентных матриц Gx и Gy 
    Gx = np.zeros(image_array.shape)
    Gy = np.zeros(image_array.shape)
    
    # Операторы Собеля для вычисления градиентов
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Применение фильтров
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            Gx[i, j] = np.sum(sobel_x * image_array[i-1:i+2, j-1:j+2])
            Gy[i, j] = np.sum(sobel_y * image_array[i-1:i+2, j-1:j+2])
    
    return Gx, Gy

def normalize(image_array):
    normalized = (255 * (image_array - np.min(image_array)) / np.ptp(image_array)).astype(np.uint8)
    return normalized

def binary_threshold(image_array, threshold):
    binary_image = (image_array > threshold) * 255
    return binary_image.astype(np.uint8)

def process_image(input_image_path, output_folder):

    img_src = Image.open(input_image_path)
    img_src_arr = np.array(img_src)
    
    # Преобразование к полутоновому изображению
    img_gray_array = to_grayscale(img_src_arr)
    img_gray = Image.fromarray(img_gray_array, 'L')
    gray_path = os.path.join(output_folder, 'gray_' + os.path.basename(input_image_path))
    img_gray.save(gray_path)
    
    # Вычисление градиентных матриц Gx и Gy
    Gx, Gy = gradient_matrices(img_gray_array)
    Gx_norm = normalize(Gx)
    Gy_norm = normalize(Gy)
    G = np.abs(Gx) + np.abs(Gy)
    G_norm = normalize(G)
    
    # Сохранение градиентных матриц
    Gx_path = os.path.join(output_folder, 'Gx_' + os.path.basename(input_image_path))
    Gy_path = os.path.join(output_folder, 'Gy_' + os.path.basename(input_image_path))
    G_path = os.path.join(output_folder, 'G_' + os.path.basename(input_image_path))
    Image.fromarray(Gx_norm).save(Gx_path)
    Image.fromarray(Gy_norm).save(Gy_path)
    Image.fromarray(G_norm).save(G_path)
    
    threshold = 50  # Порог подбирается опытным путём
    G_binary = binary_threshold(G_norm, threshold)
    G_binary_path = os.path.join(output_folder, 'G_binary_' + os.path.basename(input_image_path))
    Image.fromarray(G_binary).save(G_binary_path)
    
    return gray_path, Gx_path, Gy_path, G_path, G_binary_path

def create_report(input_folder, report_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("# Лабораторная работа №4. Выделение контуров на изображении\n\n")
        
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            gray_path, Gx_path, Gy_path, G_path, G_binary_path = process_image(image_path, output_folder)
            
            report.write(f"## {os.path.splitext(image_file)[0]}\n\n")
            report.write("### Исходное цветное изображение\n\n")
            report.write(f"![Исходное цветное изображение]({image_path})\n\n")
            report.write("### Полутоновое изображение\n\n")
            report.write(f"![Полутоновое изображение]({gray_path})\n\n")
            report.write("### Градиентная матрица Gx\n\n")
            report.write(f"![Градиентная матрица Gx]({Gx_path})\n\n")
            report.write("### Градиентная матрица Gy\n\n")
            report.write(f"![Градиентная матрица Gy]({Gy_path})\n\n")
            report.write("### Градиентная матрица G\n\n")
            report.write(f"![Градиентная матрица G]({G_path})\n\n")
            report.write("### Бинаризованная градиентная матрица G\n\n")
            report.write(f"![Бинаризованная градиентная матрица G]({G_binary_path})\n\n")

if __name__ == '__main__':
    input_folder = 'input'
    report_path = 'report.md'
    output_folder = 'output'
    create_report(input_folder, report_path, output_folder)
