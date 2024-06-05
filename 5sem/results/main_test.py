import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
import csv
import matplotlib.pyplot as plt

def trim_image(image):
    """Обрезка белых полей вокруг изображения."""
    inverted_image = ImageChops.invert(image)
    bbox = inverted_image.getbbox()
    return image.crop(bbox)

def generate_character_images(characters, font_path, font_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    font = ImageFont.truetype(font_path, font_size)
    
    for char in characters:
        image = Image.new('L', (font_size * 2, font_size * 2), 'white')
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), char, font=font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((font_size * 2 - width) / 2, (font_size * 2 - height) / 2), char, font=font, fill='black')
        
        image = trim_image(image)
        image_path = os.path.join(output_folder, f'{char}.png')
        image.save(image_path)

def calculate_features(image):
    """Расчет признаков изображения."""
    image_array = np.array(image)
    h, w = image_array.shape

    # Масса (вес) каждой четверти изображения символа
    q1 = image_array[:h//2, :w//2].sum()
    q2 = image_array[:h//2, w//2:].sum()
    q3 = image_array[h//2:, :w//2].sum()
    q4 = image_array[h//2:, w//2:].sum()

    # Удельный вес (вес, нормированный к четверти площади)
    area = (h * w) // 4
    uq1 = q1 / area
    uq2 = q2 / area
    uq3 = q3 / area
    uq4 = q4 / area

    # Координаты центра тяжести
    Y, X = np.indices(image_array.shape)
    total_mass = image_array.sum()
    center_x = (X * image_array).sum() / total_mass
    center_y = (Y * image_array).sum() / total_mass

    # Нормированные координаты центра тяжести
    norm_center_x = center_x / w
    norm_center_y = center_y / h

    # Осевые моменты инерции по горизонтали и вертикали
    Ix = ((Y - center_y)**2 * image_array).sum() / total_mass
    Iy = ((X - center_x)**2 * image_array).sum() / total_mass

    # Нормированные осевые моменты инерции
    norm_Ix = Ix / (h**2)
    norm_Iy = Iy / (w**2)

    # Профили X и Y
    profile_x = image_array.sum(axis=0)
    profile_y = image_array.sum(axis=1)

    return {
        'masses': (q1, q2, q3, q4),
        'unit_masses': (uq1, uq2, uq3, uq4),
        'center_of_mass': (center_x, center_y),
        'norm_center_of_mass': (norm_center_x, norm_center_y),
        'moments_of_inertia': (Ix, Iy),
        'norm_moments_of_inertia': (norm_Ix, norm_Iy),
        'profile_x': profile_x,
        'profile_y': profile_y
    }

def save_profiles(profiles, output_folder):
    """Сохранение профилей X и Y в виде столбчатых диаграмм."""
    x_folder = os.path.join(output_folder, 'x')
    y_folder = os.path.join(output_folder, 'y')
    
    os.makedirs(x_folder, exist_ok=True)
    os.makedirs(y_folder, exist_ok=True)
    
    for i, (profile_x, profile_y) in enumerate(profiles):
        plt.figure()
        plt.bar(range(len(profile_x)), profile_x)
        plt.xlabel('X')
        plt.ylabel('Sum')
        plt.title(f'Profile X for Character {i+1}')
        plt.savefig(os.path.join(x_folder, f'{i+1}.png'))
        plt.close()

        plt.figure()
        plt.barh(range(len(profile_y)), profile_y)
        plt.xlabel('Sum')
        plt.ylabel('Y')
        plt.title(f'Profile Y for Character {i+1}')
        plt.savefig(os.path.join(y_folder, f'{i+1}.png'))
        plt.close()

def save_features_to_csv(features, output_file):
    """Сохранение признаков в файл CSV."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow([
            'letter', 'weight', 'rel_weight', 'center', 'rel_center', 'inertia', 'rel_inertia'
        ])
        for char, feature in features.items():
            writer.writerow([
                char, 
                feature['masses'], 
                feature['unit_masses'], 
                feature['center_of_mass'], 
                feature['norm_center_of_mass'], 
                feature['moments_of_inertia'], 
                feature['norm_moments_of_inertia']
            ])

if __name__ == '__main__':
    characters = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
    font_path = 'C:/Windows/Fonts/times.ttf'
    font_size = 52
    output_folder = 'images'
    profile_output_folder = 'profiles'
    csv_output_file = 'features.csv'
    
    generate_character_images(characters, font_path, font_size, output_folder)
    
    features = {}
    profiles = []
    for char in characters:
        image_path = os.path.join(output_folder, f'{char}.png')
        image = Image.open(image_path).convert('L')
        feature = calculate_features(image)
        features[char] = feature
        profiles.append((feature['profile_x'], feature['profile_y']))
    
    save_features_to_csv(features, csv_output_file)
    save_profiles(profiles, profile_output_folder)
