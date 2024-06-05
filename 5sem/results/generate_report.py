import os
import csv
import matplotlib.pyplot as plt

def load_features_from_csv(csv_file):
    features = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            char = row['character']
            features[char] = {
                'weight': row['weights'],
                'rel_weight': row['specific_weights'],
                'center': row['center_of_mass'],
                'rel_center': row['norm_center_of_mass'],
                'inertia': row['moments'],
                'rel_inertia': row['norm_moments']
            }
    return features

def save_report(features, profile_folder, report_file):
    with open(report_file, mode='w', encoding='utf-8') as file:
        file.write("# Лабораторная работа №5. Выделение признаков символов\n\n")
        
        for char in list(features.keys())[:4]:
            file.write(f"## Буква {char.upper()}\n")
            
            profile_x_path = os.path.join(profile_folder, 'x', f'{char}_x.png')
            profile_y_path = os.path.join(profile_folder, 'y', f'{char}_y.png')
            
            if os.path.exists(profile_x_path) and os.path.exists(profile_y_path):
                file.write(f"![Profile X](profiles/x/{char}_x.png) ![Profile Y](profiles/y/{char}_y.png)\n")
            
            feature = features[char]
            file.write(f"- Вес (масса чёрного) каждой четверти изображения символа = {feature['weight']}\n")
            file.write(f"- Удельный вес (вес, нормированный к четверти площади) = {feature['rel_weight']}\n")
            file.write(f"- Координаты центра тяжести = {feature['center']}\n")
            file.write(f"- Нормированные координаты центра тяжести = {feature['rel_center']}\n")
            file.write(f"- Осевые моменты инерции по горизонтали и вертикали = {feature['inertia']}\n")
            file.write(f"- Нормированные осевые моменты инерции = {feature['rel_inertia']}\n")
            file.write("\n")

if __name__ == '__main__':
    csv_file = 'features.csv'
    profile_folder = 'profiles'
    report_file = 'report.md'
    
    features = load_features_from_csv(csv_file)
    save_report(features, profile_folder, report_file)
