from PIL import Image

def to_semitone(image_path: str):
    image = Image.open(image_path)
    grayscale_image = image.convert('L')
    return grayscale_image