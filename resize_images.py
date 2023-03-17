import os
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def nearest_multiple_of_64(x):
    return (x + 63) // 64 * 64

def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def process_image(idx, item, input_base_path, output_base_path):
    hint_filename = os.path.join(input_base_path, 'first_frame', f'{idx}.png')
    target_filename = os.path.join(input_base_path, 'next_frame', f'{idx}.png')

    hint = cv2.imread(hint_filename)
    target = cv2.imread(target_filename)

    # Determine the new dimensions
    height, width, _ = hint.shape
    new_height = nearest_multiple_of_64(height)
    new_width = nearest_multiple_of_64(width)

    # Resize the images
    resized_hint = resize_image(hint, new_width, new_height)
    resized_target = resize_image(target, new_width, new_height)

    # Save the resized images
    output_hint_filename = os.path.join(output_base_path, 'first_frame', f'{idx}.png')
    output_target_filename = os.path.join(output_base_path, 'next_frame', f'{idx}.png')

    cv2.imwrite(output_hint_filename, resized_hint)
    cv2.imwrite(output_target_filename, resized_target)

def main():
    with open('./training/next_frame/frame_metadata.json', 'rt') as f:
        data = json.load(f)

    input_base_path = './training/next_frame/next_frame_dataset'
    output_base_path = './training/next_frame/next_frame_dataset_resized'
    
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'first_frame'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'next_frame'), exist_ok=True)

    with ThreadPoolExecutor() as executor:
        tasks = []
        for idx, item in data.items():
            task = executor.submit(process_image, idx, item, input_base_path, output_base_path)
            tasks.append(task)

        for task in tqdm(tasks, desc='Resizing images', unit='image'):
            task.result()

if __name__ == '__main__':
    main()
