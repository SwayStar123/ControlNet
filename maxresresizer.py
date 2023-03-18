import cv2
import os
import concurrent.futures
from tqdm import tqdm

def resize_image(img_name, input_dir_first_frame, input_dir_next_frame, output_dir_first_frame, output_dir_next_frame, max_pixel_count):
    img_path_first_frame = os.path.join(input_dir_first_frame, img_name)
    img_path_next_frame = os.path.join(input_dir_next_frame, img_name)

    img_first_frame = cv2.imread(img_path_first_frame)
    img_next_frame = cv2.imread(img_path_next_frame)

    height, width = img_first_frame.shape[:2]
    pixel_count = height * width

    if pixel_count > max_pixel_count:
        aspect_ratio = float(width) / float(height)
        new_height = int((max_pixel_count / aspect_ratio) ** 0.5)
        new_width = int(aspect_ratio * new_height)

        # Round down new_height and new_width to the closest multiple of 64
        new_height = (new_height // 64) * 64
        new_width = (new_width // 64) * 64

        img_resized_first_frame = cv2.resize(img_first_frame, (new_width, new_height))
        img_resized_next_frame = cv2.resize(img_next_frame, (new_width, new_height))

        output_path_first_frame = os.path.join(output_dir_first_frame, img_name)
        output_path_next_frame = os.path.join(output_dir_next_frame, img_name)

        cv2.imwrite(output_path_first_frame, img_resized_first_frame)
        cv2.imwrite(output_path_next_frame, img_resized_next_frame)
    else:
        output_path_first_frame = os.path.join(output_dir_first_frame, img_name)
        output_path_next_frame = os.path.join(output_dir_next_frame, img_name)

        cv2.imwrite(output_path_first_frame, img_first_frame)
        cv2.imwrite(output_path_next_frame, img_next_frame)

def resize_images(input_dir_first_frame, input_dir_next_frame, output_dir_first_frame, output_dir_next_frame, max_pixel_count, num_threads=32):
    if not os.path.exists(output_dir_first_frame):
        os.makedirs(output_dir_first_frame)

    if not os.path.exists(output_dir_next_frame):
        os.makedirs(output_dir_next_frame)

    img_names = os.listdir(input_dir_first_frame)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(img_names), desc="Resizing images") as progress_bar:
            futures = [executor.submit(resize_image, img_name, input_dir_first_frame, input_dir_next_frame, output_dir_first_frame, output_dir_next_frame, max_pixel_count) for img_name in img_names]

            for future in concurrent.futures.as_completed(futures):
                future.result()
                progress_bar.update(1)

input_dir_first_frame = './training/next_frame/next_frame_dataset_resized/first_frame/'
input_dir_next_frame = './training/next_frame/next_frame_dataset_resized/next_frame/'
output_dir_first_frame = './training/next_frame/next_frame_dataset_resized/first_frame_resized/'
output_dir_next_frame = './training/next_frame/next_frame_dataset_resized/next_frame_resized/'
max_pixel_count = 512 * 512
resize_images(input_dir_first_frame, input_dir_next_frame, output_dir_first_frame, output_dir_next_frame, max_pixel_count)
