import os
import cv2
import concurrent.futures
from tqdm import tqdm

# set the directory path where the images are stored
dir_path = './training/next_frame/next_frame_dataset/'

# define a function to convert a single image
def convert_image(sub_dir, filename):
    if filename.endswith('.jpg'):
        # read the image in BGR format
        img = cv2.imread(os.path.join(dir_path, sub_dir, filename))

        # convert the image from BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # save the image in PNG format
        new_filename = os.path.splitext(filename)[0] + '.png'
        cv2.imwrite(os.path.join(dir_path, sub_dir, new_filename), img)

        # delete the original JPG file
        os.remove(os.path.join(dir_path, sub_dir, filename))

# loop through the subdirectories
for sub_dir in ['next_frame', 'first_frame']:
    sub_dir_path = os.path.join(dir_path, sub_dir)

    # create a list of all the image filenames in the subdirectory
    filenames = [f for f in os.listdir(sub_dir_path) if f.endswith('.jpg')]

    # use multithreading to convert the images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # define a function to convert an image and update the progress bar
        def update(pbar, filename):
            convert_image(sub_dir, filename)
            pbar.update()

        # create a progress bar with the number of images to convert
        with tqdm(total=len(filenames), desc=f'Converting images in {sub_dir} directory') as pbar:
            # submit each image for conversion and update the progress bar
            futures = [executor.submit(update, pbar, filename) for filename in filenames]

            # wait for all the conversions to finish
            for future in concurrent.futures.as_completed(futures):
                pass
