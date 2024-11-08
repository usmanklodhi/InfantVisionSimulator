import cv2
import os
import numpy as np

def is_black_and_white(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False  # If image couldn't be read, skip it

    # Check if all channels are identical
    # Convert to float to avoid overflow in subtraction
    b, g, r = cv2.split(image.astype("float32"))
    if np.array_equal(b, g) and np.array_equal(g, r):
        return True
    return False

def delete_black_and_white_images(folder_path):
    deleted_count = 0
    
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Check if it's an image file
        if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
            continue

        # Check if image is black and white
        if is_black_and_white(image_path):
            os.remove(image_path)
            deleted_count += 1
            print(f"Deleted black and white image: {filename}")
        else:
            print(f"Image is colored, kept: {filename}")
    print(f"Total images deleted: {deleted_count}")

# Specify the folder path containing the images
folder_path = "Humans"
delete_black_and_white_images(folder_path)
