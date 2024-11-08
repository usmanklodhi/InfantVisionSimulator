import cv2
import os
import hashlib

def dhash(image, hash_size=8):
    # Resize the input image, adding a single column to make the width hash_size + 1
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    
    # Compute the relative horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def delete_duplicate_images(folder_path):
    hashes = {}
    deleted_count = 0
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        # Check if it's an image file
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read image {filename}, skipping.")
            continue

        # Compute the hash for the image
        # Hash the image
        image_hash = dhash(image)

        # Check if this hash already exists
        if image_hash in hashes:
            os.remove(image_path)
            deleted_count += 1
            print(f"Deleted duplicate image: {filename}")
        else:
            # Store the hash and the file path
            hashes[image_hash] = filename
            print(f"Image is unique, kept: {filename}")
    print(f"Total images deleted: {deleted_count}")

# Specify the folder path containing the images
count = 0
folder_path = "Humans"
delete_duplicate_images(folder_path)

