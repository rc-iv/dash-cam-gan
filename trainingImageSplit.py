import os
import random
import shutil


def split_dataset(base_dir, output_dir, train_ratio=0.8):
    classes = ["car", "traffic", "intersection", "trees", "road"]

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)

    for cls in classes:
        # Create class-specific folders in training and validation directories
        os.makedirs(os.path.join(output_dir, "train", cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "val", cls), exist_ok=True)

        # Get a list of all the image files for the current class
        image_files = [f for f in os.listdir(os.path.join(base_dir, cls)) if f.endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(image_files)

        # Calculate the number of training images for the current class
        num_train_images = int(len(image_files) * train_ratio)

        # Split the image files into training and validation sets
        train_images = image_files[:num_train_images]
        val_images = image_files[num_train_images:]

        # Copy the images to their respective output folders
        for img in train_images:
            shutil.copy(os.path.join(base_dir, cls, img), os.path.join(output_dir, "train", cls, img))
        for img in val_images:
            shutil.copy(os.path.join(base_dir, cls, img), os.path.join(output_dir, "val", cls, img))


base_dir = "../images"
output_dir = "../images/training_data"
split_dataset(base_dir, output_dir)
