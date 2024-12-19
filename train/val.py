import os
import random
import shutil

def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    Split images in the input directory into training and validation sets.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory where train/val folders will be created.
        train_ratio (float): Ratio of images to include in the training set.
    """
    # Create output directories for train and val
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get a list of all image files in the input directory
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Shuffle the images randomly
    random.shuffle(images)

    # Split the images into train and val sets
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    # Copy images to the respective directories
    for img in train_images:
        shutil.copy2(os.path.join(input_dir, img), os.path.join(train_dir, img))

    for img in val_images:
        shutil.copy2(os.path.join(input_dir, img), os.path.join(val_dir, img))

    print(f"Dataset split complete. {len(train_images)} images in train, {len(val_images)} images in val.")

# Example usage
input_folder = 'C://Users//Administrator//Desktop//GetAppImage20241211//single_4'
output_folder = 'C://Users//Administrator//Desktop//GetAppImage20241211//single_4_split'
split_dataset(input_folder, output_folder)
