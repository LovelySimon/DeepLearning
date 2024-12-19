import os
def delete_images_ending_with_2(folder_path):
    """
    Deletes image files in the specified folder that end with the digit '2'.

    :param folder_path: Path to the folder containing the image files.
    """
    # Define common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it is a file and ends with '2' (before the extension)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(file_name)
            if ext.lower() in image_extensions and name.endswith('2'):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# Example usage
folder_path = "C://Users//Administrator//Desktop//GetAppImage20241211//single_4"  # Replace with your folder path
delete_images_ending_with_2(folder_path)
