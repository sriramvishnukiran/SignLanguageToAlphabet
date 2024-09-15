import shutil
import os
datasetPaths = ["./data", "./data_backup"]

finalPath = "./data_final"

alphabets = [chr(97+i) for i in range(26)]


def copy_folders(dest_path, src_paths, folder_name):
    # Create the destination folder
    dest_folder_path = os.path.join(dest_path, folder_name)
    os.makedirs(dest_folder_path, exist_ok=True)

    # Iterate over source paths
    for i, src_path in enumerate(src_paths):
        # Source folder path
        src_folder_path = os.path.join(src_path, folder_name)

        # Check if source folder exists
        if not os.path.exists(src_folder_path):
            print(
                f"Warning: Folder '{folder_name}' not found in source path {i + 1}")
            continue

        # Iterate over files in the source folder
        for file_name in os.listdir(src_folder_path):
            src_file_path = os.path.join(src_folder_path, file_name)

            # Create a new file name with suffix _i
            fname, ext = os.path.splitext(file_name)
            new_file_name = f"{fname}_{i}{ext}"
            dest_file_path = os.path.join(dest_folder_path, new_file_name)
            # print(dest_file_path)

            # Copy the file to the destination folder with the new name
            shutil.copy(src_file_path, dest_file_path)

    print(f"Copying completed. Files are copied to {dest_folder_path}")
