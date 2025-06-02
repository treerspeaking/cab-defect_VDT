import os
import shutil

def copy_images_with_prefix(source_dir, dest_dir, ignore_folders_list):
    """
    Copies images from subfolders of a source directory to a destination directory,
    prefixing the filename with the subfolder name and ignoring specified folders.

    Args:
        source_dir (str): The path to the source directory containing subfolders with images.
        dest_dir (str): The path to the destination directory where images will be copied.
        ignore_folders_list (list): A list of folder names (strings) to ignore.
    """
    # --- Input Validation and Setup ---
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir)
            print(f"Created destination directory: '{dest_dir}'")
        except OSError as e:
            print(f"Error: Could not create destination directory '{dest_dir}'. {e}")
            return
    elif not os.path.isdir(dest_dir):
        print(f"Error: Destination path '{dest_dir}' exists but is not a directory.")
        return

    # Ensure ignore_folders_list is a list of strings, handling potential None
    if ignore_folders_list is None:
        ignore_folders = []
    else:
        ignore_folders = [str(folder).strip() for folder in ignore_folders_list if str(folder).strip()]
    
    print(f"Ignoring folders: {ignore_folders}")

    copied_files_count = 0
    skipped_folders_count = 0

    # --- Iterate through source directory ---
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)

        if os.path.isdir(item_path):
            folder_name = item # This is the subfolder name
            if folder_name in ignore_folders:
                print(f"Skipping ignored folder: '{folder_name}'")
                skipped_folders_count += 1
                continue

            print(f"Processing folder: '{folder_name}'...")
            # --- Iterate through files in the subfolder ---
            for filename in os.listdir(item_path):
                source_file_path = os.path.join(item_path, filename)

                # Basic check if it's a file (you might want to add specific image extension checks)
                # Example: allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif')
                # if os.path.isfile(source_file_path) and filename.lower().endswith(allowed_extensions):
                if os.path.isfile(source_file_path):
                    # --- Construct new filename and path ---
                    # Sanitize folder_name to be used in filename (optional, but good practice)
                    sanitized_folder_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in folder_name)
                    new_filename = f"{sanitized_folder_name}_{filename}"
                    dest_file_path = os.path.join(dest_dir, new_filename)

                    # --- Copy the file ---
                    try:
                        shutil.copy2(source_file_path, dest_file_path) # copy2 preserves metadata
                        print(f"  Copied '{filename}' to '{new_filename}'")
                        copied_files_count += 1
                    except Exception as e:
                        print(f"  Error copying '{filename}': {e}")
                else:
                    print(f"  Skipping non-file item: '{filename}' in folder '{folder_name}'")
        else:
            print(f"Skipping non-directory item at source root: '{item}'")


    print("\n--- Summary ---")
    print(f"Total files copied: {copied_files_count}")
    print(f"Total folders skipped (due to ignore list): {skipped_folders_count}")
    if copied_files_count > 0 or skipped_folders_count > 0 : # Only print if something was processed
        print(f"Files processed. Check destination: '{os.path.abspath(dest_dir)}'")
    else:
        print("No files were copied or folders skipped based on the provided paths and ignore list.")


if __name__ == "__main__":
    print("--- Image Copy Utility (Hardcoded Paths) ---")
    
    # --- Define your paths and ignore list here ---
    # IMPORTANT: Replace these with your actual paths!
    # Example for Windows: source_folder = r"C:\Users\YourUser\Pictures\MyPhotoCollection"
    # Example for macOS/Linux: source_folder = "/home/youruser/pictures/my_photo_collection"
    
    source_folder = "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY"  # Replace with your actual source folder path
    destination_folder = "/home/treerspeaking/src/python/cabdefect/train_data/unlabeled" # Replace with your actual destination folder path
    
    # Define folders to ignore as a list of strings
    folders_to_ignore = [
        '/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY/Tủ hộp cáp/Ảnh sai',
        '/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY/Các dây nhảy được đánh số thứ tự/Ảnh đúng',
        '/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY/Các đầu adapter chưa sử dụng phải có đầu bịt chống bụi',
    ] # Add or remove folder names

    print(f"Source directory: '{os.path.abspath(source_folder)}'")
    print(f"Destination directory: '{os.path.abspath(destination_folder)}'")
    print(f"Folders to ignore: {folders_to_ignore}")

    # --- Create dummy directories and files for testing (optional) ---
    # You can uncomment and adapt this section if you want to quickly test the script
    # with placeholder directories and files.
    # print("\n--- Setting up test environment (if not exists) ---")
    # if not os.path.exists(source_folder):
    #     os.makedirs(source_folder)
    #     print(f"Created dummy source: {source_folder}")
    # if not os.path.exists(os.path.join(source_folder, "folderA")): os.makedirs(os.path.join(source_folder, "folderA"))
    # if not os.path.exists(os.path.join(source_folder, "folderB")): os.makedirs(os.path.join(source_folder, "folderB"))
    # if not os.path.exists(os.path.join(source_folder, "folderC_to_ignore")): os.makedirs(os.path.join(source_folder, "folderC_to_ignore"))
    # if not os.path.exists(os.path.join(source_folder, "drafts")): os.makedirs(os.path.join(source_folder, "drafts"))
    #
    # # Create some dummy files
    # dummy_files_info = [
    #     (os.path.join(source_folder, "folderA", "image1.jpg"), "dummy content A1"),
    #     (os.path.join(source_folder, "folderA", "image2.png"), "dummy content A2"),
    #     (os.path.join(source_folder, "folderB", "picture.gif"), "dummy content B1"),
    #     (os.path.join(source_folder, "folderC_to_ignore", "secret.jpg"), "dummy content C1"),
    #     (os.path.join(source_folder, "drafts", "temp.doc"), "dummy content D1"),
    #     (os.path.join(source_folder, "not_a_folder.txt"), "dummy content root")
    # ]
    # for f_path, content in dummy_files_info:
    #     if not os.path.exists(f_path):
    #         try:
    #             with open(f_path, 'w') as f: f.write(content)
    #             print(f"Created dummy file: {f_path}")
    #         except Exception as e:
    #             print(f"Error creating dummy file {f_path}: {e}")
    # print("--- Test environment setup complete --- \n")


    # --- Run the main function ---
    copy_images_with_prefix(source_folder, destination_folder, folders_to_ignore)
