import os
import shutil
from PIL import Image

def process_images_and_copy_folder(source_dir, dest_dir):
    """
    Recursively copies a folder structure, converting JFIF images to JPEG,
    counts total images, and reports the number of converted images.

    Args:
        source_dir (str): The path to the source directory.
        dest_dir (str): The path to the destination directory.
                        It will be created if it doesn't exist.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    if not os.path.exists(dest_dir):
        try:
            os.makedirs(dest_dir)
            print(f"Created destination directory: '{dest_dir}'")
        except OSError as e:
            print(f"Error creating destination directory '{dest_dir}': {e}")
            return
    
    print(f"Processing files from '{source_dir}' to '{dest_dir}'...")
    converted_count = 0  # Initialize counter for converted images
    total_image_count = 0 # Initialize counter for total images

    # Common image file extensions (add more if needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif', '.jfif', '.jfi']

    for root, dirs, files in os.walk(source_dir):
        # Determine the corresponding destination path
        relative_path = os.path.relpath(root, source_dir)
        current_dest_dir = os.path.join(dest_dir, relative_path)

        # Create subdirectories in the destination folder if they don't exist
        if not os.path.exists(current_dest_dir):
            try:
                os.makedirs(current_dest_dir)
            except OSError as e:
                print(f"Error creating subdirectory '{current_dest_dir}': {e}")
                continue 

        for filename in files:
            source_file_path = os.path.join(root, filename)
            dest_file_path_original = os.path.join(current_dest_dir, filename)
            
            file_base, file_ext = os.path.splitext(filename)
            file_ext_lower = file_ext.lower()

            # Count if it's an image file
            if file_ext_lower in image_extensions:
                total_image_count += 1

            # Process JFIF conversion
            if file_ext_lower in ['.jfif', '.jfi']:
                dest_file_path_converted = os.path.join(current_dest_dir, file_base + ".jpg")
                try:
                    with Image.open(source_file_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dest_file_path_converted, "JPEG", quality=95) 
                    print(f"Converted '{source_file_path}' to '{dest_file_path_converted}'")
                    converted_count += 1 # Increment counter on successful conversion
                except Exception as e:
                    print(f"Error converting file '{source_file_path}': {e}. Copying as is.")
                    try:
                        shutil.copy2(source_file_path, dest_file_path_original)
                        print(f"Copied (fallback) '{source_file_path}' to '{dest_file_path_original}'")
                    except Exception as copy_e:
                        print(f"Error copying (fallback) file '{source_file_path}': {copy_e}")
            else:
                # For all other files (including non-JFIF images), copy them as is
                try:
                    shutil.copy2(source_file_path, dest_file_path_original)
                except Exception as e:
                    print(f"Error copying file '{source_file_path}': {e}")

    print("\nProcessing complete.")
    print(f"Found a total of {total_image_count} image file(s) in '{source_dir}'.")
    print(f"Successfully converted {converted_count} JFIF image(s) to JPG format in '{dest_dir}'.")
    print(f"Other files and folders have been copied as is.")

if __name__ == "__main__":
    # --- IMPORTANT: SET YOUR FOLDER PATHS HERE ---
    # Example for Windows: "C:\\Users\\YourUser\\Desktop\\MySourceFolder"
    # Example for macOS/Linux: "/Users/YourUser/Desktop/MySourceFolder"
    
    source_folder = "/home/treerspeaking/src/python/cabdefect/AI_CAB" 
    destination_folder = "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY"

    # --- --- --- --- --- --- --- --- --- --- --- ---

    # if source_folder == "YOUR_SOURCE_FOLDER_PATH" or destination_folder == "YOUR_DESTINATION_FOLDER_PATH":
    #     print("--------------------------------------------------------------------")
    #     print("IMPORTANT: Please open the script and set the")
    #     print("'source_folder' and 'destination_folder' variables before running!")
    #     print("--------------------------------------------------------------------")
    # else:
    process_images_and_copy_folder(source_folder, destination_folder)
