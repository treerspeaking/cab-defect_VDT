import os
import shutil
from pathlib import Path

def copy_images_with_prefix(source_folder, destination_folder, ignore_folders=None, image_extensions=None):
    """
    Copy all images from nested folders to a destination folder with folder prefix.
    
    Args:
        source_folder (str): Path to the source folder containing subfolders with images
        destination_folder (str): Path where all images will be copied
        ignore_folders (list): List of folder names to ignore (case-insensitive)
        image_extensions (list): List of image file extensions to copy
    """
    
    # Default image extensions if not provided
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg']
    
    # Default ignore folders if not provided
    if ignore_folders is None:
        ignore_folders = []
    
    # Convert ignore folders to lowercase for case-insensitive comparison
    ignore_folders_lower = [folder.lower() for folder in ignore_folders]
    
    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    # Convert to Path objects
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    
    copied_count = 0
    skipped_folders = set()
    
    print(f"Starting image copy from: {source_folder}")
    print(f"Destination: {destination_folder}")
    print(f"Ignoring folders: {ignore_folders}")
    print(f"Image extensions: {image_extensions}")
    print("-" * 50)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_path):
        current_folder = Path(root)
        
        # Get the relative path from source to current folder
        try:
            relative_path = current_folder.relative_to(source_path)
            folder_name = str(relative_path) if str(relative_path) != '.' else 'root'
        except ValueError:
            # If relative_to fails, use the folder name
            folder_name = current_folder.name
        
        print(f"Processing folder: {folder_name}")
        
        # Check if current folder path should be ignored
        current_folder_str = str(current_folder)
        if current_folder_str.lower() in ignore_folders_lower:
            skipped_folders.add(current_folder_str)
            print(f"Skipping ignored folder: {current_folder_str}")
            continue
        
        # Check if any parent folder should be ignored
        skip_folder = False
        for ignore_path in ignore_folders:
            if current_folder_str.startswith(ignore_path):
                skip_folder = True
                skipped_folders.add(ignore_path)
                break
        
        if skip_folder:
            print(f"Skipping folder (parent ignored): {current_folder_str}")
            continue
        
        # Process files in current folder
        for file in files:
            file_path = current_folder / file
            file_extension = file_path.suffix.lower()
            
            # Check if file is an image
            if file_extension in [ext.lower() for ext in image_extensions]:
                # Create new filename with folder prefix (replace path separators with underscores)
                safe_folder_prefix = folder_name.replace('/', '_').replace('\\', '_')
                new_filename = f"{safe_folder_prefix}_{file}"
                
                # Handle duplicate filenames by adding a counter
                counter = 1
                original_new_filename = new_filename
                while (dest_path / new_filename).exists():
                    name_part, ext_part = os.path.splitext(original_new_filename)
                    new_filename = f"{name_part}_{counter}{ext_part}"
                    counter += 1
                
                destination_file = dest_path / new_filename
                
                try:
                    shutil.copy2(file_path, destination_file)
                    print(f"Copied: {file} -> {new_filename}")
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {file}: {e}")
    
    print("-" * 50)
    print(f"Copy completed!")
    print(f"Total images copied: {copied_count}")
    if skipped_folders:
        print(f"Folders skipped: {', '.join(sorted(skipped_folders))}")


# Example usage
if __name__ == "__main__":
    # Configuration - modify these paths and settings as needed
    SOURCE_FOLDER = "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data"
    DESTINATION_FOLDER = "data_more_label/train_data/unlabeled"
    
    # Folders to ignore (add more as needed)
    # IGNORED_FOLDERS = [
    #     '/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Tủ hộp cáp/Ảnh sai', #han_open
    #     '/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Các dây nhảy được đánh số thứ tự/Ảnh đúng', # han_close,
    #     '/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Các đầu adapter chưa sử dụng phải có đầu bịt chống bụi', # han_close,
    #     "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Ảnh bộ chia", # chia
    #     "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Ảnh cố định đầu cáp_BD THC", # dau cap
    #     # "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY/Ảnh đúng" #box_close,
    #     "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Cắt sai ống lỏng",
    #     "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Ảnh lắp dặt khay hàn, đúng quy cách có đậy nắp/Ảnh sai" # test_data
    # ]
    IGNORED_FOLDERS = [
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Ảnh cố định đầu cáp_BD THC", # dau cap
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Lỗi đi dây ống lỏng bộ chia/ẢNh lỗi",
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Tủ hộp cáp/Ảnh sai",
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Tủ hộp cáp/Ảnh đúng",
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Các đầu adapter chưa sử dụng phải có đầu bịt chống bụi/Ảnh đúng",
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Ảnh lắp dặt khay hàn, đúng quy cách có đậy nắp/Ảnh sai", # test_data
        "/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data/Ảnh bộ chia"
        
    ]
    
    # Image extensions to copy (add more if needed)
    IMAGE_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', 
        '.tiff', '.tif', '.webp', '.svg', '.ico', '.raw'
    ]
    
    # Run the copy operation
    copy_images_with_prefix(
        source_folder=SOURCE_FOLDER,
        destination_folder=DESTINATION_FOLDER,
        ignore_folders=IGNORED_FOLDERS,
        image_extensions=IMAGE_EXTENSIONS
    )