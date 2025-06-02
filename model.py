import os
import shutil
from pathlib import Path

def copy_images_with_prefix(source_folder, destination_folder, ignored_folders=None, image_extensions=None):
    """
    Copy all images from source folder to destination folder, ignoring specified folders
    and prefixing filenames with their source folder name.
    
    Args:
        source_folder (str): Path to the source folder containing images
        destination_folder (str): Path to the destination folder
        ignored_folders (list): List of folder names to ignore (default: common system folders)
        image_extensions (list): List of image file extensions to copy (default: common image formats)
    """
    
    # Default ignored folders
    if ignored_folders is None:
        ignored_folders = ['.git', '__pycache__', 'node_modules', '.DS_Store', 'Thumbs.db']
    
    # Default image extensions
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.ico']
    
    # Convert to lowercase for case-insensitive comparison
    image_extensions = [ext.lower() for ext in image_extensions]
    
    # Create destination folder if it doesn't exist
    Path(destination_folder).mkdir(parents=True, exist_ok=True)
    
    # Counter for tracking copied files
    copied_count = 0
    skipped_count = 0
    
    print(f"Starting image copy from '{source_folder}' to '{destination_folder}'")
    print(f"Ignoring folders: {ignored_folders}")
    print(f"Looking for extensions: {image_extensions}")
    print("-" * 50)
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(source_folder):
        # Remove ignored folders from dirs list to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if d not in ignored_folders]
        
        # Get the relative path from source folder
        rel_path = os.path.relpath(root, source_folder)
        
        # Create folder prefix (use folder name, replace path separators with underscores)
        if rel_path == '.':
            folder_prefix = os.path.basename(source_folder)
        else:
            folder_prefix = rel_path.replace(os.sep, '_')
        
        # Process each file in current directory
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if file is an image
            if file_ext in image_extensions:
                # Create new filename with folder prefix
                file_name, file_extension = os.path.splitext(file)
                new_filename = f"{folder_prefix}_{file_name}{file_extension}"
                
                # Handle duplicate filenames by adding a counter
                destination_path = os.path.join(destination_folder, new_filename)
                counter = 1
                while os.path.exists(destination_path):
                    file_name_base, file_ext_base = os.path.splitext(new_filename)
                    new_filename = f"{file_name_base}_{counter}{file_ext_base}"
                    destination_path = os.path.join(destination_folder, new_filename)
                    counter += 1
                
                try:
                    # Copy the file
                    shutil.copy2(file_path, destination_path)
                    print(f"Copied: {file} -> {new_filename}")
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {file}: {str(e)}")
                    skipped_count += 1
            else:
                # Skip non-image files silently (you can uncomment the line below to see skipped files)
                # print(f"Skipped (not an image): {file}")
                pass
    
    print("-" * 50)
    print(f"Copy completed!")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped/failed: {skipped_count}")

# Example usage
if __name__ == "__main__":
    # Configuration - modify these paths and settings as needed
    SOURCE_FOLDER = "path/to/your/source/folder"
    DESTINATION_FOLDER = "new_folder"
    
    # Folders to ignore (add more as needed)
    IGNORED_FOLDERS = [
        '.git',
        '__pycache__',
        'node_modules',
        '.DS_Store',
        'Thumbs.db',
        'temp',
        'cache',
        'backup'  # Add your specific folders here
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
        ignored_folders=IGNORED_FOLDERS,
        image_extensions=IMAGE_EXTENSIONS
    )