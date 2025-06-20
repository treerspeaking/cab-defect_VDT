import os
import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict
import argparse

def find_duplicate_images(directory_path, hash_size=8):
    """
    Find duplicate images in the given directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory to scan
        hash_size: Size of the hash to use (larger = more sensitive)
        
    Returns:
        Dictionary with hash values as keys and lists of duplicate image paths as values
    """
    image_hashes = defaultdict(list)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Get all image files recursively
    image_paths = [
        path for path in Path(directory_path).rglob('*')
        if path.suffix.lower() in supported_formats
    ]
    
    print(f"Found {len(image_paths)} images to analyze")
    
    # Calculate hash for each image
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            # Using perceptual hash (pHash) which is good for finding visually identical images
            h = str(imagehash.phash(img, hash_size=hash_size))
            image_hashes[h].append(str(img_path))
            print(f"Processed: {img_path}", end="\r")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Filter out unique images
    duplicates = {h: paths for h, paths in image_hashes.items() if len(paths) > 1}
    
    return duplicates

def main():
    parser = argparse.ArgumentParser(description='Find duplicate images in a directory')
    parser.add_argument('directory', type=str, nargs='?', 
                        default='/home/treerspeaking/src/python/cabdefect/AI_CAB_COPY_data',
                        help='Directory to scan for duplicate images')
    parser.add_argument('--hash-size', type=int, default=8, 
                        help='Hash size to use (higher is more sensitive)')
    args = parser.parse_args()
    
    print(f"Scanning directory: {args.directory}")
    duplicate_groups = find_duplicate_images(args.directory, args.hash_size)
    
    # Display results
    if duplicate_groups:
        print(f"\nFound {len(duplicate_groups)} groups of duplicate images:")
        for i, (_, paths) in enumerate(duplicate_groups.items(), 1):
            print(f"\nGroup {i} ({len(paths)} images):")
            for path in paths:
                print(f"  - {path}")
    else:
        print("No duplicate images found.")

if __name__ == "__main__":
    main()