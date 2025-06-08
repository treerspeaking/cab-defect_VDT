from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Load your custom model
model = YOLO("/home/treerspeaking/src/python/cabdefect/runs/classify/goat2/weights/best.pt")

# Single image prediction
def predict_single_image(image_path):
    """Predict on a single image"""
    results = model(image_path)
    
    # Print results
    for result in results:
        print(f"Image: {result.path}")
        print(f"Predictions: {result.probs.top1}")  # Top prediction index
        print(f"Confidence: {result.probs.top1conf:.4f}")  # Confidence score
        print(f"Class names: {result.names}")  # All class names
        print(f"Top class: {result.names[result.probs.top1]}")  # Top predicted class name
        print("-" * 50)
    
    return results

# Batch prediction on multiple images
def predict_batch(image_folder):
    """Predict on all images in a folder"""
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_folder.glob(f'*{ext}'))
        image_paths.extend(image_folder.glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print("No images found in the specified folder!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Predict on all images
    results = model(image_paths)
    
    # Process results
    for result in results:
        print(f"Image: {Path(result.path).name}")
        print(f"Top class: {result.names[result.probs.top1]}")
        print(f"Confidence: {result.probs.top1conf:.4f}")
        print("-" * 30)
    
    return results

# Predict with confidence threshold
def predict_with_threshold(image_path, confidence_threshold=0.5):
    """Predict only if confidence is above threshold"""
    results = model(image_path, conf=confidence_threshold)
    
    for result in results:
        if result.probs.top1conf >= confidence_threshold:
            print(f"High confidence prediction:")
            print(f"Class: {result.names[result.probs.top1]}")
            print(f"Confidence: {result.probs.top1conf:.4f}")
        else:
            print(f"Low confidence prediction (below {confidence_threshold})")
            print(f"Class: {result.names[result.probs.top1]}")
            print(f"Confidence: {result.probs.top1conf:.4f}")
    
    return results

# Get top N predictions
def get_top_n_predictions(image_path, n=3):
    """Get top N predictions with probabilities"""
    results = model(image_path)
    
    for result in results:
        # Get all probabilities
        probs = result.probs.data.cpu().numpy()
        
        # Get top N indices
        top_n_indices = np.argsort(probs)[-n:][::-1]
        
        print(f"Top {n} predictions for {Path(image_path).name}:")
        for i, idx in enumerate(top_n_indices):
            class_name = result.names[idx]
            confidence = probs[idx]
            print(f"{i+1}. {class_name}: {confidence:.4f}")
        print("-" * 40)
    
    return results

# Save results to file
def predict_and_save_results(image_path, output_file="predictions.txt"):
    """Predict and save results to a text file"""
    results = model(image_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Image: {result.path}\n")
            f.write(f"Top class: {result.names[result.probs.top1]}\n")
            f.write(f"Confidence: {result.probs.top1conf:.4f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"Results saved to {output_file}")
    return results

# Example usage
if __name__ == "__main__":
    # Your image path
    image_path = "/home/treerspeaking/src/python/cabdefect/data_more_label/train_data/unlabeled/Lỗi đi dây ống lỏng bộ chia_Ảnh đạt_WhatsApp Image 2025-03-17 at 13.44.57_2b6a56c1.jpg"
    
    print("=== Single Image Prediction ===")
    predict_single_image(image_path)
    
    # print("\n=== Top 3 Predictions ===")
    # get_top_n_predictions(image_path, n=3)
    
    # print("\n=== Prediction with Threshold ===")
    # predict_with_threshold(image_path, confidence_threshold=0.7)
    
    # # Uncomment to test batch prediction
    # # print("\n=== Batch Prediction ===")
    # # folder_path = "/home/treerspeaking/src/python/cabdefect/data_more_label/train_data/unlabeled/"
    # # predict_batch(folder_path)
    
    # # Save results
    # predict_and_save_results(image_path)