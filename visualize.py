import os
from pathlib import Path

def visualize_yolo_labels(images_dir, labels_dir, class_names=None, output_dir=None):
    """
    Visualize YOLO dataset labels with bounding boxes
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels (.txt files)
        class_names: List of class names
        output_dir: Directory to save visualized images (optional)
    """
    
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required. Install it with: python -m pip install opencv-python"
        ) from exc

    # Default class names if not provided
    if class_names is None:
        class_names = [f'class_{i}' for i in range(80)]  # Up to 80 classes
    
    # Create colors for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Load corresponding label file
        label_filename = img_path.stem + '.txt'
        label_path = Path(labels_dir) / label_filename
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Get color for this class
                        color = colors[class_id % len(colors)]
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add class name
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            # Draw background for text
                            cv2.rectangle(img, (x1, y1-20), (x1+len(class_name)*10, y1), color, -1)
                            cv2.putText(img, class_name, (x1, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
                            
                    except ValueError as e:
                        print(f"Error parsing line in {label_path}: {line}")
                        continue
        
        # Display or save image
        if output_dir:
            output_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(output_path), img)
            print(f"Saved: {output_path}")
        else:
            # Display image
            height, width = img.shape[:2]
            resized = cv2.resize(img, (width // 2, height // 2), 
                                interpolation=cv2.INTER_AREA)

            cv2.imshow('High Quality Resize', resized)

            print(f"Showing: {img_path.name}")
            print("Press any key to continue, 'q' to quit...")

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images")
    parser.add_argument("labels_dir", type=Path, help="Directory with YOLO .txt labels")
    parser.add_argument("images_dir", type=Path, help="Directory with source images")
    args = parser.parse_args()

    if not args.labels_dir.is_dir():
        parser.error(f"labels_dir does not exist or is not a directory: {args.labels_dir}")
    if not args.images_dir.is_dir():
        parser.error(f"images_dir does not exist or is not a directory: {args.images_dir}")

    visualize_yolo_labels(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
    )


if __name__ == "__main__":
    main()
