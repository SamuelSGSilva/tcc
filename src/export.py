import cv2
import yaml
from pathlib import Path

def create_yolo_yaml(output_dir, classes):
    """
    Creates the dataset.yaml file required by YOLOv8.
    """
    yaml_path = Path(output_dir) / "dataset.yaml"
    
    # YOLO expects paths relative to the dataset.yaml file or absolute paths.
    # We will use relative paths here.
    data = {
        'path': str(Path(output_dir).absolute()), # Absolute path to dataset root
        'train': 'images', # Default to treating all images as train for now
        'val': 'images',
        'test': 'images',
        'nc': len(classes),
        'names': classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def generate_yolo_annotation(binary_mask, class_id):
    """
    Finds the bounding box of the crack in the binary mask and returns the YOLO format string.
    Returns None if no bounding box could be found (e.g. empty mask).
    """
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # We can either bound the largest contour or all white pixels.
    # To be safe, let's bound all white pixels (the combined crack).
    # Get all non-zero pixels
    y_coords, x_coords = binary_mask.nonzero()
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None
        
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    img_h, img_w = binary_mask.shape
    
    # Calculate center, width, height in pixels
    w_px = x_max - x_min
    h_px = y_max - y_min
    center_x_px = x_min + (w_px / 2.0)
    center_y_px = y_min + (h_px / 2.0)
    
    # Normalize to 0-1
    center_x = center_x_px / img_w
    center_y = center_y_px / img_h
    w = w_px / img_w
    h = h_px / img_h
    
    # YOLO format: class_id center_x center_y width height
    annotation = f"{class_id} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}"
    
    return annotation
