import os
import random
import math
import shutil
import yaml
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import zipfile
import argparse

def create_directory_structure():
    """Creates the necessary directory structure for the YOLO dataset."""
    # Main directories
    base_dirs = ['dataset', 'dataset/images', 'dataset/labels', 
                'dataset/images/train', 'dataset/images/val', 'dataset/images/test',
                'dataset/labels/train', 'dataset/labels/val', 'dataset/labels/test']
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    print("✓ Directory structure created successfully")

def get_card_files(cards_dir):
    """Gets the list of Pokemon card files, recursively traversing subdirectories."""
    card_files = []
    for root, dirs, files in os.walk(cards_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                card_files.append(os.path.join(root, f))
    return card_files

def get_background_files(backgrounds_dir):
    """Gets the list of background files."""
    return [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def preload_images(file_paths, description="images"):
    """
    Preloads all images from file paths into memory.
    Returns a list of PIL Image objects.
    """
    print(f"Preloading {len(file_paths)} {description}...")
    images = []
    for file_path in tqdm(file_paths, desc=f"Loading {description}"):
        try:
            img = Image.open(file_path).convert("RGBA")
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    print(f"✓ Successfully loaded {len(images)} {description}")
    return images

def calculate_iou(box1, box2):
    """
    Calculates the IoU (Intersection over Union) between two bounding boxes.
    box1, box2: each box is represented by [x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])  
    
    # Intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union_area = box1_area + box2_area - intersection_area
    
    # IoU
    if union_area <= 0:
        return 0
    return intersection_area / union_area

def place_card_on_background(card_img, bg_img, existing_boxes=None, max_attempts=50, scale_factor=None):
    """
    Places a Pokemon card on a background, using a grid-based approach.
    """
    if existing_boxes is None:
        existing_boxes = []
    
    # Get image dimensions
    bg_width, bg_height = bg_img.size
    card_width, card_height = card_img.size
    
    # Use the provided scale factor or generate a default one
    if scale_factor is None:
        scale_factor = random.uniform(0.08, 0.15)
    
    new_card_width = int(bg_width * scale_factor)
    new_card_height = int(card_height * (new_card_width / card_width))
    
    # Save original dimensions before rotation
    original_width = new_card_width
    original_height = new_card_height
    
    card_img = card_img.resize((new_card_width, new_card_height), Image.LANCZOS)
    
    # More limited random angle to maintain a tidy appearance
    # Reduces angle for large cards
    if scale_factor > 0.3:
        angle_deg = random.uniform(-5, 5)  # Smaller angle for large cards
    else:
        angle_deg = random.uniform(-15, 15)
        
    card_img = card_img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
    
    # Get new dimensions after rotation
    rotated_width, rotated_height = card_img.size
    
    # Define a grid
    grid_cols = 5  # Number of columns in the grid
    grid_rows = 4  # Number of rows in the grid
    cell_width = bg_width // grid_cols
    cell_height = bg_height // grid_rows
    
    # Try positions in the grid
    for _ in range(max_attempts):
        # Choose a grid cell
        grid_x = random.randint(0, grid_cols - 1)
        grid_y = random.randint(0, grid_rows - 1)
        
        # Calculate base position in the cell with some random variation
        base_x = grid_x * cell_width + random.randint(-10, 10)
        base_y = grid_y * cell_height + random.randint(-10, 10)
        
        # Calculate actual position, ensuring it's within the image
        paste_x = max(0, min(base_x, bg_width - rotated_width))
        paste_y = max(0, min(base_y, bg_height - rotated_height))
        
        # Calculate bounding box in absolute coordinates (pixels)
        x1 = paste_x
        y1 = paste_y
        x2 = paste_x + rotated_width
        y2 = paste_y + rotated_height
        
        # Check if the bounding box is valid
        if x1 >= bg_width or y1 >= bg_height or x2 <= 0 or y2 <= 0:
            continue
        
        # Ensure the bounding box is within the image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bg_width, x2)
        y2 = min(bg_height, y2)
        
        # Check if the bounding box is large enough
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        
        # Create bbox in pixel format for overlap checking
        pixel_bbox = [x1, y1, x2, y2]
        
        # Lower IoU threshold to allow closer cards
        overlap_detected = False
        for existing_box in existing_boxes:
            iou = calculate_iou(pixel_bbox, existing_box)
            if iou > 0.1:  # IoU threshold reduced to 10%
                overlap_detected = True
                break
                
        if not overlap_detected:
            # Create a copy of the background to avoid modifying the original
            result_img = bg_img.copy()
            
            # Create a mask for transparency
            if card_img.mode == 'RGBA':
                mask = card_img.split()[3]
            else:
                mask = None
                
            # Paste the card on the background
            result_img.paste(card_img, (paste_x, paste_y), mask)
            
            # Calculate the four corners of the rotated rectangle
            # For OBB, we need the coordinates of the 4 corners
            # Calculate the center of the card
            center_x = paste_x + rotated_width / 2
            center_y = paste_y + rotated_height / 2
            
            # Calculate the four corners before rotation (relative to center)
            # Use the original card dimensions, not those after rotation
            half_width = original_width / 2
            half_height = original_height / 2
            corners = [
                [-half_width, -half_height],  # Top-left
                [half_width, -half_height],   # Top-right
                [half_width, half_height],    # Bottom-right
                [-half_width, half_height]    # Bottom-left
            ]
            
            # Convert angle to radians (for rotation calculation)
            angle_rad = math.radians(-angle_deg)  # Negative because PIL rotates counterclockwise
            
            # Apply rotation and translate to center
            rotated_corners = []
            for x, y in corners:
                # Apply rotation
                x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
                y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
                
                # Translate to center and normalize for YOLO
                x_final = (center_x + x_rot) / bg_width
                y_final = (center_y + y_rot) / bg_height
                
                # Ensure values are in range [0, 1]
                x_final = max(0, min(1, x_final))
                y_final = max(0, min(1, y_final))
                
                rotated_corners.append(x_final)
                rotated_corners.append(y_final)
            
            # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
            yolo_bbox = [0] + rotated_corners
            
            # For overlap checking, we still use the non-rotated rectangle
            pixel_bbox = [x1, y1, x2, y2]
            
            return result_img, yolo_bbox, pixel_bbox
    
    # If after all attempts it wasn't possible to avoid overlaps
    # Return the last generated position
    result_img = bg_img.copy()
    
    if card_img.mode == 'RGBA':
        mask = card_img.split()[3]
    else:
        mask = None
        
    result_img.paste(card_img, (paste_x, paste_y), mask)
    
    # Ensure the bounding box is within the image
    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(bg_width, paste_x + rotated_width)
    y2 = min(bg_height, paste_y + rotated_height)
    
    # Calculate the center of the card
    center_x = paste_x + rotated_width / 2
    center_y = paste_y + rotated_height / 2
    
    # Calculate the four corners before rotation (relative to center)
    half_width = original_width / 2
    half_height = original_height / 2
    corners = [
        [-half_width, -half_height],  # Top-left
        [half_width, -half_height],   # Top-right
        [half_width, half_height],    # Bottom-right
        [-half_width, half_height]    # Bottom-left
    ]
    
    # Convert angle to radians (for rotation calculation)
    angle_rad = math.radians(-angle_deg)  # Negative because PIL rotates counterclockwise
    
    # Apply rotation and translate to center
    rotated_corners = []
    for x, y in corners:
        # Apply rotation
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # Translate to center and normalize for YOLO
        x_final = (center_x + x_rot) / bg_width
        y_final = (center_y + y_rot) / bg_height
        
        # Ensure values are in range [0, 1]
        x_final = max(0, min(1, x_final))
        y_final = max(0, min(1, y_final))
        
        rotated_corners.append(x_final)
        rotated_corners.append(y_final)
    
    # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
    yolo_bbox = [0] + rotated_corners
    
    pixel_bbox = [x1, y1, x2, y2]
    
    return result_img, yolo_bbox, pixel_bbox

def generate_dataset_image(cards_images, backgrounds_images, idx, save_dir, labels_dir, cards_per_image=None, flip_cards=False):
    """
    Generates a dataset image with multiple Pokemon cards on a background.
    Saves the image and corresponding label file.

    Args:
        cards_images: List of preloaded card Image objects
        backgrounds_images: List of preloaded background Image objects
        flip_cards: If True, rotates cards 90 degrees clockwise before placement
    """
    if cards_per_image is None:
        if random.random() < 0.5:
            grid_rows = random.randint(5, 15)
            grid_cols = random.randint(10, 20)
            cards_per_image = grid_rows * grid_cols
            use_grid = True
        else:
            # For random placement, sometimes generate very few cards
            cards_per_image = random.choices(
                [1, 2, 3, random.randint(8, 15)],
                weights=[0.1, 0.1, 0.1, 0.7]  # 30% chance for 1-3 cards, 70% for many cards
            )[0]
            use_grid = False
    else:
        use_grid = False
    
    # Select a random background and make a copy
    bg_img = random.choice(backgrounds_images).copy()

    # Resize background to a standard size for YOLO
    target_size = (640, 640)  # standard size for YOLO dataset
    bg_img = bg_img.resize(target_size, Image.LANCZOS)
    
    # List to keep track of existing bounding boxes
    existing_boxes = []
    
    # List to store YOLO coordinates
    yolo_bboxes = []
    
    if use_grid:
        # Generate a regular grid of cards
        cell_width = bg_img.width // grid_cols
        cell_height = bg_img.height // grid_rows
        
        # Calculate card size (slightly smaller than the cell)
        card_scale = 0.99  # Leave a small space between cards
        card_width = int(cell_width * card_scale)
        card_height = int(cell_height * card_scale)
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Select a random card and make a copy
                card_img = random.choice(cards_images).copy()

                # Flip card 90 degrees clockwise if requested
                if flip_cards:
                    card_img = card_img.rotate(-90, expand=True)

                # Resize the card to fit the cell
                original_width, original_height = card_img.size
                aspect_ratio = original_height / original_width
                new_card_width = card_width
                new_card_height = int(new_card_width * aspect_ratio)
                
                # If height is too large, resize based on height
                if new_card_height > card_height:
                    new_card_height = card_height
                    new_card_width = int(new_card_height / aspect_ratio)
                
                card_img = card_img.resize((new_card_width, new_card_height), Image.LANCZOS)
                
                # Calculate center position in the cell
                paste_x = col * cell_width + (cell_width - new_card_width) // 2
                paste_y = row * cell_height + (cell_height - new_card_height) // 2
                
                # Paste the card on the background
                if card_img.mode == 'RGBA':
                    mask = card_img.split()[3]
                else:
                    mask = None
                
                bg_img.paste(card_img, (paste_x, paste_y), mask)
                
                # Calculate bounding box in YOLO format (normalized)
                x1 = paste_x / bg_img.width
                y1 = paste_y / bg_img.height
                x2 = (paste_x + new_card_width) / bg_img.width
                y2 = y1
                x3 = x2
                y3 = (paste_y + new_card_height) / bg_img.height
                x4 = x1
                y4 = y3
                
                # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                yolo_bbox = [0, x1, y1, x2, y2, x3, y3, x4, y4]
                yolo_bboxes.append(yolo_bbox)
                
                # Add bounding box to existing list (for compatibility)
                pixel_bbox = [paste_x, paste_y, paste_x + new_card_width, paste_y + new_card_height]
                existing_boxes.append(pixel_bbox)
    else:
        # Use original random placement
        for _ in range(cards_per_image):
            # Select a random card and make a copy
            card_img = random.choice(cards_images).copy()

            # Flip card 90 degrees clockwise if requested
            if flip_cards:
                card_img = card_img.rotate(-90, expand=True)

            # Determine scale factor based on number of cards
            if cards_per_image == 1:
                scale_factor = random.uniform(0.7, 0.8)  # Very large cards (70-80% of image)
            elif cards_per_image == 2:
                scale_factor = random.uniform(0.35, 0.45)  # Large cards (35-45% of image)
            elif cards_per_image == 3:
                scale_factor = random.uniform(0.25, 0.35)  # Medium cards (25-35% of image)
            else:
                scale_factor = random.uniform(0.08, 0.15)  # Small cards (8-15% of image)
            
            # Place the card on the background with the specified scale factor
            result, yolo_bbox, bbox = place_card_on_background(
                card_img, bg_img, existing_boxes, scale_factor=scale_factor
            )
            bg_img = result
            
            # Add bounding box to existing list
            existing_boxes.append(bbox)
            
            # Store YOLO coordinates
            yolo_bboxes.append(yolo_bbox)
    
    # Save the image
    img_filename = f"image_{idx:06d}.jpg"
    img_path = os.path.join(save_dir, img_filename)
    bg_img = bg_img.convert("RGB")  # Convert to RGB to save as JPG
    bg_img.save(img_path, quality=95)
    
    # Save YOLO OBB label file
    label_filename = f"image_{idx:06d}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    
    with open(label_path, "w") as f:
        for bbox in yolo_bboxes:
            # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
            formatted_bbox = [
                int(bbox[0]),  # class (integer)
            ]
            # Add the 8 coordinates (4 points) rounded to 6 decimals
            for i in range(1, 9):
                formatted_bbox.append(round(float(bbox[i]), 6))
                
            line = " ".join(map(str, formatted_bbox))
            f.write(line + "\n")
            
    return img_path, label_path

def create_yaml_file():
    """Creates the YAML file for YOLOv8 training."""
    data = {
        'train': './train/images',  # Relative path for Ultralytics HUB
        'val': './val/images',
        'test': './test/images',
        'nc': 1,  # Number of classes
        'names': ['pokemon_card']  # Class name
    }
    
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("✓ YAML file created successfully")

def split_dataset(total_images=10000, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Calculates the dataset split into train, validation and test.
    Returns the number of images for each split.
    """
    train_images = int(total_images * train_ratio)
    val_images = int(total_images * val_ratio)
    test_images = total_images - train_images - val_images
    
    return {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

def create_zip_file():
    """Creates a ZIP file of the dataset."""
    zip_filename = 'pokemon_cards_dataset.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Create a structure as required by Ultralytics HUB
        for split in ['train', 'val', 'test']:
            # Add images
            imgs_dir = f'dataset/images/{split}'
            for img_file in os.listdir(imgs_dir):
                img_path = os.path.join(imgs_dir, img_file)
                # Path in ZIP file: train/images/file.jpg
                zipf.write(img_path, f'{split}/images/{img_file}')
            
            # Add labels
            labels_dir = f'dataset/labels/{split}'
            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)
                # Path in ZIP file: train/labels/file.txt
                zipf.write(label_path, f'{split}/labels/{label_file}')
        
        # Add YAML file
        zipf.write('dataset/data.yaml', 'data.yaml')
    
    print(f"✓ ZIP file '{zip_filename}' created successfully")
    return zip_filename

def verify_labels(labels_dir):
    """Verifies that label files are correctly formatted for OBB."""
    invalid_files = []
    
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:  # OBB format: class + 8 coordinates
                    invalid_files.append(label_file)
                    break
                try:
                    # Check that values are numbers and in the correct range
                    class_id = int(parts[0])
                    
                    # Verify that all coordinates are in range [0,1]
                    for i in range(1, 9):
                        coord = float(parts[i])
                        if coord < 0 or coord > 1:
                            invalid_files.append(label_file)
                            break
                            
                except ValueError:
                    invalid_files.append(label_file)
                    break
    
    if invalid_files:
        print(f"WARNING: Found {len(invalid_files)} invalid label files")
        return False
    else:
        print("✓ All label files are correctly formatted for OBB")
        return True

def main():
    """Main function to generate the dataset."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate YOLO dataset for Pokemon card detection')
    parser.add_argument('--flipped', action='store_true',
                        help='Generate 5000 additional images with cards flipped 90 degrees clockwise')
    parser.add_argument('--cards-dir', type=str, default='carte_pokemon',
                        help='Directory containing Pokemon card images (default: carte_pokemon)')
    parser.add_argument('--backgrounds-dir', type=str, default='background',
                        help='Directory containing background images (default: background)')
    parser.add_argument('--max-cards', type=int, default=None,
                        help='Maximum number of card images to load into memory (default: load all)')
    args = parser.parse_args()

    print("Generating YOLO dataset for Pokemon card detection")

    # Define paths from arguments
    cards_dir = args.cards_dir
    backgrounds_dir = args.backgrounds_dir

    # Verify folder existence
    if not os.path.exists(cards_dir):
        raise FileNotFoundError(f"The folder {cards_dir} does not exist")
    if not os.path.exists(backgrounds_dir):
        raise FileNotFoundError(f"The folder {backgrounds_dir} does not exist")
    
    # Create directory structure
    create_directory_structure()
    
    # Get files
    cards_files = get_card_files(cards_dir)
    backgrounds_files = get_background_files(backgrounds_dir)

    if not cards_files:
        raise ValueError(f"No images found in folder {cards_dir}")
    if not backgrounds_files:
        raise ValueError(f"No images found in folder {backgrounds_dir}")

    # Preload images into memory for faster generation
    # Randomly sample cards if max_cards is specified
    if args.max_cards and len(cards_files) > args.max_cards:
        print(f"Randomly sampling {args.max_cards} cards from {len(cards_files)} available cards")
        cards_files = random.sample(cards_files, args.max_cards)

    cards_images = preload_images(cards_files, "card images")
    backgrounds_images = preload_images(backgrounds_files, "background images")

    # Calculate dataset split
    total_images = 10000  # Total number of images required
    split_counts = split_dataset(total_images)
    
    print(f"Generating {total_images} images:")
    print(f" - Train: {split_counts['train']} images")
    print(f" - Validation: {split_counts['val']} images")
    print(f" - Test: {split_counts['test']} images")
    
    # Generate the dataset
    current_idx = 0
    
    # Generate train images
    print("\nGenerating train images...")
    for i in tqdm(range(split_counts['train'])):
        generate_dataset_image(
            cards_images,
            backgrounds_images,
            current_idx,
            'dataset/images/train',
            'dataset/labels/train'
        )
        current_idx += 1
    
    # Generate validation images
    print("\nGenerating validation images...")
    for i in tqdm(range(split_counts['val'])):
        generate_dataset_image(
            cards_images,
            backgrounds_images,
            current_idx,
            'dataset/images/val',
            'dataset/labels/val'
        )
        current_idx += 1
    
    # Generate test images
    print("\nGenerating test images...")
    for i in tqdm(range(split_counts['test'])):
        generate_dataset_image(
            cards_images,
            backgrounds_images,
            current_idx,
            'dataset/images/test',
            'dataset/labels/test'
        )
        current_idx += 1

    # Generate flipped images if requested
    if args.flipped:
        flipped_images = 5000
        flipped_split = split_dataset(flipped_images)

        print(f"\nGenerating {flipped_images} additional flipped images (90° clockwise):")
        print(f" - Train: {flipped_split['train']} images")
        print(f" - Validation: {flipped_split['val']} images")
        print(f" - Test: {flipped_split['test']} images")

        # Generate flipped train images
        print("\nGenerating flipped train images...")
        for i in tqdm(range(flipped_split['train'])):
            generate_dataset_image(
                cards_images,
                backgrounds_images,
                current_idx,
                'dataset/images/train',
                'dataset/labels/train',
                flip_cards=True
            )
            current_idx += 1

        # Generate flipped validation images
        print("\nGenerating flipped validation images...")
        for i in tqdm(range(flipped_split['val'])):
            generate_dataset_image(
                cards_images,
                backgrounds_images,
                current_idx,
                'dataset/images/val',
                'dataset/labels/val',
                flip_cards=True
            )
            current_idx += 1

        # Generate flipped test images
        print("\nGenerating flipped test images...")
        for i in tqdm(range(flipped_split['test'])):
            generate_dataset_image(
                cards_images,
                backgrounds_images,
                current_idx,
                'dataset/images/test',
                'dataset/labels/test',
                flip_cards=True
            )
            current_idx += 1

    # Verify label files
    print("\nVerifying label files...")
    verify_labels('dataset/labels/train')
    
    # Create YAML file
    create_yaml_file()
    
    # Create ZIP file
    zip_filename = create_zip_file()
    
    print("\nDataset generation completed!")
    print(f"The dataset has been saved in the 'dataset' folder and compressed in '{zip_filename}'")
    print("\nSummary:")
    print("- Recommended model: YOLOv8n (Nano) for optimized browser execution")
    print("- For training: yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640")
    print("- For conversion to web format: yolo export model=runs/train/best.pt format=tfjs")
    
    return True

if __name__ == "__main__":
    main()