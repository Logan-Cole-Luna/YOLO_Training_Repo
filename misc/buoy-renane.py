import os
import yaml

# NEW: Define original and new dataset roots and related paths.
ORIGINAL_DATASET_ROOT = r'C:/VSCode/datasets/robosub_images'
NEW_DATASET_ROOT = r'C:/VSCode/datasets/robosub_images_new'
# Read config from original YAML
ORIGINAL_YAML_PATH = os.path.join(ORIGINAL_DATASET_ROOT, 'robosub_dataset.yaml')
NEW_YAML_PATH = os.path.join(NEW_DATASET_ROOT, 'robosub_dataset.yaml')
# Original and new labels directories
ORIGINAL_LABELS_DIR = os.path.join(ORIGINAL_DATASET_ROOT, 'labels')
NEW_LABELS_DIR = os.path.join(NEW_DATASET_ROOT, 'labels')
SPLITS = ['train', 'val', 'test']

# Create new dataset directory and split label subdirectories
os.makedirs(NEW_DATASET_ROOT, exist_ok=True)
for split in SPLITS:
    os.makedirs(os.path.join(NEW_LABELS_DIR, split), exist_ok=True)

# NEW: Define and create images directories for the new dataset.
ORIGINAL_IMAGES_DIR = os.path.join(ORIGINAL_DATASET_ROOT, 'images')
NEW_IMAGES_DIR = os.path.join(NEW_DATASET_ROOT, 'images')
for split in SPLITS:
    os.makedirs(os.path.join(NEW_IMAGES_DIR, split), exist_ok=True)

# Load original YAML configuration from the original dataset
with open(ORIGINAL_YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)
# Clone old class mapping (id -> name)
old_names = config['names'].copy()  # e.g., {0: "buoy_red", 1: "buoy_green", ...}

# Build new mapping: keep only buoy and gate, renaming qual_gate to gate
unique_names = set()
for cls in old_names.values():
    lower = cls.lower()
    if "buoy" in lower:
        unique_names.add("buoy")
    elif lower == "qual_gate":
        unique_names.add("gate")
    # Drop channel and torpedo classes
# Generate new mapping with sorted order
new_mapping = {name: idx for idx, name in enumerate(sorted(unique_names))}

# Print new classes mapping
print("New class mapping:", new_mapping)

# Update YAML configuration with new mapping (id -> name)
new_names = {idx: name for name, idx in new_mapping.items()}
config['names'] = new_names

# NEW: Update YAML configuration "path" to new dataset and names mapping
config['path'] = "C:/VSCode/datasets/robosub_images_new"  # Updated dataset root
config['names'] = new_names
with open(NEW_YAML_PATH, 'w') as f:
    yaml.dump(config, f, sort_keys=False)

# After YAML update and creation of new_names, add:
label_names = new_names

# Build reverse mapping from old id to new id using allowed names only
old_to_new = {}
for old_id, old_cls in old_names.items():
    lower = old_cls.lower()
    if "buoy" in lower:
        unified = "buoy"
    elif lower == "qual_gate":
        unified = "gate"
    else:
        continue  # skip others
    old_to_new[int(old_id)] = new_mapping[unified]

# Build reverse mapping (only for allowed labels)
old_to_new = {}
for old_id, old_cls in old_names.items():
    lower = old_cls.lower()
    if "buoy" in lower:
        unified = "buoy"
    elif lower == "qual_gate":
        unified = "gate"
    else:
        continue  # skip dropped classes
    old_to_new[int(old_id)] = new_mapping[unified]

# Update label files from the original dataset and write to the new dataset
for split in SPLITS:
    orig_split_dir = os.path.join(ORIGINAL_LABELS_DIR, split)
    new_split_dir = os.path.join(NEW_LABELS_DIR, split)
    if not os.path.isdir(orig_split_dir):
        continue
    for file in os.listdir(orig_split_dir):
        if not file.endswith('.txt'):
            continue
        orig_file_path = os.path.join(orig_split_dir, file)
        new_file_path = os.path.join(new_split_dir, file)
        with open(orig_file_path, 'r') as lf:
            lines = lf.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            old_id = int(parts[0])
            # Skip labels that were dropped
            if old_id not in old_to_new:
                continue
            new_id = old_to_new[old_id]
            new_lines.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if new_lines:
            with open(new_file_path, 'w') as lf:
                lf.writelines(new_lines)
        else:
            # If no valid labels remain, remove the label file and associated image.
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            image_file = file.replace('.txt', '.jpg')
            new_image_path = os.path.join(NEW_IMAGES_DIR, split, image_file)
            if os.path.exists(new_image_path):
                os.remove(new_image_path)

# NEW: Copy images from original dataset to new dataset based on split.
for split in SPLITS:
    orig_split_img_dir = os.path.join(ORIGINAL_IMAGES_DIR, split)
    new_split_img_dir = os.path.join(NEW_IMAGES_DIR, split)
    if not os.path.isdir(orig_split_img_dir):
        continue
    for img_file in os.listdir(orig_split_img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(orig_split_img_dir, img_file)
            dst = os.path.join(new_split_img_dir, img_file)
            if not os.path.exists(dst):
                # Copy the image file to new dataset images split folder.
                import shutil
                shutil.copy(src, dst)

import cv2
import numpy as np

# Updated function: display class name instead of number
def draw_bboxes(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            x_center *= w; box_w *= w
            y_center *= h; box_h *= h
            x1 = int(x_center - box_w/2)
            y1 = int(y_center - box_h/2)
            x2 = int(x_center + box_w/2)
            y2 = int(y_center + box_h/2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label_names.get(cls, str(cls)), (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return image

# NEW: Collect one sample image per class
sample_images = {}
for split in ['train', 'val', 'test']:
    labels_dir = os.path.join(NEW_LABELS_DIR, split)
    images_dir = os.path.join(NEW_DATASET_ROOT, 'images', split)
    if not os.path.isdir(labels_dir):
        continue
    for lab_file in os.listdir(labels_dir):
        if not lab_file.endswith('.txt'):
            continue
        lab_path = os.path.join(labels_dir, lab_file)
        with open(lab_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls not in sample_images:
                img_file = lab_file.replace('.txt', '.jpg')
                img_path = os.path.join(images_dir, img_file)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_drawn = draw_bboxes(img.copy(), lab_path)
                        sample_images[cls] = img_drawn
        if len(sample_images) == len(new_mapping):
            break
    if len(sample_images) == len(new_mapping):
        break

# NEW: Create and display mosaic of sample images (thumbnails)
if sample_images:
    thumbnails = []
    for cls, img in sorted(sample_images.items()):
        thumb = cv2.resize(img, (200,200))
        thumbnails.append(thumb)
    cols = 3
    rows = (len(thumbnails) + cols - 1) // cols
    while len(thumbnails) < rows * cols:
        thumbnails.append(np.zeros((200,200,3), dtype=np.uint8))
    mosaic_rows = []
    for i in range(rows):
        mosaic_rows.append(np.hstack(thumbnails[i*cols:(i+1)*cols]))
    mosaic = np.vstack(mosaic_rows)
    cv2.imshow("Mosaic", mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No sample images found for mosaic.")
