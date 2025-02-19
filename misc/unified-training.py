import os
import shutil
import cv2
import numpy as np
import tkinter as tk
import yaml

print("Starting unified-training process...")

# ---------- Helper functions ----------
def parse_classes(filepath):
    classes = {}
    with open(filepath, 'r') as f:
        for line in f:
            # Remove comments and commas
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.rstrip(',').split(':')
            if len(parts) >= 2:
                idx = int(parts[0])
                cls_name = parts[1].strip().strip('"')
                classes[idx] = cls_name
    return classes

# New helper to draw bounding boxes with labels from a YOLO-format label file.
def draw_bboxes_with_labels(image, label_path):
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
            cv2.putText(image, id_to_class.get(cls, str(cls)), (x1, max(y1-10,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return image

# ---------- Paths ----------
print("Setting up dataset directories...")
bottom_dir = r'C:\VSCode\datasets\ARVP\ARVP\bottom_fisheye_camera'
front_dir  = r'C:\VSCode\datasets\ARVP\ARVP\front_camera'
unified_dir = r'C:\VSCode\datasets\ARVP\ARVP\arvp'

# ---------- Parse classes from each camera ----------
print("Parsing class files...")
bottom_classes = parse_classes(os.path.join(bottom_dir, 'classes.txt'))
front_classes  = parse_classes(os.path.join(front_dir, 'classes.txt'))

# Filter out unwanted classes
drop_set = {"GateAbydos", "GateEarth", "TorpedoClosed", "BuoyAbydos1", "BuoyAbydos2",
            "BuoyEarth1", "BuoyEarth2", "BinAbydos1", "BinAbydos2", "BinEarth1", "BinEarth2"}

all_classes = (set(bottom_classes.values()).union(set(front_classes.values()))) - drop_set
new_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
print("Unified class mapping:", new_mapping)

# Build reverse mappings (only for classes kept)
bottom_old_to_new = {old: new_mapping[cls] for old, cls in bottom_classes.items() if cls in new_mapping}
front_old_to_new  = {old: new_mapping[cls] for old, cls in front_classes.items() if cls in new_mapping}

# ---------- Create unified dataset directories ----------
print("Creating unified dataset directories...")
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(unified_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'labels', split), exist_ok=True)

# ---------- Function to process one dataset ----------
def process_dataset(source_dir, prefix, old_to_new_mapping, split_map):
    for sp in ['train', 'valid', 'test']:
        src_split = os.path.join(source_dir, sp)
        tgt_split = split_map.get(sp, sp)
        dst_images = os.path.join(unified_dir, 'images', tgt_split)
        dst_labels = os.path.join(unified_dir, 'labels', tgt_split)
        if not os.path.isdir(src_split):
            continue
        for file in os.listdir(src_split):
            file_path = os.path.join(src_split, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    new_file = prefix + "_" + file
                    shutil.copy(file_path, os.path.join(dst_images, new_file))
                elif file.lower().endswith('.txt'):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            old_id = int(parts[0])
                            if old_id in old_to_new_mapping:
                                new_id = old_to_new_mapping[old_id]
                                new_lines.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                    if new_lines:
                        new_label_file = prefix + "_" + file
                        with open(os.path.join(dst_labels, new_label_file), 'w') as f:
                            f.writelines(new_lines)

print("Processing ARVP datasets...")
split_mapping = {'valid': 'val'}  
process_dataset(bottom_dir, 'bottom', bottom_old_to_new, split_mapping)
process_dataset(front_dir, 'front', front_old_to_new, split_mapping)

# ---------- Write unified classes.txt ----------
with open(os.path.join(unified_dir, 'classes.txt'), 'w') as f:
    for cls, idx in new_mapping.items():
        f.write(f'{idx}:"{cls}"\n')
print("Unified classes.txt written.")

# ---------- Generate YOLOv8 YAML configuration ----------
print("Generating YOLOv8 YAML configuration...")
yaml_content = f"""
path: {unified_dir.replace('\\', '/')}
train: images/train
val: images/val
test: images/test
nc: {len(new_mapping)}
names: [{', '.join(f'"{cls}"' for cls in sorted(new_mapping, key=new_mapping.get))}]
"""
yaml_path = os.path.join(unified_dir, 'arvp_dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print("YAML configuration saved to", yaml_path)

# ---------- Build reverse mapping from class ID to class name ----------
id_to_class = {idx: cls for cls, idx in new_mapping.items()}

# ---------- Draw sample images and create mosaic thumbnails ----------
print("Creating sample images for mosaic preview...")
sample_images = {}
'''
splits = ['train', 'val', 'test']
for split in splits:
    labels_dir = os.path.join(unified_dir, 'labels', split)
    images_dir = os.path.join(unified_dir, 'images', split)
    if not os.path.isdir(labels_dir):
        continue
    for lbl_file in os.listdir(labels_dir):
        if not lbl_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, lbl_file)
        with open(label_path, 'r') as lf:
            first_line = lf.readline().strip().split()
        if first_line and len(first_line) == 5:
            cls_id = int(first_line[0])
            if cls_id not in sample_images:
                img_filename = lbl_file.replace('.txt', '.jpg')
                img_path = os.path.join(images_dir, img_filename)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_scaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                        img_with_boxes = draw_bboxes_with_labels(img_scaled.copy(), label_path)
                        sample_images[cls_id] = img_with_boxes
        if len(sample_images) == len(new_mapping):
            break
    if len(sample_images) == len(new_mapping):
        break

if sample_images:
    thumbs = []
    thumb_w, thumb_h = 300, 300
    title_h = 40
    for cid in sorted(sample_images.keys()):
        class_name = id_to_class.get(cid, str(cid))
        thumb = cv2.resize(sample_images[cid], (thumb_w, thumb_h))
        canvas = np.ones((thumb_h + title_h, thumb_w, 3), dtype=np.uint8) * 255
        text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = (thumb_w - text_size[0]) // 2
        text_y = (title_h + text_size[1]) // 2
        cv2.putText(canvas, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        canvas[title_h:thumb_h+title_h, :] = thumb
        thumbs.append(canvas)
    
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    while len(thumbs) < rows * cols:
        thumbs.append(np.ones((thumb_h + title_h, thumb_w, 3), dtype=np.uint8) * 255)
    mosaic_rows = [np.hstack(thumbs[i*cols:(i+1)*cols]) for i in range(rows)]
    mosaic = np.vstack(mosaic_rows)
    
    mosaic_down = cv2.resize(mosaic, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Mosaic Preview", mosaic_down)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Mosaic preview displayed.")
else:
    print("No sample images found for mosaic.")
'''

# ---------- Cleanup: Remove images with no corresponding label file ----------
print("Cleaning up images without labels...")
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(unified_dir, 'images', split)
    lbl_dir = os.path.join(unified_dir, 'labels', split)
    for img_file in os.listdir(img_dir):
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"
        if label_file not in os.listdir(lbl_dir):
            os.remove(os.path.join(img_dir, img_file))
print("Cleanup completed.")

# ---------- Combine ARVP and buoy-rename outputs into a unified dataset ----------
print("Starting combination of ARVP and buoy-rename outputs...")
# Source directories
ARVP_DIR = unified_dir  # ARVP unified output
BUOY_DIR = r'C:/VSCode/datasets/robosub_images_new'  # buoy-rename output

# New unified dataset output path
UNIFIED_NEW = r'C:/VSCode/datasets/unified'
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(UNIFIED_NEW, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(UNIFIED_NEW, 'labels', split), exist_ok=True)

def load_classes(path):
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':')
                idx = int(parts[0])
                cls = parts[1].replace('"','').strip()
                mapping[idx] = cls
    return mapping

print("Loading ARVP and buoy-rename class mappings...")
arvp_classes = load_classes(os.path.join(ARVP_DIR, 'classes.txt'))
BUOY_YAML = os.path.join(BUOY_DIR, 'robosub_dataset.yaml')
with open(BUOY_YAML, 'r') as f:
    buoy_config = yaml.safe_load(f)
buoy_classes = {int(k): v for k, v in buoy_config['names'].items()}

def convert_label(source, cls_name):
    if source == "arvp":
        if cls_name.lower() == "buoy":
            return "torpedo-buoy"
        elif "gate" in cls_name.lower():
            return "FullGate" if cls_name == "FullGate" else "HalfGate"
        else:
            return cls_name
    elif source == "buoy":
        if "gate" in cls_name.lower():
            return "HalfGate"
        else:
            return cls_name
    return cls_name

unified_class_set = set()
sources = [
    (ARVP_DIR, arvp_classes, "arvp"),
    (BUOY_DIR, buoy_classes, "buoy")
]

print("Converting labels from sources...")
for src_dir, cls_map, source_tag in sources:
    for split in ['train', 'val', 'test']:
        img_src_dir = os.path.join(src_dir, 'images', split)
        lbl_src_dir = os.path.join(src_dir, 'labels', split)
        if not os.path.isdir(lbl_src_dir): 
            continue
        for lbl_file in os.listdir(lbl_src_dir):
            if not lbl_file.endswith('.txt'):
                continue
            src_lbl_path = os.path.join(lbl_src_dir, lbl_file)
            new_lines = []
            with open(src_lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    old_id = int(parts[0])
                    orig_cls = cls_map.get(old_id, None)
                    if orig_cls is None:
                        continue
                    new_cls = convert_label(source_tag, orig_cls)
                    unified_class_set.add(new_cls)
                    new_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
            if not new_lines:
                continue
            img_filename = lbl_file.replace('.txt', '.jpg')
            src_img_path = os.path.join(img_src_dir, img_filename)
            if not os.path.exists(src_img_path):
                continue
            new_img_filename = f"{source_tag}_{img_filename}"
            new_lbl_filename = f"{source_tag}_{lbl_file}"
            dst_img_path = os.path.join(UNIFIED_NEW, 'images', split, new_img_filename)
            dst_lbl_path = os.path.join(UNIFIED_NEW, 'labels', split, new_lbl_filename)
            shutil.copy(src_img_path, dst_img_path)
            with open(dst_lbl_path, 'w') as f:
                f.writelines(new_lines)
print("Label conversion complete.")

# Create unified mapping: assign new IDs in sorted order.
unified_names = {idx: name for idx, name in enumerate(sorted(unified_class_set))}
name_to_new_id = {name: idx for name, idx in unified_names.items()}

print("Updating label files in unified dataset...")
for split in ['train', 'val', 'test']:
    lbl_dir = os.path.join(UNIFIED_NEW, 'labels', split)
    if not os.path.isdir(lbl_dir):
        continue
    for lbl_file in os.listdir(lbl_dir):
        lbl_path = os.path.join(lbl_dir, lbl_file)
        new_lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_name = parts[0]
                new_id = name_to_new_id.get(cls_name, None)
                if new_id is None:
                    continue
                new_lines.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        if new_lines:
            with open(lbl_path, 'w') as f:
                f.writelines(new_lines)
        else:
            os.remove(lbl_path)
            img_file = lbl_file.replace('.txt', '.jpg')
            img_path = os.path.join(UNIFIED_NEW, 'images', split, img_file)
            if os.path.exists(img_path):
                os.remove(img_path)

with open(os.path.join(UNIFIED_NEW, 'classes.txt'), 'w') as f:
    for idx, name in unified_names.items():
        f.write(f'{idx}:"{name}"\n')

yaml_content = f"""
path: {UNIFIED_NEW.replace('\\', '/')}
train: images/train
val: images/val
test: images/test
nc: {len(unified_names)}
names: [{', '.join(f'"{unified_names[idx]}"' for idx in sorted(unified_names))}]
"""
yaml_out_path = os.path.join(UNIFIED_NEW, 'unified_dataset.yaml')
with open(yaml_out_path, 'w') as f:
    f.write(yaml_content)
print("Unified dataset created at", UNIFIED_NEW)
print("Process complete.")
