import os
import shutil
import cv2
import numpy as np
import tkinter as tk
import yaml

print("Starting ARVP preprocessing...")

# ---------- Helper functions ----------
def parse_classes(filepath):
    classes = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.rstrip(',').split(':')
            if len(parts) >= 2:
                idx = int(parts[0])
                cls_name = parts[1].strip().strip('"')
                classes[idx] = cls_name
    return classes

def draw_bboxes_with_labels(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            x_center *= w; box_w *= w; y_center *= h; box_h *= h
            x1 = int(x_center - box_w/2)
            y1 = int(y_center + box_w/2)
            x2 = int(x_center + box_w/2)
            y2 = int(y_center + box_w/2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, id_to_class.get(cls, str(cls)), (x1, max(y1-10,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return image

# ---------- Paths ----------
print("Setting up ARVP directories...")
bottom_dir = r'C:\VSCode\datasets\ARVP\ARVP\bottom_fisheye_camera'
front_dir  = r'C:\VSCode\datasets\ARVP\ARVP\front_camera'
unified_dir = r'C:\VSCode\datasets\ARVP\ARVP\arvp'

# ---------- Parse classes from each camera ----------
print("Parsing ARVP class files...")
bottom_classes = parse_classes(os.path.join(bottom_dir, 'classes.txt'))
front_classes  = parse_classes(os.path.join(front_dir, 'classes.txt'))

# Filter out unwanted classes
drop_set = {"GateAbydos", "GateEarth", "TorpedoClosed", "BuoyAbydos1", "BuoyAbydos2",
            "BuoyEarth1", "BuoyEarth2", "BinAbydos1", "BinAbydos2", "BinEarth1", "BinEarth2"}
all_classes = (set(bottom_classes.values()).union(set(front_classes.values()))) - drop_set
new_mapping = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
print("Unified ARVP class mapping:", new_mapping)

# Add the following line to define id_to_class:
id_to_class = {idx: cls for cls, idx in new_mapping.items()}

bottom_old_to_new = {old: new_mapping[cls] for old, cls in bottom_classes.items() if cls in new_mapping}
front_old_to_new  = {old: new_mapping[cls] for old, cls in front_classes.items() if cls in new_mapping}

# ---------- Create ARVP unified dataset directories ----------
print("Creating ARVP unified dataset directories...")
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(unified_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(unified_dir, 'labels', split), exist_ok=True)

# ---------- Process ARVP datasets ----------
def process_dataset(source_dir, prefix, old_to_new_mapping, split_map):
    for sp in ['train', 'valid', 'test']:
        src_split = os.path.join(source_dir, sp)
        if not os.path.isdir(src_split):
            print(f"Source directory {src_split} not found.")
            continue
        print(f"Found {len(os.listdir(src_split))} files in {src_split}.")
        tgt_split = split_map.get(sp, sp)
        dst_images = os.path.join(unified_dir, 'images', tgt_split)
        dst_labels = os.path.join(unified_dir, 'labels', tgt_split)
        for file in os.listdir(src_split):
            file_path = os.path.join(src_split, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    new_file = prefix + "_" + file
                    shutil.copy(file_path, os.path.join(dst_images, new_file))
                    #print(f"Copied image {file} to {dst_images}")
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
                        #print(f"Copied label {file} to {dst_labels}")

print("Processing ARVP datasets...")
split_mapping = {'valid': 'val'}
process_dataset(bottom_dir, 'bottom', bottom_old_to_new, split_mapping)
process_dataset(front_dir, 'front', front_old_to_new, split_mapping)

# ---------- Write ARVP unified classes.txt and YAML ----------
with open(os.path.join(unified_dir, 'classes.txt'), 'w') as f:
    for cls, idx in new_mapping.items():
        f.write(f'{idx}:"{cls}"\n')
print("Unified ARVP classes.txt written.")

print("Generating ARVP YOLOv8 YAML configuration...")
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
print("ARVP YAML configuration saved to", yaml_path)

# ---------- Optional: Cleanup (if needed) ----------
print("Cleaning up images without labels in ARVP output...")
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(unified_dir, 'images', split)
    lbl_dir = os.path.join(unified_dir, 'labels', split)
    for img_file in os.listdir(img_dir):
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"
        if label_file not in os.listdir(lbl_dir):
            os.remove(os.path.join(img_dir, img_file))
print("ARVP preprocessing complete.")