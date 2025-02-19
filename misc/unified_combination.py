import os
import shutil
import yaml
import cv2
import numpy as np

print("Starting unified combination process...")

# ---------- Define source and destination paths ----------
# ARVP unified output from arvp_preprocess.py
ARVP_DIR = r'C:\VSCode\datasets\ARVP\ARVP\arvp'
# Buoy-rename output directory (assumed preprocessed separately)
BUOY_DIR = r'C:/VSCode/datasets/robosub_images_new'
# New combined unified dataset output path
UNIFIED_NEW = r'C:/VSCode/datasets/unified'

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(UNIFIED_NEW, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(UNIFIED_NEW, 'labels', split), exist_ok=True)

# ---------- Load class mappings ----------
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

print("Loading ARVP class mapping...")
arvp_classes = load_classes(os.path.join(ARVP_DIR, 'classes.txt'))
BUOY_YAML = os.path.join(BUOY_DIR, 'robosub_dataset.yaml')
print("Loading buoy-rename class mapping...")
with open(BUOY_YAML, 'r') as f:
    buoy_config = yaml.safe_load(f)
buoy_classes = {int(k):v for k,v in buoy_config['names'].items()}

# ---------- Collect all unique class names ----------
unified_class_names = set()
for cls_map in [arvp_classes, buoy_classes]:
    unified_class_names.update(cls_map.values())

# ---------- Build new unified mapping ----------
def unify_class_name(cls_name, source):
    cls_name_lower = cls_name.lower()
    if source == "arvp":
        if cls_name_lower == "buoy":
            return "torpedo-buoy"
        elif "gate" in cls_name_lower:
            return "FullGate" if cls_name == "FullGate" else "HalfGate"
        else:
            return cls_name
    elif source == "buoy":
        if "gate" in cls_name_lower:
            return "HalfGate"
        else:
            return cls_name
    return cls_name

unified_class_names = set(unify_class_name(cls, source)
                            for cls_map, source in [(arvp_classes, "arvp"), (buoy_classes, "buoy")]
                            for cls in cls_map.values())

unified_names = {idx: name for idx, name in enumerate(sorted(unified_class_names))}
name_to_new_id = {name: idx for idx, name in unified_names.items()}
print("New unified class mapping:", unified_names)

def convert_label(source, cls_name):
    new_cls = unify_class_name(cls_name, source)
    return name_to_new_id.get(new_cls)

# ---------- Combine sources ----------
print("Converting labels from ARVP and buoy-rename sources...")
sources = [
    (ARVP_DIR, arvp_classes, "arvp"),
    (BUOY_DIR, buoy_classes, "buoy")
]

for src_dir, cls_map, source_tag in sources:
    for split in splits:
        img_src_dir = os.path.join(src_dir, 'images', split)
        lbl_src_dir = os.path.join(src_dir, 'labels', split)
        if not os.path.isdir(lbl_src_dir):
            print(f"Labels directory {lbl_src_dir} not found for source {source_tag}.")
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
                    new_cls_id = convert_label(source_tag, orig_cls)
                    if new_cls_id is not None:
                        new_lines.append(f"{new_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
            if not new_lines:
                print(f"No valid label lines in {src_lbl_path}; skipping.")
                continue
            img_filename = lbl_file.replace('.txt', '.jpg')
            src_img_path = os.path.join(img_src_dir, img_filename)
            if not os.path.exists(src_img_path):
                print(f"Image {src_img_path} not found; skipping label {lbl_file}.")
                continue
            new_img_filename = f"{source_tag}_{img_filename}"
            new_lbl_filename = f"{source_tag}_{lbl_file}"
            dst_img_path = os.path.join(UNIFIED_NEW, 'images', split, new_img_filename)
            dst_lbl_path = os.path.join(UNIFIED_NEW, 'labels', split, new_lbl_filename)
            shutil.copy(src_img_path, dst_img_path)
            with open(dst_lbl_path, 'w') as f:
                f.writelines(new_lines)
            #print(f"Copied {new_img_filename} to {os.path.join(UNIFIED_NEW, 'images', split)}")
            #print(f"Wrote labels to {dst_lbl_path}")
print("Label conversion complete.")

# ---------- Write new unified classes.txt and YAML ----------
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
print("Unified dataset YAML created at", yaml_out_path)
print("Unified combination process complete.")