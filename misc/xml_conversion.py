'''
XML data sample: 
<annotation><object><name>qual_gate</name><bndbox><xmin>226</xmin><ymin>200</ymin><xmax>243</xmax><ymax>338</ymax></bndbox></object><object><name>qual_gate</name><bndbox><xmin>480</xmin><ymin>187</ymin><xmax>507</xmax><ymax>332</ymax></bndbox></object><object><name>buoy_yellow</name><bndbox><xmin>274</xmin><ymin>329</xmin><xmax>284</xmax><ymax>343</ymax></bndbox></object><object><name>buoy_green</name><bndbox><xmin>331</xmin><ymin>336</xmin><xmax>341</xmin><ymax>351</ymax></bndbox></object><object><name>buoy_red</name><bndbox><xmin>384</xmin><ymin>321</xmin><xmax>395</xmin><ymax>338</ymax></bndbox></object></annotation>

Directory struct:
logan@DESKTOP-JA3B0L7:/mnt/c/VSCode/datasets/robosub_transdec_dataset-master/robosub_transdec_dataset-master$ ls -R
.:
Annotations  ImageSets  Images  LICENSE  README.md  VGG16_faster_rcnn_final.caffemodel  results

./Annotations:
1.xml     1127.xml  1256.xml  1385.xml  1513.xml  1642.xml  229.xml  358.xml  487.xml  615.xml  744.xml  873.xml
10.xml    1128.xml  1257.xml  1386.xml  1514.xml  1643.xml  23.xml   359.xml  488.xml  616.xml  745.xml  874.xml

./Images:
1.jpg     1127.jpg  1256.jpg  1385.jpg  1513.jpg  1642.jpg  229.jpg  358.jpg  487.jpg  615.jpg  744.jpg  873.jpg

'''

import os
import xml.etree.ElementTree as ET
import random
import shutil
from collections import defaultdict

# Paths (modify these paths according to your directory structure)
DATASET_DIR = r'C:\VSCode\datasets\robosub_transdec_dataset-master\robosub_transdec_dataset-master'
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'Annotations')
IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')
OUTPUT_IMAGES_DIR = os.path.join('C:/VSCode/datasets/robosub_images/', 'images')  # New path for images
OUTPUT_LABELS_DIR = os.path.join('C:/VSCode/datasets/robosub_images/', 'labels')  # New path for labels
# CLASSES_FILE = os.path.join('C:/VSCode/datasets/', 'classes.txt')  # Removed this line

# Define dataset splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Set random seed for reproducibility
random.seed(42)

# Gather all XML files
xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]

# Shuffle XML files
random.shuffle(xml_files)

# Calculate split indices
total = len(xml_files)
train_end = int(total * TRAIN_SPLIT)
val_end = train_end + int(total * VAL_SPLIT)

train_files = xml_files[:train_end]
val_files = xml_files[train_end:val_end]
test_files = xml_files[val_end:]

# Create directories for splits
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_LABELS_DIR, split), exist_ok=True)

# Define labels to skip
skip_labels = {"background", "bin", "bin_cover", "object_dropoff", "object_pickup"}

# Load class names from XML files and create a mapping to class IDs
class_names = set()
for xml_file in xml_files:
    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name.lower() in skip_labels:
            continue
        class_names.add(class_name)
class_to_id = {name: idx for idx, name in enumerate(sorted(class_names))}

print("Class to ID mapping:", class_to_id)

# Function to convert Pascal VOC bbox to YOLO format
def convert_bbox(size, bbox):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (bbox['xmin'] + bbox['xmax']) / 2.0
    y_center = (bbox['ymin'] + bbox['ymax']) / 2.0
    width = bbox['xmax'] - bbox['xmin']
    height = bbox['ymax'] - bbox['ymin']
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return x_center, y_center, width, height

# Function to get split based on filename
def get_split(xml_file):
    if xml_file in train_files:
        return 'train'
    elif xml_file in val_files:
        return 'val'
    else:
        return 'test'

label_counts = defaultdict(int)

# Process each XML file
for xml_file in xml_files:
    split = get_split(xml_file)
    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image filename and size
    filename = root.find('filename').text if root.find('filename') is not None else xml_file.replace('.xml', '.jpg')
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        print(f"Image file {filename} not found for annotation {xml_file}. Skipping.")
        continue

    # Copy image to corresponding split directory
    shutil.copy(image_path, os.path.join(OUTPUT_IMAGES_DIR, split, filename))

    # Get image size
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size

    valid_objects = []

    # Open the corresponding label file
    label_filename = xml_file.replace('.xml', '.txt')
    label_file = os.path.join(OUTPUT_LABELS_DIR, split, label_filename)
    with open(label_file, 'w') as lf:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name.lower() in skip_labels:
                continue
            if class_name not in class_to_id:
                print(f"Class {class_name} not found. Skipping object.")
                continue
            bndbox = obj.find('bndbox')
            bbox = {
                'xmin': float(bndbox.find('xmin').text),
                'ymin': float(bndbox.find('ymin').text),
                'xmax': float(bndbox.find('xmax').text),
                'ymax': float(bndbox.find('ymax').text)
            }
            valid_objects.append((class_name, bbox))

    if not valid_objects:
        # Skip this image if only skipped labels or no labels were present
        #print(f"Skipping {xml_file} because it contains only skipped labels.")
        continue

    with open(label_file, 'w') as lf:
        for class_name, bbox in valid_objects:
            class_id = class_to_id[class_name]
            label_counts[class_name] += 1
            x_center, y_center, bbox_width, bbox_height = convert_bbox((width, height), bbox)
            lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    # Generate YAML configuration
    yaml_content = f"""
# Dataset root directory
path: C:/VSCode/datasets/robosub_images

# Train/val/test sets
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
    for name, idx in sorted(class_to_id.items(), key=lambda item: item[1]):
        yaml_content += f"  {idx}: {name}\n"

    # Write YAML file
    yaml_path = os.path.join('C:/VSCode/datasets/robosub_images/', 'robosub_dataset.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    #print(f"YAML configuration saved to {yaml_path}")

# After processing all files, print label counts
print("\nLabel counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
