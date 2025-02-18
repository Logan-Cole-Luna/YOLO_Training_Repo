from ultralytics import YOLO
import cv2, os, numpy as np

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
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Use model.names for label lookup
            cv2.putText(image, model.names.get(cls, str(cls)), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

if __name__ == '__main__':
    # Load the trained model. Use raw string or forward slashes for Windows paths.
    model = YOLO(r"C:\VSCode\YOLO_Training_Repo\YoloV10RobosubV2\Attempt3\weights\last.pt")
    
    # Run validation on the test split and print the metrics as a dict.
    results = model.val(data="C:/VSCode/datasets/robosub_images_new/robosub_dataset.yaml", split='test')
    metrics = results.results_dict  # convert results to a dictionary
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Source directory for test images (as defined in your YAML: "images/test")
    test_img_dir = r"C:\VSCode\datasets\robosub_images_new\images\test"
    
    # Collect one prediction from each class.
    sample_preds = {}
    for img_file in os.listdir(test_img_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(test_img_dir, img_file)
        preds = model.predict(source=img_path, conf=0.25, show=False, save=False)
        # Each prediction (list) has a .boxes attribute with .cls values.
        if preds and preds[0].boxes is not None:
            boxes = preds[0].boxes
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for c in cls_ids:
                if c not in sample_preds:
                    sample_preds[c] = preds[0].plot()  # returns an image with drawn boxes
        if len(sample_preds) == len(model.names):
            break

    # Collect ground truth images from test labels folder.
    sample_gts = {}
    gt_labels_dir = r"C:\VSCode\datasets\robosub_images_new\labels\test"
    for lab_file in os.listdir(gt_labels_dir):
        if not lab_file.endswith('.txt'):
            continue
        lab_path = os.path.join(gt_labels_dir, lab_file)
        img_file = lab_file.replace('.txt', '.jpg')
        img_path = os.path.join(test_img_dir, img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            # Draw ground truth boxes from label file (using same helper)
            gt_img = draw_bboxes(img.copy(), lab_path)
            # Use the first label's class as the key (if multiple, this is just one sample)
            with open(lab_path, 'r') as f:
                line = f.readline()
            if line:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if cls not in sample_gts:
                        sample_gts[cls] = gt_img
        if len(sample_gts) == len(model.names):
            break

    # Create side-by-side mosaics: for each class, stack prediction and ground truth together
    combined_samples = []
    all_classes = sorted(set(list(sample_preds.keys()) | set(sample_gts.keys())))
    for c in all_classes:
        pred_img = sample_preds.get(c, np.zeros((200,200,3), dtype=np.uint8))
        gt_img = sample_gts.get(c, np.zeros((200,200,3), dtype=np.uint8))
        # Resize each to thumbnail size
        pred_thumb = cv2.resize(pred_img, (200,200))
        gt_thumb = cv2.resize(gt_img, (200,200))
        # Add class name on each thumbnail
        cv2.putText(pred_thumb, "Pred: "+model.names.get(c, str(c)), (5,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(gt_thumb, "GT: "+model.names.get(c, str(c)), (5,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Stack prediction (left) and ground truth (right) horizontally
        combined = np.hstack([pred_thumb, gt_thumb])
        combined_samples.append(combined)
    
    # Arrange combined samples into a mosaic (assume 2 per row)
    cols = 2
    rows = (len(combined_samples) + cols - 1) // cols
    # Fill missing images with blank images (each blank image has width 400, height 200)
    while len(combined_samples) < rows*cols:
        combined_samples.append(np.zeros((200,400,3), dtype=np.uint8))
    mosaic_rows = []
    for i in range(rows):
        mosaic_rows.append(np.hstack(combined_samples[i*cols:(i+1)*cols]))
    mosaic = np.vstack(mosaic_rows)
    
    # Scale up the mosaic by a factor (e.g., 2x)
    scale_factor = 2
    height, width = mosaic.shape[:2]
    mosaic_scaled = cv2.resize(mosaic, (width*scale_factor, height*scale_factor),
                               interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow("Predictions vs Ground Truth", mosaic_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# ...existing code...