from ultralytics import YOLO
from misc.utils import print_system_info 

if __name__ == '__main__':
    print_system_info()

    # Load the model, ensure it has the specification for norm_type: "group", the default currently in yolov8.yaml
    # is currently norm_type: "batch" as this is the standard normalization used in YOLO
    # Please be sure to change it before running to view changes
    model = YOLO("yolov10n.yaml").load("yolov10n.pt")  # build from YAML and transfer weights

    # Print the normalization type, default to 'batch' if not specified
    #print(f'\nNormalization type: {model.model.yaml.get("norm_type", "batch")} normalization\n')


    # results = model.train(data="coco128.yaml", epochs=1000, imgsz=640, batch=4)

    # Train the model with specified parameters
    results = model.train(
        #data="coco128.yaml",       # Path to the dataset configuration file
        data="C:/VSCode/datasets/unified/unified_dataset.yaml",
        model="yolov10n.pt",                     # Model weights to start with (YOLOv8 Nano)
        epochs=100,                             # Total number of training epochs
        # imgsz=640,                              # Image size (pixels) used for training
        batch=32,                               # Batch size (number of samples per batch)
        # device=1,                               # For GPU Training
        # amp=True,                               # Automatic mixed precision to speed up training
        # mosaic=1.0,                             # Mosaic augmentation probability for images
        # auto_augment="randaugment",             # Automatically apply random augmentations
        # hsv_h=0.015,                            # Hue variation for color augmentation
        # hsv_s=0.7,                              # Saturation variation for color augmentation
        # hsv_v=0.4,                              # Value (brightness) variation for color augmentation
        # degrees=0.0,                            # Degrees of image rotation for augmentation
        # translate=0.1,                          # Max translation of the image in both directions
        # scale=0.5,                              # Scale factor for image resizing
        # shear=0.0,                              # Shear factor for image distortion
        # perspective=0.0,                        # Perspective distortion factor
        # flipud=0.0,                             # Probability of vertical flipping
        # fliplr=0.5,                             # Probability of horizontal flipping
        # cos_lr=True,                            # Use cosine learning rate decay
        # warmup_epochs=3.0,                      # Number of warmup epochs to stabilize training
        # warmup_momentum=0.8,                    # Initial momentum during warmup
        # warmup_bias_lr=0.1,                     # Initial learning rate for the bias parameters
        # lr0=0.01,                               # Initial learning rate
        # lrf=0.01,                               # Final learning rate factor for decay
        # momentum=0.937,                         # Momentum for optimizer
        # weight_decay=0.0005,                    # Regularization strength (weight decay)
        project="YoloV10RobosubV3",       # Folder where results are saved
        name="Attempt",                        # Experiment name for this training run
        plots=True,                             # Generate training plots after each epoch
        visualize=True,                         # Visualize model predictions during training
        # save_txt=True,                          # Save detection results in .txt format
        pretrained=True,                        # Use pretrained weights for better starting point
        # workers=8                               # Number of data loading workers
    )


    #val_results = model.val()
    #test_results = model.predict("path/to/test/data")
    #print("Validation results:", val_results)
    #print("Test results:", test_results)