import os
import random

from ultralytics import YOLO


for fold in range(1,5):
    model = YOLO(f'yolo11s.pt')
    results = model.train(
        data=f"./datasets/all_yolo_data/yolo_5fold/fold_{fold}/baseline.yaml",
        name=f"fold{fold}",
        project	=f"/root/aicup2025_II/upload/model/detect/all/yolo11s_fold{fold}",
        epochs=100,
        batch=32,
        imgsz=768,
        device=0,
        single_cls=True,
        deterministic=False,
        cos_lr=True,
        compile=True,
        cache=True,
        seed=random.randint(0,2**32),
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        degrees=0,
        translate=0.1,
        scale=0.5,
        shear=0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1,
        mixup=0.0,
        cutmix=0.0,
        dropout=0.1,
    )
