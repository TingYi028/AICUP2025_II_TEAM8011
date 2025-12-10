import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import cv2  # 用於圖片處理


def preprocess_image(img_array, clip_min=123, clip_max=230):
    """
    對圖片進行截斷和正規化預處理

    參數:
        img_array: 輸入圖片的 numpy 陣列
        clip_min: 截斷下限 (預設 123)
        clip_max: 截斷上限 (預設 230)

    返回:
        處理後的圖片 (0-255 範圍)
    """
    # 將圖片轉換為浮點數進行運算
    img_float = img_array.astype(np.float32)

    # 步驟 1: 截斷到 [clip_min, clip_max]
    img_clipped = np.clip(img_float, clip_min, clip_max)

    # 步驟 2: 重新正規化到 [0, 255]
    img_normalized = (img_clipped - clip_min) / (clip_max - clip_min) * 255.0

    # 步驟 3: 轉換回 uint8 格式
    img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)

    return img_normalized


def create_multichannel_image(image_paths, current_idx, n_channels, apply_preprocessing=True):
    """
    建立多通道時序圖像

    參數:
        image_paths: 同一病人的所有圖片路徑列表（已排序）
        current_idx: 當前圖片在列表中的索引
        n_channels: 總通道數（必須是奇數）
        apply_preprocessing: 是否套用預處理

    返回:
        多通道圖像的 numpy 陣列 (CHW 格式)
    """
    half_window = n_channels // 2
    total_images = len(image_paths)
    channels = []

    # 建立通道列表：前半部、當前、後半部
    for offset in range(-half_window, half_window + 1):
        # 計算目標索引，並處理邊界情況
        target_idx = current_idx + offset

        # 邊界處理：填充最近的圖片
        if target_idx < 0:
            target_idx = 0
        elif target_idx >= total_images:
            target_idx = total_images - 1

        # 讀取圖片
        img = cv2.imread(str(image_paths[target_idx]), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"無法讀取圖片: {image_paths[target_idx]}")

        # 轉換為灰階（如果是彩色圖片）
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 預處理
        if apply_preprocessing:
            img = preprocess_image(img)

        channels.append(img)

    # 堆疊成 CHW 格式
    multichannel_img = np.stack(channels, axis=0)  # Shape: (n_channels, height, width)

    return multichannel_img


def copy_and_preprocess_image(src_img_path, dst_img_path, apply_preprocessing=True,
                              multi_channel=None, patient_images=None, current_idx=None):
    """
    讀取、預處理並儲存圖片（支援多通道）

    參數:
        src_img_path: 來源圖片路徑
        dst_img_path: 目標圖片路徑
        apply_preprocessing: 是否套用預處理 (預設 True)
        multi_channel: 多通道數量（必須是奇數或 None）
        patient_images: 同一病人的所有圖片路徑列表（用於多通道）
        current_idx: 當前圖片在 patient_images 中的索引
    """
    # 多通道模式
    if multi_channel is not None:
        if multi_channel % 2 == 0:
            raise ValueError(f"multi_channel 必須是奇數，當前值為: {multi_channel}")

        if patient_images is None or current_idx is None:
            raise ValueError("多通道模式需要提供 patient_images 和 current_idx")

        # 建立多通道圖像 (CHW 格式)
        multichannel_img = create_multichannel_image(
            patient_images, current_idx, multi_channel, apply_preprocessing
        )

        # 將 CHW 轉換為多頁 TIFF 格式（每個通道一頁）
        channel_list = [multichannel_img[i] for i in range(multichannel_img.shape[0])]

        # 修改副檔名為 .tiff
        dst_img_path = dst_img_path.with_suffix('.tiff')

        # 使用 imwritemulti 儲存多頁 TIFF
        cv2.imwritemulti(str(dst_img_path), channel_list)

        return True

    # 單通道模式（原始邏輯）
    else:
        img = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"警告: 無法讀取圖片 {src_img_path}")
            return False

        # 如果需要預處理
        if apply_preprocessing:
            # 處理彩色圖片 (逐通道處理)
            if len(img.shape) == 3:
                img_processed = np.zeros_like(img)
                for i in range(img.shape[2]):
                    img_processed[:, :, i] = preprocess_image(img[:, :, i])
                img = img_processed
            # 處理灰階圖片
            else:
                img = preprocess_image(img)

        # 儲存處理後的圖片
        cv2.imwrite(str(dst_img_path), img)
        return True


def create_yolo_kfold_dataset(source_dir, output_dir, n_splits=5, apply_preprocessing=True,
                              has_label=True, multi_channel=None):
    """
    將 YOLO 資料集 (train + val) 依病人 ID 順序切分成 K-Fold 交叉驗證格式
    確保同一病人的所有影像都在同一個 fold 中

    參數:
        source_dir: 原始 YOLO 資料集路徑 (包含 train 和 val 資料夾)
        output_dir: 輸出資料夾路徑
        n_splits: 折數 (預設 5)
        apply_preprocessing: 是否對圖片進行預處理 (預設 True)
        has_label: 是否只包含有對應標籤檔案的圖片 (預設 True)
        multi_channel: 多通道數量（必須是奇數或 None，預設 None）
    """

    # 驗證 multi_channel 參數
    if multi_channel is not None:
        if not isinstance(multi_channel, int) or multi_channel % 2 == 0 or multi_channel < 1:
            raise ValueError(f"multi_channel 必須是奇數（如 3, 5, 7...）或 None，當前值為: {multi_channel}")

    # 設定路徑
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 收集所有影像檔案 (從 train 和 val)
    all_image_files = []
    all_label_dirs = {}  # 記錄每個影像對應的標籤資料夾

    # 支援的影像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 定義要搜索的來源資料夾
    sources_to_check = [
        ("train", source_dir / "train" / "images", source_dir / "train" / "labels"),
        ("val", source_dir / "val" / "images", source_dir / "val" / "labels")
    ]

    print("開始掃描來源資料夾...")
    for name, images_dir, labels_dir in sources_to_check:
        if images_dir.exists():
            print(f"讀取 {name} 資料夾...")
            for ext in image_extensions:
                # 遍歷大小寫副檔名
                for img_file in list(images_dir.glob(f"*{ext}")) + list(images_dir.glob(f"*{ext.upper()}")):

                    # 檢查標籤是否存在
                    if has_label:
                        label_file = labels_dir / f"{img_file.stem}.txt"
                        if not label_file.exists():
                            continue  # 如果要求有標籤但標籤不存在，則跳過此圖片

                    # 將符合條件的圖片加入列表
                    all_image_files.append(img_file)
                    all_label_dirs[img_file] = labels_dir

    if len(all_image_files) == 0:
        error_msg = f"在 {source_dir} 中找不到任何影像檔案。"
        if has_label:
            error_msg += " 請確認影像和標籤檔案是否配對，或嘗試設定 has_label=False。"
        raise FileNotFoundError(error_msg)

    # 依檔名排序
    all_image_files = sorted(all_image_files, key=lambda x: x.name)

    # 輸出設定資訊
    preprocessing_status = "啟用預處理 (123-230 截斷)" if apply_preprocessing else "不進行預處理"
    label_status = "只包含有標籤的圖片" if has_label else "包含所有圖片"
    multichannel_status = f"多通道模式 ({multi_channel} 通道)" if multi_channel else "單通道模式"

    print("-" * 50)
    print(f"掃描完成!")
    print(f"總共找到 {len(all_image_files)} 張符合條件的影像")
    print(f"圖片預處理: {preprocessing_status}")
    print(f"標籤過濾: {label_status}")
    print(f"通道模式: {multichannel_status}")
    print("-" * 50)

    # 按病人 ID 分組
    patient_groups = defaultdict(list)
    for img_file in all_image_files:
        # 從檔名提取病人 ID (例如: patient0007_0013 -> patient0007)
        filename = img_file.stem
        patient_id = filename.split('_')[0]
        patient_groups[patient_id].append(img_file)

    # 對每個病人的圖片按檔名排序（重要：確保時序正確）
    for patient_id in patient_groups:
        patient_groups[patient_id] = sorted(patient_groups[patient_id], key=lambda x: x.name)

    # 取得排序後的病人 ID 列表
    patient_ids = sorted(patient_groups.keys())
    print(f"找到 {len(patient_ids)} 位病人")

    # 顯示每位病人的影像數量 (前 5 位)
    for patient_id in patient_ids[:5]:
        print(f"  {patient_id}: {len(patient_groups[patient_id])} 張影像")
    if len(patient_ids) > 5:
        print(f"  ...")

    # 將病人 ID 順序切分成 n_splits 份
    total_patients = len(patient_ids)
    fold_sizes = np.full(n_splits, total_patients // n_splits, dtype=int)
    fold_sizes[:total_patients % n_splits] += 1  # 將餘數分配給前面的 fold

    # 切分病人 ID
    fold_patient_ids = []
    current_idx = 0
    for size in fold_sizes:
        fold_patient_ids.append(patient_ids[current_idx:current_idx + size])
        current_idx += size

    # 為每個 fold 建立資料集
    for fold_idx in range(n_splits):
        print(f"\n處理 Fold {fold_idx + 1}/{n_splits}...")

        # 驗證集為當前 fold 的病人
        val_patient_ids = fold_patient_ids[fold_idx]

        # 訓練集為其他所有 fold 的病人
        train_patient_ids = []
        for i in range(n_splits):
            if i != fold_idx:
                train_patient_ids.extend(fold_patient_ids[i])

        # 取得對應的影像檔案
        train_images = [img for pid in train_patient_ids for img in patient_groups[pid]]
        val_images = [img for pid in val_patient_ids for img in patient_groups[pid]]

        print(f"  訓練集: {len(train_patient_ids)} 位病人, {len(train_images)} 張影像")
        print(f"  驗證集: {len(val_patient_ids)} 位病人, {len(val_images)} 張影像")

        # 建立資料夾結構
        fold_dir = output_dir / f"fold_{fold_idx + 1}"
        train_img_dir = fold_dir / "train" / "images"
        train_lbl_dir = fold_dir / "train" / "labels"
        val_img_dir = fold_dir / "val" / "images"
        val_lbl_dir = fold_dir / "val" / "labels"

        for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # 複製訓練集檔案
        print(f"  處理訓練集...")
        for patient_id in tqdm(train_patient_ids, desc="  Train", ncols=100):
            patient_images = patient_groups[patient_id]

            for idx, src_img in enumerate(patient_images):
                src_lbl_dir = all_label_dirs[src_img]
                src_lbl = src_lbl_dir / f"{src_img.stem}.txt"
                dst_img = train_img_dir / src_img.name

                # 處理圖片（支援多通道）
                copy_and_preprocess_image(
                    src_img, dst_img, apply_preprocessing,
                    multi_channel=multi_channel,
                    patient_images=patient_images,
                    current_idx=idx
                )

                # 複製標籤
                if src_lbl.exists():
                    shutil.copy2(src_lbl, train_lbl_dir / src_lbl.name)

        # 複製驗證集檔案
        print(f"  處理驗證集...")
        for patient_id in tqdm(val_patient_ids, desc="  Val  ", ncols=100):
            patient_images = patient_groups[patient_id]

            for idx, src_img in enumerate(patient_images):
                src_lbl_dir = all_label_dirs[src_img]
                src_lbl = src_lbl_dir / f"{src_img.stem}.txt"
                dst_img = val_img_dir / src_img.name

                # 處理圖片（支援多通道）
                copy_and_preprocess_image(
                    src_img, dst_img, apply_preprocessing,
                    multi_channel=multi_channel,
                    patient_images=patient_images,
                    current_idx=idx
                )

                # 複製標籤
                if src_lbl.exists():
                    shutil.copy2(src_lbl, val_lbl_dir / src_lbl.name)

        # 建立 baseline.yaml 配置檔案
        create_data_yaml(fold_dir, fold_idx + 1, multi_channel)
        print(f"  Fold {fold_idx + 1} 完成!")

    print(f"\n所有 {n_splits} 個 fold 已成功建立於: {output_dir}")
    print("\n各 Fold 的病人分配:")
    for fold_idx in range(n_splits):
        patient_ids_in_fold = fold_patient_ids[fold_idx]
        print(
            f"  Fold {fold_idx + 1}: {len(patient_ids_in_fold)} 位病人 ({patient_ids_in_fold[0]} ~ {patient_ids_in_fold[-1]})")


def create_data_yaml(fold_dir, fold_idx, multi_channel=None):
    """
    為每個 fold 建立 baseline.yaml 配置檔案

    參數:
        fold_dir: fold 資料夾路徑
        fold_idx: fold 編號
        multi_channel: 多通道數量（如果有的話）
    """
    channel_line = f"channels: {multi_channel}  # 多通道模式\n" if multi_channel else ""

    yaml_content = f"""# YOLO Dataset Configuration - Fold {fold_idx}
path: {fold_dir.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
{channel_line}
# Classes (請根據您的資料集修改)
names:
  0: class_0
  1: class_1
  # 新增更多類別...
"""
    yaml_path = fold_dir / "baseline.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


# 使用範例
if __name__ == "__main__":
    # --- 設定參數 ---
    SOURCE_DIR = "yolo"  # 您的原始 YOLO 資料集路徑
    OUTPUT_DIR = "yolo_5fold_1"  # 輸出資料夾路徑
    N_SPLITS = 5  # K-Fold 折數

    # --- 控制開關 ---
    # 是否對圖片進行預處理 (截斷 123-230 並正規化)
    APPLY_PREPROCESSING = False
    # 是否只處理有對應 .txt 標籤檔案的圖片
    HAS_LABEL = False
    # 多通道設定（必須是奇數，如 3, 5, 7, 9...，或 None 表示單通道）
    MULTI_CHANNEL = 1  # 例如: 5 通道 = [前2張, 前1張, 當前, 後1張, 後2張]

    # 執行切分
    create_yolo_kfold_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        n_splits=N_SPLITS,
        apply_preprocessing=APPLY_PREPROCESSING,
        has_label=HAS_LABEL,
        multi_channel=MULTI_CHANNEL
    )

    print("\n--- 執行完成 ---")
    print(f"K-Fold 資料集已建立於: {Path(OUTPUT_DIR).absolute()}")
    print("\n您可以開始訓練，例如:")
    print(f"yolo train data={OUTPUT_DIR}/fold_1/baseline.yaml model=yolov8n.pt epochs=100")
