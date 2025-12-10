import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_yolo_label(label_path, img_width, img_height):
    """
    讀取 YOLO 格式標註文件
    返回絕對座標的邊界框列表 [(x1, y1, x2, y2), ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # YOLO 格式: class x_center y_center width height (歸一化座標)
            _, x_center, y_center, width, height = map(float, parts[:5])

            # 轉換為絕對座標
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # 計算邊界框座標
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # 確保座標在圖像範圍內
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            boxes.append((x1, y1, x2, y2))

    return boxes


def collect_box_pixels(dataset_root, splits=['train', 'val']):
    """
    收集所有 bounding box 內的像素值
    """
    all_pixels = []

    for split in splits:
        images_dir = os.path.join(dataset_root, split, 'images')
        labels_dir = os.path.join(dataset_root, split, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"警告: {split} 目錄不存在，跳過")
            continue

        # 獲取所有圖像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(images_dir).glob(ext))

        print(f"處理 {split} 集: 找到 {len(image_files)} 張圖像")

        for img_path in tqdm(image_files, desc=f"處理 {split}"):
            # 讀取圖像
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_height, img_width = img.shape[:2]

            # 讀取對應的標註文件
            label_path = os.path.join(labels_dir, img_path.stem + '.txt')
            boxes = read_yolo_label(label_path, img_width, img_height)

            # 提取每個 box 內的像素值
            for x1, y1, x2, y2 in boxes:
                if x2 > x1 and y2 > y1:
                    box_region = img[y1:y2, x1:x2]
                    # 將所有通道的像素值展平
                    pixels = box_region.flatten()
                    all_pixels.extend(pixels.tolist())

    return np.array(all_pixels)


def calculate_statistics(pixels):
    """
    計算像素值的統計信息
    """
    stats = {
        '總像素數': len(pixels),
        '最小值': np.min(pixels),
        '最大值': np.max(pixels),
        '平均值': np.mean(pixels),
        '0.5百分位': np.percentile(pixels, 0.5),
        '99.5百分位': np.percentile(pixels, 99.5),
        '中位數': np.median(pixels),
        '標準差': np.std(pixels)
    }
    return stats


def plot_distribution(pixels, stats, save_path='pixel_distribution.png'):
    """
    繪製並保存像素分佈直方圖
    """
    print(f"正在繪製分佈圖，請稍候... (數據量: {len(pixels):,})")

    plt.figure(figsize=(12, 6))

    # 設置風格
    plt.style.use('ggplot')

    # 繪製直方圖 (使用 bins=256 對應 0-255 的像素值)
    # density=True 表示顯示頻率密度，這樣數據量大時 y 軸數值不會過大
    plt.hist(pixels, bins=256, range=(0, 255), density=True,
             color='#4F81BD', alpha=0.7, label='Pixel Density')

    # 繪製統計線
    plt.axvline(stats['平均值'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {stats['平均值']:.2f}")
    plt.axvline(stats['中位數'], color='green', linestyle='-', linewidth=2,
                label=f"Median: {stats['中位數']:.2f}")

    # 標註百分位
    plt.axvline(stats['0.5百分位'], color='orange', linestyle=':', alpha=0.6)
    plt.axvline(stats['99.5百分位'], color='orange', linestyle=':', alpha=0.6,
                label='0.5% - 99.5% Range')

    plt.title('Pixel Intensity Distribution within YOLO Bounding Boxes', fontsize=14)
    plt.xlabel('Pixel Value (0-255)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 255)

    # 保存圖片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 關閉圖表釋放內存
    print(f"分佈圖已保存至: {save_path}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 設定數據集根目錄（請根據實際情況修改）
    dataset_root = './yolo'

    print("=" * 60)
    print("YOLO 數據集 Bounding Box 內像素統計與視覺化")
    print("=" * 60)

    # 檢查目錄是否存在
    if not os.path.exists(dataset_root):
        print(f"錯誤: 數據集目錄 {dataset_root} 不存在")
        print("請將此代碼放在包含 'yolo' 文件夾的目錄下")
        print("或修改 dataset_root 變量為正確的路徑")
    else:
        print(f"數據集路徑: {dataset_root}")
        print("\n開始收集像素數據...")

        # 收集所有 box 內的像素值
        pixels = collect_box_pixels(dataset_root, splits=['train', 'val'])

        if len(pixels) > 0:
            print(f"\n成功收集 {len(pixels):,} 個像素值")
            print("\n計算統計信息...")

            # 計算統計信息
            stats = calculate_statistics(pixels)

            print("\n" + "=" * 60)
            print("統計結果")
            print("=" * 60)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key:15s}: {value:.2f}")
                else:
                    print(f"{key:15s}: {value:,}")
            print("=" * 60)

            # 保存統計文字結果
            output_txt = 'bbox_pixel_statistics.txt'
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write("YOLO 數據集 Bounding Box 內像素統計\n")
                f.write("=" * 60 + "\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value:,}\n")
            print(f"\n文字報告已保存至: {output_txt}")

            # ================= 新增：繪製分佈圖 =================
            output_img = 'bbox_pixel_distribution.png'
            try:
                plot_distribution(pixels, stats, output_img)
            except Exception as e:
                print(f"繪圖時發生錯誤: {e}")
            # ===================================================

        else:
            print("\n錯誤: 未能收集到任何像素數據")
            print("請檢查:")
            print("1. 數據集目錄結構是否正確")
            print("2. images 和 labels 文件夾是否存在")
            print("3. 標註文件是否與圖像文件對應")

print("\n程式執行完畢！")