import os
import cv2
from tqdm import tqdm

from ultralytics import YOLO

# 1. 初始化兩個YOLO模型
detect_model = YOLO('/root/aicup2025_II/upload/model/detect/1024/fold5_yolo11s2/weights/best.pt')

# 2. 定義輸入和輸出路徑
input_folder = 'datasets/privateTest'
output_file = 'upload_txt/fold5_yolo11s_last_1024_768_2.txt'

# 3. 從txt檔案名稱提取視覺化資料夾名稱
txt_filename = os.path.splitext(os.path.basename(output_file))[0]
visualize_folder = f'./visualize/{txt_filename}'

# 4. 創建視覺化資料夾和YOLO格式標註資料夾
os.makedirs(visualize_folder, exist_ok=True)
yolo_labels_folder = f'./yolo_labels/{txt_filename}'
os.makedirs(yolo_labels_folder, exist_ok=True)

def draw_boxes_on_image(image, boxes):
    """
    在圖片上手動繪製邊界框和標籤

    參數:
        image: 原始圖片(numpy array)
        boxes: 包含預測框信息的列表

    返回:
        繪製後的圖片
    """
    img_copy = image.copy()

    for box_info in boxes:
        x1, y1, x2, y2 = box_info['coords']
        class_id = box_info['class_id']
        confidence = box_info['confidence']

        # 繪製矩形框（綠色，線寬2）
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 準備標籤文字
        label = f"Class {class_id}: {confidence:.2f}"

        # 設定字體參數
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # 計算文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # 繪製文字背景（深綠色填充）
        cv2.rectangle(
            img_copy,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            (0, 180, 0),
            cv2.FILLED
        )

        # 繪製文字（白色）
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

    return img_copy

def xyxy_to_yolo_format(x1, y1, x2, y2, img_width, img_height):
    """
    將xyxy格式轉換為YOLO格式（正規化的xywh）

    參數:
        x1, y1, x2, y2: 邊界框的左上角和右下角座標（像素）
        img_width, img_height: 圖片尺寸

    返回:
        x_center, y_center, width, height: YOLO格式的正規化座標
    """
    # 計算中心點座標（像素）
    x_center_pixel = (x1 + x2) / 2
    y_center_pixel = (y1 + y2) / 2

    # 計算寬度和高度（像素）
    width_pixel = x2 - x1
    height_pixel = y2 - y1

    # 正規化座標（除以圖片尺寸）
    x_center = x_center_pixel / img_width
    y_center = y_center_pixel / img_height
    width = width_pixel / img_width
    height = height_pixel / img_height

    return x_center, y_center, width, height

def run_yolo_prediction():
    """
    執行YOLO預測，將結果寫入檔案、手動繪製視覺化圖片，並輸出YOLO格式標註
    """
    with open(output_file, 'w') as f:
        for img_name in tqdm(os.listdir(input_folder)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(input_folder, img_name)

                # 讀取原始圖片
                original_img = cv2.imread(img_path)
                img_height, img_width = original_img.shape[:2]

                # 執行物件偵測
                detect_results = detect_model.predict(
                    img_path, conf=0.1, iou=0.7, verbose=False, max_det=3, imgsz=768
                )

                boxes_to_draw = []
                yolo_annotations = []

                for result in detect_results:
                    # 寫入預測結果到txt檔案
                    for box in result.boxes:
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())

                        img_name_without_ext = os.path.splitext(img_name)[0]
                        line = f"{img_name_without_ext} {class_id} {confidence} {x1} {y1} {x2} {y2}\n"
                        f.write(line)

                        # 收集邊界框信息用於繪製
                        boxes_to_draw.append({
                            'coords': (x1, y1, x2, y2),
                            'class_id': class_id,
                            'confidence': confidence
                        })

                        # 轉換為YOLO格式並收集
                        x_center, y_center, width, height = xyxy_to_yolo_format(
                            x1, y1, x2, y2, img_width, img_height
                        )
                        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        yolo_annotations.append(yolo_line)

                # 手動繪製預測框
                # if boxes_to_draw:
                #     annotated_img = draw_boxes_on_image(original_img, boxes_to_draw)
                # else:
                #     annotated_img = original_img
                #
                # # 儲存視覺化圖片
                # output_img_path = os.path.join(visualize_folder, img_name)
                # cv2.imwrite(output_img_path, annotated_img)

                # 儲存YOLO格式標註檔案
                img_name_without_ext = os.path.splitext(img_name)[0]
                yolo_label_path = os.path.join(yolo_labels_folder, f"{img_name_without_ext}.txt")
                with open(yolo_label_path, 'w') as yolo_file:
                    yolo_file.write('\n'.join(yolo_annotations))

    print(f"預測結果已寫入 {output_file}")
    print(f"視覺化圖片已儲存至 {visualize_folder}")
    print(f"YOLO格式標註已儲存至 {yolo_labels_folder}")

if __name__ == "__main__":
    run_yolo_prediction()
