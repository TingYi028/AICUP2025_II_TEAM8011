import os
import glob
import yaml

# --- 1. 從 YAML 檔案載入資料集資訊 ---
# 和您主程式一樣，先讀取 yaml 檔來定位標籤資料夾
data_yaml_path = r'D:\Pycharm Project\aicup2025_II\datasets\all_yolo_data\yolo_5fold\fold_5\baseline.yaml'

with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# --- 2. 確定標籤檔案的路徑 ---
# 根據您的 yaml 設定，推斷出標籤資料夾的路徑
# 這段邏輯需要和您的主程式 (coco_val.py) 保持一致
try:
    # 組合出驗證集圖片資料夾的完整路徑
    val_images_dir = os.path.join(data_config['path'], data_config['val'])

    # 通常標籤資料夾與圖片資料夾結構對稱，只是 'images' 變成 'labels'
    labels_dir = val_images_dir.replace('images', 'labels')

    if not os.path.isdir(labels_dir):
        # 如果路徑替換失敗，拋出錯誤讓使用者手動確認
        raise FileNotFoundError(f"自動推斷的標籤路徑不存在: {labels_dir}")

except KeyError as e:
    print(f"錯誤：您的 {data_yaml_path} 檔案中缺少 '{e.args[0]}' 這個鍵。請檢查檔案內容。")
    exit()
except FileNotFoundError as e:
    print(f"錯誤：{e}")
    print("請手動確認並在下方 'labels_dir' 變數中填入正確的標籤資料夾路徑。")
    # 如果自動推斷失敗，請在此手動填寫正確的路徑
    labels_dir = '/path/to/your/label/folder'


print(f"準備處理標籤資料夾: {labels_dir}")

# --- 3. 遍歷並修正所有 .txt 標籤檔 ---
# 使用 glob 尋找資料夾中所有的 .txt 檔案
label_files = glob.glob(os.path.join(labels_dir, '*.txt'))

if not label_files:
    print(f"警告：在 '{labels_dir}' 中沒有找到任何 .txt 檔案。請確認路徑是否正確。")
else:
    corrected_count = 0
    for file_path in label_files:
        is_corrected = False
        new_lines = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue # 跳過空行

                # 取得類別 ID 字串
                class_id_str = parts[0]

                # 檢查是否為浮點數格式
                if '.' in class_id_str:
                    # 先轉成 float，再轉成 int，最後轉回 string
                    correct_class_id = str(int(float(class_id_str)))
                    parts[0] = correct_class_id
                    new_line = " ".join(parts)
                    new_lines.append(new_line + '\n')
                    is_corrected = True
                else:
                    new_lines.append(line) # 如果格式正確，直接保留原樣

            # 如果檔案有被修正，才進行寫入操作
            if is_corrected:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                corrected_count += 1

        except Exception as e:
            print(f"處理檔案 {file_path} 時發生錯誤: {e}")

    print(f"處理完成！總共檢查了 {len(label_files)} 個檔案，其中 {corrected_count} 個檔案的格式已被修正。")


