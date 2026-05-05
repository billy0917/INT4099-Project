import os
import shutil
import random
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

class UltralyticsTrainer:
    """
    使用Ultralytics YOLO訓練和識別自定義物體的類
    可以獨立運行，也可以與FingerObjectRecognizer一起使用
    """
    def __init__(self, base_folder="ultralytics_yolo"):
        # 創建必要的資料夾
        self.base_folder = base_folder
        self.data_folder = os.path.join(base_folder, "data")
        self.model_folder = os.path.join(base_folder, "model")
        
        for folder in [self.base_folder, self.data_folder, self.model_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # YOLO資料夾結構
        self.images_train = os.path.join(self.data_folder, "images", "train")
        self.images_val = os.path.join(self.data_folder, "images", "val")
        self.labels_train = os.path.join(self.data_folder, "labels", "train")
        self.labels_val = os.path.join(self.data_folder, "labels", "val")
        
        for folder in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # 標籤管理
        self.labels = []
        self.labels_path = os.path.join(self.model_folder, "labels.txt")
        
        # 模型配置
        self.trained_model_path = os.path.join(self.model_folder, "best.pt")
        self.model = None
        self.conf_threshold = 0.05  # 降低置信度閾值至0.05
        
        # 檢查是否已存在標籤文件
        if os.path.exists(self.labels_path):
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        
        # 顏色
        self.colors = {}
        
        # 嘗試加載模型
        self.load_model()
    
    def load_model(self):
        """加載訓練好的YOLO模型（如果存在）"""
        try:
            if os.path.exists(self.trained_model_path):
                print(f"從以下位置加載模型: {self.trained_model_path}")
                # 使用Ultralytics新的API加載模型
                self.model = YOLO(self.trained_model_path)
                print(f"Ultralytics YOLO模型加載成功")
                # 使用空白圖像測試模型預測功能是否正常
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                try:
                    self.model.predict(test_img, verbose=False)
                    print("模型預測測試成功")
                    return True
                except Exception as test_error:
                    print(f"模型預測測試失敗: {test_error}")
                    return False
            else:
                print(f"未找到訓練好的YOLO模型: {self.trained_model_path}")
                # 嘗試使用默認模型
                try:
                    print("嘗試使用基礎YOLOv8n模型")
                    self.model = YOLO('yolov8n.pt')
                    print("使用基礎YOLOv8n模型（僅能檢測常見物體）")
                    return True
                except:
                    print("無法加載任何YOLO模型")
                    self.model = None
                    return False
        except Exception as e:
            print(f"加載YOLO模型時出錯: {e}")
            self.model = None
            return False
    
    def save_image(self, image, label, bbox=None):
        """
        保存一張訓練圖像
        
        Args:
            image: 要保存的圖像
            label: 物體標籤
            bbox: 物體邊界框 [x1, y1, x2, y2] (如果未提供，則假設物體佔據整個圖像)
        
        Returns:
            保存的圖像計數
        """
        # 檢查圖像是否為空
        if image is None or image.size == 0:
            print("無法保存空圖像")
            return 0
        
        # 檢查標籤是否已存在
        if label not in self.labels:
            self.labels.append(label)
            # 保存更新後的標籤列表
            with open(self.labels_path, 'w') as f:
                for lbl in self.labels:
                    f.write(f"{lbl}\n")
        
        # 生成唯一的文件名
        image_id = len(os.listdir(self.images_train)) + len(os.listdir(self.images_val)) + 1
        image_filename = f"{label}_{image_id}.jpg"
        
        # 隨機決定是訓練集還是驗證集 (80% 訓練, 20% 驗證)
        is_train = random.random() < 0.8
        
        # 確定保存路徑
        if is_train:
            image_path = os.path.join(self.images_train, image_filename)
            label_folder = self.labels_train
        else:
            image_path = os.path.join(self.images_val, image_filename)
            label_folder = self.labels_val
        
        # 保存圖像
        cv2.imwrite(image_path, image)
        
        # 獲取圖像尺寸以進行標準化
        height, width = image.shape[:2]
        
        # 創建YOLO格式的標籤文件
        label_path = os.path.join(label_folder, image_filename.replace('.jpg', '.txt'))
        
        # 如果沒有提供邊界框，則假設物體佔據整個圖像的中心80%
        if bbox is None:
            center_x = 0.5
            center_y = 0.5
            box_width = 0.8
            box_height = 0.8
        else:
            # 將邊界框轉換為YOLO格式 (中心點x, 中心點y, 寬度, 高度) - 所有值在[0,1]之間
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
        
        # 獲取標籤的索引
        label_idx = self.labels.index(label)
        
        # 寫入標籤文件
        with open(label_path, 'w') as f:
            f.write(f"{label_idx} {center_x} {center_y} {box_width} {box_height}")
        
        # 返回該標籤的圖像數量
        return sum(1 for f in os.listdir(self.images_train) if f.startswith(f"{label}_"))
    
    def train_from_images(self, image_paths, label, resize=None):
        """
        從上傳的圖像訓練模型
        
        Args:
            image_paths: 單個圖像路徑或圖像路徑列表
            label: 所有圖像的類別標籤
            resize: 可選，將圖像調整為這個尺寸 (width, height)
        
        Returns:
            保存的圖像數量
        """
        # 確保image_paths是列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        count = 0
        
        # 處理每張圖像
        for img_path in image_paths:
            # 檢查文件是否存在
            if not os.path.exists(img_path):
                print(f"圖像文件不存在: {img_path}")
                continue
            
            # 檢查文件是否為圖像
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in valid_extensions:
                print(f"不支持的文件格式: {img_path}")
                continue
            
            try:
                # 讀取圖像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"無法讀取圖像: {img_path}")
                    continue
                
                # 調整圖像大小（如果需要）
                if resize is not None:
                    img = cv2.resize(img, resize)
                
                # 保存圖像
                img_count = self.save_image(img, label)
                count += 1
                print(f"成功導入圖像: {img_path} (標籤: {label})")
            
            except Exception as e:
                print(f"處理圖像時出錯 {img_path}: {e}")
        
        print(f"共導入 {count} 張圖像，標籤為 {label}")
        return count

    def import_from_folder(self, folder_path, label=None, recursive=False):
        """
        從資料夾導入圖像進行訓練
        
        Args:
            folder_path: 包含圖像的資料夾路徑
            label: 所有圖像的類別標籤。如果為None，將使用資料夾名稱作為標籤
            recursive: 是否遞歸搜索子資料夾
        
        Returns:
            保存的圖像數量
        """
        if not os.path.isdir(folder_path):
            print(f"資料夾不存在: {folder_path}")
            return 0
        
        # 如果沒有提供標籤，使用資料夾名稱
        if label is None:
            label = os.path.basename(folder_path)
        
        # 收集所有圖像文件
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        if recursive:
            # 遞歸搜索所有子資料夾
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_extensions:
                        image_files.append(os.path.join(root, file))
        else:
            # 只搜索當前資料夾
            for file in os.listdir(folder_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            print(f"在資料夾中未找到圖像: {folder_path}")
            return 0
        
        # 使用train_from_images方法處理找到的圖像
        return self.train_from_images(image_files, label)
    
    def create_yolo_dataset_config(self):
        """創建Ultralytics YOLO訓練所需的數據集配置文件"""
        data_yaml_path = os.path.join(self.data_folder, "dataset.yaml")
        
        # 配置數據
        data_yaml = {
            'path': os.path.abspath(self.data_folder),
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'nc': len(self.labels),  # 類別數量
            'names': self.labels     # 類別名稱
        }
        
        # 保存YAML文件
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return data_yaml_path
    
    def train_model(self):
        """使用獨立腳本訓練YOLO模型"""
        if len(self.labels) < 1:
            print("至少需要1個標籤類別才能訓練模型")
            return False
        
        # 檢查訓練圖像數量
        train_images = len(os.listdir(self.images_train))
        val_images = len(os.listdir(self.images_val))
        
        if train_images + val_images < 10:
            print(f"警告: 訓練圖像數量較少 (僅{train_images + val_images}張)，建議每個標籤至少捕獲20-30張圖像")
        
        # 創建數據集配置
        data_yaml_path = self.create_yolo_dataset_config()
        
        # 顯示更多診斷信息
        print(f"資料集配置: {data_yaml_path}")
        print(f"訓練圖像: {train_images}張, 驗證圖像: {val_images}張")
        print(f"標籤: {self.labels}")
        
        # 先釋放現有模型
        if self.model is not None:
            print("釋放現有模型資源...")
            self.model = None
            import gc
            gc.collect()  # 強制垃圾收集
        
        try:
            # 檢查獨立訓練腳本是否存在
            standalone_script = "standalone_yolo_trainer.py"
            
            if not os.path.exists(standalone_script):
                print(f"獨立訓練腳本不存在，無法啟動訓練")
                return False
                
            # 設置日誌文件路徑
            log_file = "yolo_training_log.txt"
            
            # 啟動單獨的進程來運行訓練腳本
            import subprocess
            import sys
            
            print(f"啟動獨立訓練進程，日誌將保存到：{log_file}")
            print("訓練可能需要幾分鐘時間，請耐心等待...")
            print("您可以繼續使用其他功能，訓練會在後台進行")
            print("訓練完成後，在下次啟動程序時模型將可用")
            
            # 使用Popen啟動獨立進程，允許主程序繼續運行
            try:
                # 嘗試使用CREATE_NEW_CONSOLE標誌（僅限Windows）
                training_process = subprocess.Popen(
                    [sys.executable, standalone_script, self.base_folder, log_file],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            except (AttributeError, TypeError):
                # 非Windows系統或CREATE_NEW_CONSOLE不可用
                training_process = subprocess.Popen(
                    [sys.executable, standalone_script, self.base_folder, log_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            print(f"訓練進程已啟動 (PID: {training_process.pid})")
            print(f"請查看訓練日誌: {log_file}")
            
            # 不等待進程完成，讓UI保持響應
            print("您可以關閉程序，訓練將繼續在後台運行")
            
            return True  # 返回True表示訓練已啟動
            
        except Exception as e:
            print(f"啟動訓練時出錯: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image):
        """
        預測圖像中的物體
        
        Args:
            image: 要識別的圖像
            
        Returns:
            (label, confidence): 識別到的標籤和置信度
            如果未識別到物體，則返回(None, 0.0)
        """
        if self.model is None:
            return None, 0.0
        
        # 檢查圖像是否為空
        if image is None or image.size == 0:
            print("無法預測空圖像")
            return None, 0.0
        
        try:
            # 嘗試增強圖像
            alpha = 1.2  # 對比度增強
            beta = 10    # 亮度增強
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # 先嘗試使用增強的圖像
            results = self.model.predict(enhanced, conf=self.conf_threshold, verbose=False)
            
            # 如果增強圖像沒有檢測結果，嘗試原始圖像
            if len(results[0].boxes) == 0:
                results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
            
            # 獲取結果
            boxes = results[0].boxes
            if len(boxes) == 0:
                return None, 0.0
                
            # 獲取最高置信度的檢測
            max_conf_idx = boxes.conf.argmax().item()
            confidence = boxes.conf[max_conf_idx].item()
            class_id = int(boxes.cls[max_conf_idx].item())
            
            # 獲取對應的標籤名稱
            if class_id < len(self.labels):
                label = self.labels[class_id]
            else:
                label = f"Class_{class_id}"
            
            return label, confidence
        except Exception as e:
            print(f"預測錯誤: {e}")
            return None, 0.0
    
    def capture_from_webcam(self):
        """直接從網絡攝像頭捕獲並訓練圖像"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Unable to open camera")
            return
        
        current_label = ""
        captured_images = {}
        
        print("=== Ultralytics YOLO Object Training Mode ===")
        print("Press 'l' to set label")
        print("Press 'c' to capture image")
        print("Press 't' to train model")
        print("Press 'q' to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to get frame")
                break
            
            # 水平翻轉以獲得鏡像效果
            frame = cv2.flip(frame, 1)
            
            # 顯示當前標籤和已捕獲的圖像數量
            info_frame = frame.copy()
            cv2.putText(info_frame, "YOLO Training Mode", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(info_frame, f"Current label: {current_label}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if current_label in captured_images:
                cv2.putText(info_frame, f"Captured: {captured_images[current_label]} images", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示中心區域框
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = min(w, h) // 3
            
            # 繪製目標框
            x1, y1 = center_x - box_size, center_y - box_size
            x2, y2 = center_x + box_size, center_y + box_size
            cv2.rectangle(info_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # 顯示提示信息
            cv2.putText(info_frame, "Place object in box", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("YOLO Object Training", info_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                current_label = input("Enter object label name: ")
                if current_label not in captured_images:
                    captured_images[current_label] = 0
                print(f"Current label set to: {current_label}")
            elif key == ord('c'):
                if not current_label:
                    print("Please set a label first")
                    continue
                
                # 捕獲中心區域
                roi = frame[y1:y2, x1:x2]
                count = self.save_image(roi, current_label)
                captured_images[current_label] = count
                print(f"Captured image {count} for '{current_label}'")
            elif key == ord('t'):
                print("Starting YOLO model training...")
                if self.train_model():
                    print("YOLO model training successful!")
                else:
                    print("YOLO model training failed!")
        
        cap.release()
        cv2.destroyAllWindows()

    def recognize_from_webcam(self):
        """使用訓練好的YOLO模型從網絡攝像頭識別物體"""
        if self.model is None:
            if not self.load_model():
                print("No trained YOLO model available. Please train the model first.")
                return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Unable to open camera")
            return
        
        print("=== YOLO Object Recognition Mode ===")
        print("Press 'q' to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to get frame")
                break
            
            # 水平翻轉以獲得鏡像效果
            frame = cv2.flip(frame, 1)
            
            # 執行預測
            if self.model:
                # 使用Ultralytics API進行預測並獲取結果
                results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
                
                # 在圖像上繪製結果
                annotated_frame = results[0].plot()
                cv2.imshow("YOLO Object Recognition", annotated_frame)
            else:
                cv2.putText(frame, "No YOLO model loaded", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("YOLO Object Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def integrate_with_finger_recognizer(self, frame, box_coordinates):
        """與手指識別系統集成，提供對框選區域的識別"""
        if self.model is None:
            return "Untrained Model", 0.0
        
        x1, y1, x2, y2 = box_coordinates
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "Invalid Area", 0.0
        
        return self.predict(roi)


# 獨立運行訓練和識別
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Object Recognition')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], 
                        default='train', help='Select mode: training or testing')
    
    args = parser.parse_args()
    
    trainer = UltralyticsTrainer()
    
    if args.mode == 'train':
        trainer.capture_from_webcam()
    else:
        trainer.recognize_from_webcam()