import cv2
import numpy as np
import os
import time
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class ObjectTrainer:
    """
    用於訓練和識別自定義物體的類
    可以獨立運行，也可以與FingerObjectRecognizer一起使用
    """
    def __init__(self, base_folder="object_recognition"):
        # 創建必要的資料夾
        self.base_folder = base_folder
        self.data_folder = os.path.join(base_folder, "training_data")
        self.model_folder = os.path.join(base_folder, "model")
        
        for folder in [self.base_folder, self.data_folder, self.model_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # 標籤管理
        self.labels = []
        self.labels_path = os.path.join(self.model_folder, "labels.pkl")
        
        # 模型配置
        self.model_path = os.path.join(self.model_folder, "svm_model.pkl")
        self.scaler_path = os.path.join(self.model_folder, "scaler.pkl")
        self.model = None
        self.scaler = None
        self.img_size = (128, 128)  # 用於特徵提取的標準尺寸
        
        # 嘗試加載現有的模型和標籤
        self.load_model()
        
        # 用於界面的顏色
        self.colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255),
            'cyan': (255, 255, 0)
        }
    
    def load_model(self):
        """加載訓練好的模型和標籤（如果存在）"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model loaded: {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Feature scaler loaded: {self.scaler_path}")
            
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'rb') as f:
                    self.labels = pickle.load(f)
                print(f"Labels loaded: {self.labels}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            self.labels = []
    
    def save_image(self, image, label):
        """保存一張訓練圖像"""
        # 為標籤創建目錄
        label_dir = os.path.join(self.data_folder, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # 生成唯一的文件名
        timestamp = int(time.time() * 1000)
        filename = os.path.join(label_dir, f"{label}_{timestamp}.jpg")
        
        # 保存圖像
        cv2.imwrite(filename, image)
        print(f"Training image saved: {filename}")
        
        # 如果是新標籤，添加到列表
        if label not in self.labels:
            self.labels.append(label)
            # 保存更新後的標籤列表
            with open(self.labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        
        # 返回該標籤的圖像數量
        return len([f for f in os.listdir(label_dir) if f.endswith('.jpg')])
    
    def extract_features(self, image):
        """從圖像中提取特徵"""
        # 調整大小
        img = cv2.resize(image, self.img_size)
        
        # 轉換為灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 提取HOG特徵
        win_size = self.img_size
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(gray)
        
        return features.flatten()
    
    def train_model(self):
        """訓練模型"""
        if len(self.labels) < 2:
            print("At least 2 classes are needed to train the model")
            return False
        
        features = []
        labels = []
        total_images = 0
        
        print(f"Starting model training with the following classes: {self.labels}")
        
        # 從所有圖像中提取特徵
        for label in self.labels:
            label_dir = os.path.join(self.data_folder, label)
            if not os.path.exists(label_dir):
                continue
                
            label_images = 0
            for filename in os.listdir(label_dir):
                if not filename.endswith('.jpg'):
                    continue
                    
                img_path = os.path.join(label_dir, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Unable to read image: {img_path}")
                    continue
                
                # 提取特徵
                feature_vector = self.extract_features(img)
                features.append(feature_vector)
                labels.append(label)
                label_images += 1
            
            total_images += label_images
            print(f"Class '{label}' processed {label_images} images")
        
        if len(features) == 0:
            print("No features extracted. Please check your images.")
            return False
        
        # 轉換為numpy數組
        X = np.array(features)
        y = np.array(labels)
        
        # 標準化特徵
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練SVM模型
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_scaled, y)
        
        # 保存模型和標準化器
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model training completed! Used {total_images} images across {len(self.labels)} classes")
        return True
    
    def predict(self, image):
        """預測圖像的類別"""
        if self.model is None or self.scaler is None:
            return None, 0.0
        
        # 提取特徵
        features = self.extract_features(image)
        features = features.reshape(1, -1)  # 重塑為單個樣本
        
        # 標準化特徵
        scaled_features = self.scaler.transform(features)
        
        # 預測
        predictions = self.model.predict_proba(scaled_features)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # 獲取類別標籤
        predicted_class = self.model.classes_[predicted_class_idx]
        
        return predicted_class, confidence
    
    def capture_from_webcam(self):
        """直接從網絡攝像頭捕獲並訓練圖像"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Unable to open camera")
            return
        
        current_label = ""
        captured_images = {}
        
        print("=== Object Training Mode ===")
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
            cv2.putText(info_frame, "Training Mode", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['red'], 2)
            
            cv2.putText(info_frame, f"Current label: {current_label}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['green'], 2)
            
            if current_label in captured_images:
                cv2.putText(info_frame, f"Captured: {captured_images[current_label]} images", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['green'], 2)
            
            # 顯示中心區域框
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = min(w, h) // 3
            
            # 繪製目標框
            x1, y1 = center_x - box_size, center_y - box_size
            x2, y2 = center_x + box_size, center_y + box_size
            cv2.rectangle(info_frame, (x1, y1), (x2, y2), self.colors['pink'], 2)
            
            # 顯示提示信息
            cv2.putText(info_frame, "Place object in box", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['pink'], 2)
            
            cv2.imshow("Object Training", info_frame)
            
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
                print("Starting model training...")
                if self.train_model():
                    print("Model training successful!")
                else:
                    print("Model training failed!")
        
        cap.release()
        cv2.destroyAllWindows()

    def recognize_from_webcam(self):
        """使用訓練好的模型從網絡攝像頭識別物體"""
        if self.model is None:
            print("No trained model available. Please train the model first.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Unable to open camera")
            return
        
        print("=== Object Recognition Mode ===")
        print("Press 'q' to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to get frame")
                break
            
            # 水平翻轉以獲得鏡像效果
            frame = cv2.flip(frame, 1)
            
            # 顯示識別模式
            info_frame = frame.copy()
            cv2.putText(info_frame, "Recognition Mode", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['blue'], 2)
            
            # 顯示中心區域框
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = min(w, h) // 3
            
            # 繪製目標框
            x1, y1 = center_x - box_size, center_y - box_size
            x2, y2 = center_x + box_size, center_y + box_size
            cv2.rectangle(info_frame, (x1, y1), (x2, y2), self.colors['pink'], 2)
            
            # 識別框內的物體
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                label, confidence = self.predict(roi)
                
                if label and confidence > 0.5:
                    # 顯示預測結果
                    cv2.putText(info_frame, f"{label} ({confidence:.2f})", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, self.colors['pink'], 2)
                else:
                    cv2.putText(info_frame, "Unknown Object", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['pink'], 2)
            
            cv2.imshow("Object Recognition", info_frame)
            
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
    
    parser = argparse.ArgumentParser(description='Object Recognition Training and Testing')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], 
                        default='train', help='Select mode: training or testing')
    
    args = parser.parse_args()
    
    trainer = ObjectTrainer()
    
    if args.mode == 'train':
        trainer.capture_from_webcam()
    else:
        trainer.recognize_from_webcam()