import sys
import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import logging
import pickle
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                              QMessageBox, QListWidget, QFileDialog, QSplitter,
                              QDialog, QGridLayout, QScrollArea, QSlider,
                              QProgressBar, QComboBox)
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QSize, QThread, QMutex
from attendance import add_attendance_tab


# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("face_recognition_app.log"),
                             logging.StreamHandler()])
logger = logging.getLogger(__name__)


class ImageProcessing:
    """圖像處理工具類"""
    
    @staticmethod
    def resize_image(image, width=None, height=None):
        """調整圖像大小保持比例"""
        if width is None and height is None:
            return image
            
        h, w = image.shape[:2]
        if width is None:
            aspect = height / float(h)
            dim = (int(w * aspect), height)
        else:
            aspect = width / float(w)
            dim = (width, int(h * aspect))
            
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_brightness(image):
        """規範化圖像亮度"""
        # 轉換到 LAB 顏色空間
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 對 L 通道應用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 合併通道
        limg = cv2.merge((cl, a, b))
        
        # 轉回 RGB
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final


class FaceRecognitionThread(QThread):
    """人臉識別線程"""
    # 定義信號
    result_ready = Signal(object, list, list, list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = 0.6
        self.running = False
        self.mutex = QMutex()
        self.process_size = (320, 240)  # 縮小尺寸進行處理
        self.last_processed_time = 0  # 上次處理時間
        self.min_process_interval = 200  # 最小處理間隔(毫秒)
        
    def set_frame(self, frame):
        """設置要處理的幀"""
        if frame is not None:
            self.mutex.lock()
            self.frame = frame.copy()
            self.mutex.unlock()
            
    def set_known_faces(self, encodings, names):
        """設置已知人臉"""
        self.mutex.lock()
        self.known_face_encodings = encodings.copy()
        self.known_face_names = names.copy()
        self.mutex.unlock()
        
    def set_tolerance(self, tolerance):
        """設置容差"""
        self.tolerance = tolerance
        
    def stop(self):
        """停止線程"""
        self.running = False
        self.wait()
        
    def run(self):
        """執行人臉識別"""
        self.running = True
        
        while self.running:
            # 檢查是否達到最小處理間隔
            current_time = datetime.now().timestamp() * 1000  # 轉換為毫秒
            if current_time - self.last_processed_time < self.min_process_interval:
                self.msleep(10)  # 短暫休眠以減少CPU使用
                continue
                
            self.mutex.lock()
            if self.frame is None or len(self.known_face_encodings) == 0:
                self.mutex.unlock()
                self.msleep(30)
                continue
                
            # 複製數據以避免競態條件
            frame = self.frame.copy()
            known_encodings = self.known_face_encodings.copy()
            known_names = self.known_face_names.copy()
            tolerance = self.tolerance
            self.mutex.unlock()
            
            # 調整圖像大小以加快處理速度
            small_frame = cv2.resize(frame, self.process_size, interpolation=cv2.INTER_AREA)
            
            # 預處理圖像
            processed_frame = ImageProcessing.normalize_brightness(small_frame)
                
            # 檢測人臉位置 - 使用更快的模型
            face_locations = face_recognition.face_locations(processed_frame, model="hog")
            
            # 更新處理時間
            self.last_processed_time = datetime.now().timestamp() * 1000
            
            if not face_locations:
                self.msleep(30)
                continue
                
            # 提取人臉編碼
            face_encodings = face_recognition.face_encodings(processed_frame, face_locations)
            
            # 轉換回原始尺寸的座標
            h_ratio = frame.shape[0] / processed_frame.shape[0]
            w_ratio = frame.shape[1] / processed_frame.shape[1]
            
            original_locations = []
            for top, right, bottom, left in face_locations:
                original_top = int(top * h_ratio)
                original_right = int(right * w_ratio)
                original_bottom = int(bottom * h_ratio)
                original_left = int(left * w_ratio)
                original_locations.append((original_top, original_right, original_bottom, original_left))
            
            # 識別結果
            names = []
            confidences = []
            
            for face_encoding in face_encodings:
                # 計算所有已知人臉的距離
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    # 找到最小距離
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    if min_distance <= tolerance:
                        name = known_names[best_match_index]
                        confidence = 1 - min_distance  # 轉換為置信度
                    else:
                        name = "未知"
                        confidence = 0.0
                        
                    names.append(name)
                    confidences.append(confidence)
                else:
                    names.append("未知")
                    confidences.append(0.0)
            
            # 發送結果
            self.result_ready.emit(frame, original_locations, names, confidences)
            
            # 添加短暫延遲以避免CPU使用率過高
            self.msleep(30)


class CameraWidget(QWidget):
    """攝像頭顯示控件"""
    
    def __init__(self, parent=None, camera_id=0):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera = None
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.recognition_active = False  # 識別模式標誌
        self.displaying_results = False  # 是否正在顯示結果的標誌
        
        # 新增：存儲最近的識別結果
        self.last_recognition_frame = None  # 最後一幀包含識別結果的完整幀
        self.last_face_locations = []  # 最後一次檢測到的人臉位置
        self.last_names = []  # 最後一次識別的人名
        self.last_confidences = []  # 最後一次識別的置信度
        
        # 創建攝像頭選擇下拉框
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("選擇攝像頭:"))
        self.camera_selector = QComboBox()
        self.refresh_cameras()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        camera_layout.addWidget(self.camera_selector)
        
        # 刷新攝像頭按鈕
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(refresh_btn)
        
        layout = QVBoxLayout()
        layout.addLayout(camera_layout)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
        # 初始化攝像頭
        self.init_camera()
        
        # 設置定時器，定期更新攝像頭圖像
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 約30FPS
        
    def refresh_cameras(self):
        """刷新可用攝像頭列表"""
        self.camera_selector.clear()
        # 檢測系統中的攝像頭
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.camera_selector.addItem(f"攝像頭 {index}")
            cap.release()
            index += 1
            if index > 10:  # 最多檢測10個攝像頭
                break
                
        if self.camera_selector.count() == 0:
            self.camera_selector.addItem("未找到攝像頭")
        else:
            # 選擇當前攝像頭
            self.camera_selector.setCurrentIndex(min(self.camera_id, self.camera_selector.count()-1))
    
    def change_camera(self, index):
        """切換攝像頭"""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            
        self.camera_id = index
        self.init_camera()
        
    def init_camera(self):
        """初始化攝像頭"""
        self.camera = cv2.VideoCapture(self.camera_id)
        
        # 設置較低解析度以提高性能
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.camera.isOpened():
            QMessageBox.critical(self, "錯誤", f"無法打開攝像頭 {self.camera_id}！")
            logger.error(f"無法打開攝像頭 {self.camera_id}")
            
    def update_frame(self):
        """更新攝像頭畫面"""
        if self.camera is None or not self.camera.isOpened():
            return
            
        ret, frame = self.camera.read()
        if not ret:
            return
            
        # 將圖像轉換為RGB（face_recognition庫使用RGB）
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 防止與識別結果更新衝突
        if not self.displaying_results:
            display_frame = self.current_frame.copy()
            
            # 如果有識別模式開啟且有先前的識別結果，在當前幀上繪製識別框
            if self.recognition_active and self.last_face_locations and len(self.last_face_locations) == len(self.last_names):
                # 在當前幀上繪製上一次的識別結果
                for (top, right, bottom, left), name, confidence in zip(
                        self.last_face_locations, self.last_names, self.last_confidences):
                    
                    # 顯示名稱和置信度
                    label = f"{name} ({confidence:.2f})"
                    
                    # 根據置信度決定顏色 (綠色=高置信度，紅色=低置信度)
                    if name != "未知":
                        # 從紅色到綠色的漸變
                        green = min(255, int(confidence * 255))
                        red = min(255, int((1 - confidence) * 255))
                        color = (red, green, 0)  # RGB
                    else:
                        color = (255, 0, 0)  # 紅色表示未知
                    
                    # 繪製人臉框和名稱
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # 添加底部填充
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    # 繪製文本 (使用白色)
                    cv2.putText(display_frame, label, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # 將OpenCV圖像轉換為QImage
            height, width, channels = display_frame.shape
            bytes_per_line = channels * width
            q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 設置圖像到標籤
            self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def get_current_frame(self):
        """獲取當前幀"""
        if hasattr(self, 'current_frame'):
            return self.current_frame.copy()
        return None
        
    def closeEvent(self, event):
        """關閉事件，釋放攝像頭"""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        event.accept()


class ImageViewer(QDialog):
    """圖像查看器對話框，用於查看訓練圖像"""
    
    def __init__(self, images, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{name}的訓練圖像")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # 創建滾動區域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 創建網格布局來顯示圖像
        container = QWidget()
        grid_layout = QGridLayout(container)
        
        # 每行顯示4張圖像
        cols = 4
        for i, image_path in enumerate(images):
            try:
                # 讀取圖像
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 調整圖像大小
                img = ImageProcessing.resize_image(img, width=150)
                
                # 創建QLabel顯示圖像
                img_label = QLabel()
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                img_label.setPixmap(QPixmap.fromImage(q_img))
                
                # 添加到網格
                row, col = i // cols, i % cols
                grid_layout.addWidget(img_label, row, col)
                
            except Exception as e:
                logger.error(f"顯示圖像錯誤: {e}")
                
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
        
        # 添加關閉按鈕
        close_btn = QPushButton("關閉")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class FaceRecognitionApp(QMainWindow):
    """人臉識別應用程序主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("人臉識別應用程序")
        self.setMinimumSize(1200, 800)
        
        # 人臉數據存儲路徑
        self.data_dir = "face_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 已知的人臉編碼和對應的名稱
        self.known_face_encodings = []
        self.known_face_names = []
        self.names_to_encodings = {}  # 名稱到編碼的映射
        
        # 使用面部編碼的緩存，避免每次都重新計算
        self.face_encodings_cache = {}
        
        # 人臉識別參數
        self.tolerance = 0.6  # 默認容差
        self.min_training_images = 5  # 建議的最小訓練圖像數
        
        # 當前要訓練的人名
        self.current_person_name = ""
        self.training_count = 0
        
        # 用於限制識別頻率的計數器和閾值
        self.frame_counter = 0
        self.skip_frames = 2  # 每隔多少幀進行一次識別
        
        # 創建UI
        self.init_ui()
        
        # 創建人臉識別線程
        self.recognition_thread = FaceRecognitionThread()
        self.recognition_thread.result_ready.connect(self.update_recognition_results)
        
        # 載入已有的人臉數據
        self.load_known_faces()
        
    def init_ui(self):
        """初始化UI"""
        # 主佈局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # 左側：攝像頭和控制面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 攝像頭顯示
        self.camera_widget = CameraWidget()
        left_layout.addWidget(self.camera_widget)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 姓名輸入
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("姓名:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        control_layout.addLayout(name_layout)
        
        # 拍照按鈕
        self.capture_btn = QPushButton("拍照訓練")
        self.capture_btn.clicked.connect(self.capture_training_image)
        control_layout.addWidget(self.capture_btn)
        
        # 開始識別按鈕
        self.recognize_btn = QPushButton("開始識別")
        self.recognize_btn.clicked.connect(self.toggle_recognition)
        self.recognize_btn.setCheckable(True)
        control_layout.addWidget(self.recognize_btn)
        
        # 連續拍照按鈕
        self.continuous_capture_btn = QPushButton("連續拍攝")
        self.continuous_capture_btn.setCheckable(True)
        self.continuous_capture_btn.clicked.connect(self.toggle_continuous_capture)
        control_layout.addWidget(self.continuous_capture_btn)
        
        left_layout.addLayout(control_layout)
        
        # 參數設置
        param_layout = QHBoxLayout()
        
        # 容差調整
        param_layout.addWidget(QLabel("識別容差:"))
        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setMinimum(20)
        self.tolerance_slider.setMaximum(100)
        self.tolerance_slider.setValue(int(self.tolerance * 100))
        self.tolerance_slider.setTickPosition(QSlider.TicksBelow)
        self.tolerance_slider.setTickInterval(10)
        self.tolerance_slider.valueChanged.connect(self.update_tolerance)
        param_layout.addWidget(self.tolerance_slider)
        
        self.tolerance_label = QLabel(f"{self.tolerance:.2f}")
        param_layout.addWidget(self.tolerance_label)
        
        # 跳幀設置
        param_layout.addWidget(QLabel("跳幀數:"))
        self.skip_frames_slider = QSlider(Qt.Horizontal)
        self.skip_frames_slider.setMinimum(0)
        self.skip_frames_slider.setMaximum(10)
        self.skip_frames_slider.setValue(self.skip_frames)
        self.skip_frames_slider.setTickPosition(QSlider.TicksBelow)
        self.skip_frames_slider.setTickInterval(1)
        self.skip_frames_slider.valueChanged.connect(self.update_skip_frames)
        param_layout.addWidget(self.skip_frames_slider)
        
        self.skip_frames_label = QLabel(f"{self.skip_frames}")
        param_layout.addWidget(self.skip_frames_label)
        
        left_layout.addLayout(param_layout)
        
        # 右側：已訓練的人臉列表和訓練數據
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("已訓練的人臉:"))
        self.faces_list = QListWidget()
        self.faces_list.itemClicked.connect(self.show_person_images)
        right_layout.addWidget(self.faces_list)
        
        # 操作按鈕
        action_layout = QHBoxLayout()
        
        # 刪除人臉按鈕
        self.delete_face_btn = QPushButton("刪除選中人臉")
        self.delete_face_btn.clicked.connect(self.delete_selected_face)
        action_layout.addWidget(self.delete_face_btn)
        
        # 刪除所有按鈕
        self.delete_all_btn = QPushButton("刪除所有人臉")
        self.delete_all_btn.clicked.connect(self.delete_all_faces)
        action_layout.addWidget(self.delete_all_btn)
        
        right_layout.addLayout(action_layout)
        
        # 狀態顯示
        self.status_label = QLabel("就緒")
        right_layout.addWidget(self.status_label)
        
        # 進度顯示
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)
        
        # 人臉識別定時器（默認不啟動）
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.feed_recognition_thread)
        
        # 連續拍照定時器
        self.continuous_capture_timer = QTimer()
        self.continuous_capture_timer.timeout.connect(self.capture_training_image)
        
    def update_tolerance(self):
        """更新容差值"""
        self.tolerance = self.tolerance_slider.value() / 100.0
        self.tolerance_label.setText(f"{self.tolerance:.2f}")
        self.recognition_thread.set_tolerance(self.tolerance)
        
    def update_skip_frames(self):
        """更新跳幀數"""
        self.skip_frames = self.skip_frames_slider.value()
        self.skip_frames_label.setText(f"{self.skip_frames}")
        
    def toggle_continuous_capture(self):
        """切換連續拍照模式"""
        if self.continuous_capture_btn.isChecked():
            name = self.name_input.text().strip()
            if not name:
                QMessageBox.warning(self, "警告", "請輸入姓名！")
                self.continuous_capture_btn.setChecked(False)
                return
                
            self.continuous_capture_btn.setText("停止拍攝")
            # 每2秒拍一次照
            self.continuous_capture_timer.start(2000)
        else:
            self.continuous_capture_btn.setText("連續拍攝")
            self.continuous_capture_timer.stop()
            
    def capture_training_image(self):
        """捕獲訓練圖像"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "請輸入姓名！")
            return
            
        # 獲取當前幀
        frame = self.camera_widget.get_current_frame()
        if frame is None:
            QMessageBox.warning(self, "警告", "無法獲取圖像！")
            return
            
        # 預處理圖像 - 規範化亮度
        processed_frame = ImageProcessing.normalize_brightness(frame)
        
        # 檢測人臉
        face_locations = face_recognition.face_locations(processed_frame)
        if not face_locations:
            if not self.continuous_capture_btn.isChecked():  # 只在手動模式下顯示警告
                QMessageBox.warning(self, "警告", "未檢測到人臉！")
            return
            
        # 只使用第一個檢測到的人臉（假設只有一個人）
        top, right, bottom, left = face_locations[0]
        
        # 截取人臉圖像，稍微擴大區域
        height, width = processed_frame.shape[:2]
        # 擴展邊界20%
        expand_percent = 0.2
        expand_x = int((right - left) * expand_percent)
        expand_y = int((bottom - top) * expand_percent)
        
        # 確保不超出圖像邊界
        left = max(0, left - expand_x)
        top = max(0, top - expand_y)
        right = min(width, right + expand_x)
        bottom = min(height, bottom + expand_y)
        
        face_image = processed_frame[top:bottom, left:right]
        
        # 為該人創建目錄
        person_dir = os.path.join(self.data_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # 保存人臉圖像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(person_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        
        # 更新訓練計數
        self.training_count += 1
        self.status_label.setText(f"已為 {name} 拍攝 {self.training_count} 張訓練照片")
        
        # 如果是新的人，更新人臉列表
        if name not in self.known_face_names:
            self.faces_list.addItem(name)
            
        # 提取並保存人臉編碼
        face_encodings = face_recognition.face_encodings(processed_frame, [face_locations[0]])
        if face_encodings:
            face_encoding = face_encodings[0]
            self.save_face_encoding(name, face_encoding)
        else:
            logger.warning(f"無法為 {name} 提取人臉編碼")
            
    def save_face_encoding(self, name, encoding):
        """保存人臉編碼"""
        # 確保編碼目錄存在
        encodings_dir = os.path.join(self.data_dir, "encodings")
        os.makedirs(encodings_dir, exist_ok=True)
        
        # 獲取或創建該人的編碼列表
        if name in self.names_to_encodings:
            encodings = self.names_to_encodings[name]
            encodings.append(encoding)
        else:
            encodings = [encoding]
            self.names_to_encodings[name] = encodings
            
        # 更新全局編碼列表
        if name not in self.known_face_names:
            self.known_face_names.append(name)
            
        self.known_face_encodings.append(encoding)
        
        # 將更新後的編碼保存到文件
        self.save_encodings_to_file(name, encodings)
        
        # 更新識別線程的已知人臉
        self.recognition_thread.set_known_faces(
            np.array(self.known_face_encodings), 
            self.known_face_names
        )
        
    def save_encodings_to_file(self, name, encodings):
        """將人臉編碼保存到文件"""
        # 使用pickle保存，更可靠
        encodings_dir = os.path.join(self.data_dir, "encodings")
        encoding_file = os.path.join(encodings_dir, f"{name}.pkl")
        
        try:
            with open(encoding_file, 'wb') as f:
                pickle.dump(encodings, f)
            logger.info(f"已保存 {name} 的 {len(encodings)} 個人臉編碼")
        except Exception as e:
            logger.error(f"保存編碼錯誤: {e}")
            
        # 同時保存為numpy格式作為備份
        np_file = os.path.join(encodings_dir, f"{name}.npy")
        try:
            np.save(np_file, np.array(encodings))
        except Exception as e:
            logger.error(f"保存numpy編碼錯誤: {e}")
            
    def load_known_faces(self):
        """載入已知的人臉編碼"""
        encodings_dir = os.path.join(self.data_dir, "encodings")
        if not os.path.exists(encodings_dir):
            return
            
        # 清空現有數據
        self.known_face_encodings = []
        self.known_face_names = []
        self.names_to_encodings = {}
        self.faces_list.clear()
        
        # 遍歷編碼目錄
        total_encodings = 0
        
        # 首先查找pickle文件
        for filename in os.listdir(encodings_dir):
            if filename.endswith(".pkl"):
                name = os.path.splitext(filename)[0]
                encoding_file = os.path.join(encodings_dir, filename)
                
                try:
                    # 載入編碼
                    with open(encoding_file, 'rb') as f:
                        encodings = pickle.load(f)
                        
                    # 將每個編碼添加到列表
                    self.names_to_encodings[name] = encodings
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)
                        
                    total_encodings += len(encodings)
                        
                    # 添加到UI列表（不重複）
                    if name not in [self.faces_list.item(i).text() for i in range(self.faces_list.count())]:
                        self.faces_list.addItem(name)
                        
                except Exception as e:
                    logger.error(f"載入編碼錯誤: {e}")
                    # 嘗試加載numpy文件作為備份
                    try:
                        np_file = os.path.join(encodings_dir, f"{name}.npy")
                        if os.path.exists(np_file):
                            encodings = np.load(np_file)
                            self.names_to_encodings[name] = encodings
                            for encoding in encodings:
                                self.known_face_encodings.append(encoding)
                                self.known_face_names.append(name)
                                
                            total_encodings += len(encodings)
                                
                            # 添加到UI列表（不重複）
                            if name not in [self.faces_list.item(i).text() for i in range(self.faces_list.count())]:
                                self.faces_list.addItem(name)
                    except Exception as e2:
                        logger.error(f"載入numpy編碼錯誤: {e2}")
                    
        # 如果沒有pickle文件，查找numpy文件
        if total_encodings == 0:
            for filename in os.listdir(encodings_dir):
                if filename.endswith(".npy"):
                    name = os.path.splitext(filename)[0]
                    encoding_file = os.path.join(encodings_dir, filename)
                    
                    try:
                        # 載入編碼
                        encodings = np.load(encoding_file)
                        
                        # 確保編碼是二維數組
                        if len(encodings.shape) == 1:
                            encodings = encodings.reshape(1, -1)
                            
                        # 將每個編碼添加到列表
                        self.names_to_encodings[name] = encodings
                        for encoding in encodings:
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(name)
                            
                        total_encodings += len(encodings)
                            
                        # 添加到UI列表（不重複）
                        if name not in [self.faces_list.item(i).text() for i in range(self.faces_list.count())]:
                            self.faces_list.addItem(name)
                            
                        # 將numpy數據轉換為pickle格式以便今後使用
                        self.save_encodings_to_file(name, encodings)
                            
                    except Exception as e:
                        logger.error(f"載入numpy編碼錯誤: {e}")
        
        # 初始化識別線程的已知人臉
        if hasattr(self, 'recognition_thread') and self.known_face_encodings:
            self.recognition_thread.set_known_faces(
                np.array(self.known_face_encodings), 
                self.known_face_names
            )
        
        unique_faces = len(set(self.known_face_names))
        self.status_label.setText(f"已載入 {unique_faces} 個人的 {total_encodings} 個面部數據")
        logger.info(f"已載入 {unique_faces} 個人的 {total_encodings} 個面部數據")
            
    def show_person_images(self, item):
        """顯示某人的所有訓練圖像"""
        name = item.text()
        person_dir = os.path.join(self.data_dir, name)
        
        if not os.path.exists(person_dir):
            QMessageBox.information(self, "信息", f"未找到 {name} 的訓練圖像")
            return
            
        # 獲取所有圖像文件
        image_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            QMessageBox.information(self, "信息", f"{name} 沒有訓練圖像")
            return
            
        # 創建並顯示圖像查看器對話框
        viewer = ImageViewer(image_files, name, self)
        viewer.exec()
        
    def delete_selected_face(self):
        """刪除選中的人臉"""
        selected_items = self.faces_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "請先選擇要刪除的人臉！")
            return
            
        name = selected_items[0].text()
        reply = QMessageBox.question(self, "確認刪除", 
                                    f"確定要刪除 {name} 的所有訓練數據嗎？", 
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 刪除編碼文件
            encodings_dir = os.path.join(self.data_dir, "encodings")
            pkl_file = os.path.join(encodings_dir, f"{name}.pkl")
            np_file = os.path.join(encodings_dir, f"{name}.npy")
            
            if os.path.exists(pkl_file):
                os.remove(pkl_file)
            if os.path.exists(np_file):
                os.remove(np_file)
                
            # 刪除訓練圖像目錄
            person_dir = os.path.join(self.data_dir, name)
            if os.path.exists(person_dir):
                for file in os.listdir(person_dir):
                    os.remove(os.path.join(person_dir, file))
                os.rmdir(person_dir)
                
            # 從列表中移除
            row = self.faces_list.row(selected_items[0])
            self.faces_list.takeItem(row)
            
            # 從內存中移除
            if name in self.names_to_encodings:
                del self.names_to_encodings[name]
                
            # 更新編碼列表
            self.known_face_encodings = []
            self.known_face_names = []
            for n, encodings in self.names_to_encodings.items():
                for encoding in encodings:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(n)
            
            # 更新識別線程的已知人臉
            if hasattr(self, 'recognition_thread'):
                self.recognition_thread.set_known_faces(
                    np.array(self.known_face_encodings), 
                    self.known_face_names
                )
                    
            self.status_label.setText(f"已刪除 {name} 的訓練數據")
            
    def delete_all_faces(self):
        """刪除所有人臉數據"""
        if self.faces_list.count() == 0:
            return
            
        reply = QMessageBox.question(self, "確認刪除", 
                                    "確定要刪除所有人臉訓練數據嗎？", 
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 刪除編碼目錄
            encodings_dir = os.path.join(self.data_dir, "encodings")
            if os.path.exists(encodings_dir):
                for file in os.listdir(encodings_dir):
                    os.remove(os.path.join(encodings_dir, file))
                    
            # 刪除所有人臉圖像目錄
            for name in [self.faces_list.item(i).text() for i in range(self.faces_list.count())]:
                person_dir = os.path.join(self.data_dir, name)
                if os.path.exists(person_dir):
                    for file in os.listdir(person_dir):
                        os.remove(os.path.join(person_dir, file))
                    os.rmdir(person_dir)
                    
            # 清空列表
            self.faces_list.clear()
            self.known_face_encodings = []
            self.known_face_names = []
            self.names_to_encodings = {}
            
            # 更新識別線程的已知人臉
            if hasattr(self, 'recognition_thread'):
                self.recognition_thread.set_known_faces(np.array([]), [])
            
            self.status_label.setText("已刪除所有人臉數據")
            
    def toggle_recognition(self):
        """切換人臉識別狀態"""
        if self.recognize_btn.isChecked():
            if not self.known_face_encodings:
                QMessageBox.warning(self, "警告", "沒有訓練數據，請先拍照訓練！")
                self.recognize_btn.setChecked(False)
                return
                
            self.recognize_btn.setText("停止識別")
            self.camera_widget.recognition_active = True  # 設置攝像頭的識別模式標誌
            self.recognition_timer.start(300)  # 降低頻率，每300毫秒進行一次識別
            self.recognition_thread.start()
            self.status_label.setText("正在進行人臉識別...")
        else:
            self.recognize_btn.setText("開始識別")
            self.camera_widget.recognition_active = False  # 關閉攝像頭的識別模式標誌
            # 清除上一次的識別結果
            self.camera_widget.last_face_locations = []
            self.camera_widget.last_names = []
            self.camera_widget.last_confidences = []
            self.recognition_timer.stop()
            self.recognition_thread.stop()
            self.status_label.setText("識別已停止")
            
    def feed_recognition_thread(self):
        """將當前幀發送到識別線程"""
        # 增加計數器以實現跳幀
        self.frame_counter += 1
        if self.frame_counter <= self.skip_frames:
            return
            
        self.frame_counter = 0
        frame = self.camera_widget.get_current_frame()
        if frame is not None:
            self.recognition_thread.set_frame(frame)
            
    def update_recognition_results(self, frame, face_locations, names, confidences):
        """更新識別結果"""
        try:
            # 標記正在顯示結果，避免與相機更新衝突
            self.camera_widget.displaying_results = True
            
            # 更新攝像頭控件中存儲的最後識別結果
            self.camera_widget.last_face_locations = face_locations
            self.camera_widget.last_names = names
            self.camera_widget.last_confidences = confidences
            
            # 在圖像上繪製識別結果
            result_frame = frame.copy()
            
            for (top, right, bottom, left), name, confidence in zip(face_locations, names, confidences):
                # 顯示名稱和置信度
                label = f"{name} ({confidence:.2f})"
                
                # 根據置信度決定顏色 (綠色=高置信度，紅色=低置信度)
                if name != "未知":
                    # 從紅色到綠色的漸變
                    green = min(255, int(confidence * 255))
                    red = min(255, int((1 - confidence) * 255))
                    color = (red, green, 0)  # RGB
                else:
                    color = (255, 0, 0)  # 紅色表示未知
                
                # 繪製人臉框和名稱
                cv2.rectangle(result_frame, (left, top), (right, bottom), color, 2)
                
                # 添加底部填充
                cv2.rectangle(result_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # 繪製文本 (使用白色)
                text_color = (255, 255, 255)  # 白色
                cv2.putText(result_frame, label, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, text_color, 1)
            
            # 儲存最後一幀結果（可選，用於高級功能）
            self.camera_widget.last_recognition_frame = result_frame.copy()
            
            # 將結果顯示在UI上
            height, width, channels = result_frame.shape
            bytes_per_line = channels * width
            q_image = QImage(result_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            self.camera_widget.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.camera_widget.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        finally:
            # 顯示完成後重置標誌，允許攝像頭更新
            self.camera_widget.displaying_results = False
            
    def closeEvent(self, event):
        """關閉事件，停止線程"""
        if hasattr(self, 'recognition_thread'):
            self.recognition_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    
    add_attendance_tab(window)
    window.show()
    sys.exit(app.exec())