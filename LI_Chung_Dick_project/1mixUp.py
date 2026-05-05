import sys
import os
import json
import cv2
import threading
import time
import importlib.util
import subprocess  # Added for launching face recognition
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QDialog, QLineEdit, QMessageBox, QFrame, QTabWidget,
                               QProgressBar)
from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread, QMetaObject, Q_ARG
from PySide6.QtGui import QIcon, QPixmap, QFont, QColor, QPainter, QBrush, QImage, QPen


class VideoThread(QThread):
    """處理視訊捕獲的線程"""
    change_pixmap_signal = Signal(QImage)
    update_status_signal = Signal(str)  # 新增狀態更新信號
    
    def __init__(self, capture_function):
        super().__init__()
        self.running = True
        self.capture_function = capture_function

    def run(self):
        """運行線程的主要循環"""
        while self.running:
            result = self.capture_function()
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    # 如果返回的是元組(frame, status_text)
                    frame, status_text = result
                    # 發射狀態信號
                    self.update_status_signal.emit(status_text)
                else:
                    # 如果只返回了幀
                    frame = result
                
                # 將OpenCV的BGR格式轉換為RGB
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                # 創建QImage
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
                
            # 適當的延遲以控制幀率
            time.sleep(0.03)  # 約33 FPS
    
    def stop(self):
        """停止執行緒"""
        self.running = False
        self.wait()


class IconGenerator:
    @staticmethod
    def create_circle_icon(color, icon_type, size):
        """Create a circle icon if image files aren't available"""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circle background
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, size, size)
        
        # Draw icon specific elements
        if icon_type == "focus":
            # Draw eye shape
            margin = size // 4
            eye_size = size // 2
            
            # White outer circle
            painter.setBrush(QBrush(QColor("white")))
            painter.drawEllipse(margin, margin, eye_size, eye_size)
            
            # Pupil
            pupil_size = eye_size // 3
            pupil_margin = margin + (eye_size - pupil_size) // 2
            painter.setBrush(QBrush(QColor("#333333")))
            painter.drawEllipse(pupil_margin, pupil_margin, pupil_size, pupil_size)
            
        elif icon_type == "posture":
            # Draw simple human outline
            margin = size // 6
            width = size - 2 * margin
            
            # Head
            head_size = width // 3
            head_y = margin
            head_x = margin + width // 2 - head_size // 2
            painter.setBrush(QBrush(QColor("white")))
            painter.drawEllipse(head_x, head_y, head_size, head_size)
            
            # Body
            body_top = head_y + head_size
            body_height = size - head_size - 2 * margin
            painter.drawRect(margin + width // 3, body_top, 
                            width // 3, body_height)
        
        elif icon_type == "face":
            # Draw face outline
            margin = size // 6
            face_size = size - 2 * margin
            
            # Draw face circle
            painter.setBrush(QBrush(QColor("white")))
            painter.drawEllipse(margin, margin, face_size, face_size)
            
            # Draw eyes
            eye_size = face_size // 5
            eye_y = margin + face_size // 3
            left_eye_x = margin + face_size // 3 - eye_size // 2
            right_eye_x = margin + 2 * face_size // 3 - eye_size // 2
            
            painter.setBrush(QBrush(QColor("#333333")))
            painter.drawEllipse(left_eye_x, eye_y, eye_size, eye_size)
            painter.drawEllipse(right_eye_x, eye_y, eye_size, eye_size)
            
            # Draw smile
            smile_width = face_size // 2
            smile_y = margin + 2 * face_size // 3
            smile_x = margin + face_size // 2 - smile_width // 2
            
            painter.setPen(QPen(QColor("#333333"), size // 20))
            painter.drawArc(smile_x, smile_y, smile_width, smile_width // 2, 0, 180 * 16)
            painter.setPen(Qt.NoPen)
        
        painter.end()
        return pixmap


class FunctionCard(QFrame):
    def __init__(self, title, description, icon_path, icon_color, icon_type, callback):
        super().__init__()
        self.title = title
        self.description = description
        self.icon_path = icon_path
        self.icon_color = icon_color
        self.icon_type = icon_type
        self.callback = callback
        
        self.setup_ui()
    
    def setup_ui(self):
        # Set frame style
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(280, 340)
        self.setObjectName("functionCard")
        self.setStyleSheet("""
            #functionCard {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
            #functionCard:hover {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
            }
            QLabel {
                background-color: transparent;
                color: #333333;
            }
            QPushButton {
                background-color: #4a7abc;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a5a8c;
            }
            QPushButton:pressed {
                background-color: #2a4a7c;
            }
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)
        
        # Icon
        icon_label = QLabel()
        if os.path.exists(self.icon_path):
            pixmap = QPixmap(self.icon_path)
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            # Generate icon if file doesn't exist
            pixmap = IconGenerator.create_circle_icon(self.icon_color, self.icon_type, 100)
        
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(QFont("Arial", 12))
        desc_label.setStyleSheet("color: #444444;")  # Ensure text is dark and visible
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        
        # Button
        button = QPushButton(self.title)
        button.setFixedHeight(40)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(self.callback)
        layout.addWidget(button)


class VideoDisplayWidget(QWidget):
    """顯示視頻流的小部件"""
    def __init__(self, title):
        super().__init__()
        self.title = title
        
        # 創建佈局
        self.layout = QVBoxLayout(self)
        
        # 標題
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        # 視頻顯示標籤
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0; color: black;")
        self.layout.addWidget(self.video_label)
        
        # 狀態訊息
        self.status_label = QLabel("Waiting to start...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: blue;") 
        self.layout.addWidget(self.status_label)
        
        # 控制按鈕
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addStretch()
        
        self.layout.addLayout(self.button_layout)
        
        # 視頻線程
        self.video_thread = None
    
    @Slot(QImage)
    def update_image(self, qt_image):
        """更新視頻顯示"""
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    @Slot(str)
    def update_status(self, status_text):
        """更新狀態訊息"""
        self.status_label.setText(status_text)


class FocusMonitorWidget(VideoDisplayWidget):
    """專注力監測小部件"""
    def __init__(self, focus_module_path):
        super().__init__("Focus Monitoring")
        self.focus_module_path = focus_module_path
        self.focus_monitor = None
        
        # 連接按鈕事件
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
    
    def start_monitoring(self):
        """開始專注力監測"""
        if not os.path.exists(self.focus_module_path):
            QMessageBox.critical(self, "Error", f"Focus monitoring module not found: {self.focus_module_path}")
            return
        
        try:
            # 動態導入專注力監測模塊
            spec = importlib.util.spec_from_file_location("focus_module", self.focus_module_path)
            focus_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(focus_module)
            
            # 創建專注力監測實例
            self.focus_monitor = focus_module.FocusMonitor()
            
            # 創建捕獲函數，現在它將返回元組 (frame, status_text)
            def capture_function():
                """專注力監測捕獲函數"""
                ret, frame = self.focus_monitor.cap.read()
                if not ret:
                    return None
                
                # 鏡像翻轉
                frame = cv2.flip(frame, 1)
                
                # 檢測並獲取專注力數據
                frame, is_focused, focus_score = self.focus_monitor.detect_focus(frame)
                
                # 視覺化
                frame = self.focus_monitor.visualize(frame, is_focused, focus_score)
                
                # 創建狀態文本，將作為返回值的一部分
                status = "Focused" if is_focused else "Distracted"
                status_text = f"Status: {status} (Score: {focus_score:.2f})"
                
                # 返回包含幀和狀態的元組
                return (frame, status_text)
            
            # 創建並啟動視頻線程
            self.video_thread = VideoThread(capture_function)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.update_status_signal.connect(self.update_status)  # 連接狀態更新信號
            self.video_thread.start()
            
            # 更新按鈕狀態
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start focus monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """停止專注力監測"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        if self.focus_monitor:
            # 釋放資源
            self.focus_monitor.cap.release()
            self.focus_monitor.face_mesh.close()
            self.focus_monitor.hands.close()
            self.focus_monitor = None
        
        # 重置視頻標籤
        self.video_label.clear()
        self.video_label.setText("Video stopped")
        
        # 更新狀態訊息
        self.status_label.setText("Monitoring stopped")
        
        # 更新按鈕狀態
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


class PostureMonitorWidget(VideoDisplayWidget):
    """坐姿监测小部件"""
    def __init__(self, posture_module_path):
        super().__init__("Posture Monitoring")
        self.posture_module_path = posture_module_path
        self.posture_monitor = None
        
        # 连接按钮事件
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
    
    def start_monitoring(self):
        """开始坐姿监测"""
        if not os.path.exists(self.posture_module_path):
            QMessageBox.critical(self, "Error", f"Posture monitoring module not found: {self.posture_module_path}")
            return
        
        try:
            # 动态导入坐姿监测模块
            spec = importlib.util.spec_from_file_location("posture_module", self.posture_module_path)
            posture_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(posture_module)
            
            # 创建坐姿监测实例
            self.posture_monitor = posture_module.PostureMonitor()
            self.posture_monitor.initialize()
            
            # 创建捕获函数
            def capture_function():
                """坐姿监测捕获函数"""
                ret, frame = self.posture_monitor.cap.read()
                if not ret:
                    return None
                
                # 镜像翻转
                frame = cv2.flip(frame, 1)
                
                # 检测并分析坐姿
                frame, posture_result = self.posture_monitor.detect_posture(frame)
                
                # 如果检测到姿势，则创建状态文本
                if posture_result:
                    quality = posture_result.get("posture_quality", "unknown")
                    neck_incl = posture_result.get("neck_inclination", 0)
                    torso_incl = posture_result.get("torso_inclination", 0)
                    status_text = f"Posture Quality: {quality.upper()} - Neck Inclination: {neck_incl:.1f}° - Torso Inclination: {torso_incl:.1f}°"
                else:
                    status_text = "No posture detected"
                
                # 返回帧和状态文本
                return (frame, status_text)
            
            # 创建并启动视频线程
            self.video_thread = VideoThread(capture_function)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.update_status_signal.connect(self.update_status)  # 连接状态更新信号
            self.video_thread.start()
            
            # 更新按钮状态
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start posture monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """停止坐姿监测"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        if self.posture_monitor:
            # 释放资源
            self.posture_monitor.release()
            self.posture_monitor = None
        
        # 重置视频标签
        self.video_label.clear()
        self.video_label.setText("Video stopped")
        
        # 更新状态信息
        self.status_label.setText("Monitoring stopped")
        
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("Path Settings")
        self.setFixedSize(600, 300)
        self.setStyleSheet("""
            QDialog {
                background-color: #f0f0f0;
            }
            QLabel {
                font-size: 12px;
                color: #333333;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4a7abc;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a5a8c;
            }
            QPushButton:pressed {
                background-color: #2a4a7c;
            }
        """)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Focus monitoring path
        focus_layout = QHBoxLayout()
        focus_label = QLabel("Focus Monitoring Path:")
        focus_label.setFixedWidth(120)
        self.focus_entry = QLineEdit(self.config.get("focus_monitoring", ""))
        focus_button = QPushButton("Browse...")
        focus_button.setFixedWidth(80)
        focus_button.clicked.connect(lambda: self.browse_file("focus_monitoring"))
        
        focus_layout.addWidget(focus_label)
        focus_layout.addWidget(self.focus_entry)
        focus_layout.addWidget(focus_button)
        layout.addLayout(focus_layout)
        
        # Posture monitoring path
        posture_layout = QHBoxLayout()
        posture_label = QLabel("Posture Monitoring Path:")
        posture_label.setFixedWidth(120)
        self.posture_entry = QLineEdit(self.config.get("posture_monitoring", ""))
        posture_button = QPushButton("Browse...")
        posture_button.setFixedWidth(80)
        posture_button.clicked.connect(lambda: self.browse_file("posture_monitoring"))
        
        posture_layout.addWidget(posture_label)
        posture_layout.addWidget(self.posture_entry)
        posture_layout.addWidget(posture_button)
        layout.addLayout(posture_layout)
        
        # Face recognition path
        face_layout = QHBoxLayout()
        face_label = QLabel("Face Recognition Path:")
        face_label.setFixedWidth(120)
        self.face_entry = QLineEdit(self.config.get("face_recognition", ""))
        face_button = QPushButton("Browse...")
        face_button.setFixedWidth(80)
        face_button.clicked.connect(lambda: self.browse_file("face_recognition"))
        
        face_layout.addWidget(face_label)
        face_layout.addWidget(self.face_entry)
        face_layout.addWidget(face_button)
        layout.addLayout(face_layout)
        
        # Note
        note_label = QLabel("Please select the path to the corresponding Python program. The path should point to a .py file.")
        note_label.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(note_label)
        
        # Spacer
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
    
    def browse_file(self, key):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {key.split('_')[0]} monitoring program",
            "",
            "Python files (*.py);;All files (*.*)"
        )
        
        if filepath:
            if key == "focus_monitoring":
                self.focus_entry.setText(filepath)
            elif key == "posture_monitoring":
                self.posture_entry.setText(filepath)
            elif key == "face_recognition":
                self.face_entry.setText(filepath)
    
    def get_paths(self):
        return {
            "focus_monitoring": self.focus_entry.text(),
            "posture_monitoring": self.posture_entry.text(),
            "face_recognition": self.face_entry.text()
        }


class MonitoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize configuration
        self.config_file = "monitoring_config.json"
        self.paths = {
            "focus_monitoring": "1concentration.py",
            "posture_monitoring": "1posture_monitor.py",
            "face_recognition": "1face_recognition.py"  # Added default path for face recognition
        }
        self.load_config()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Window setup
        self.setWindowTitle("Health Monitoring System")
        self.setMinimumSize(1200, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QLabel#titleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333333;
            }
            QLabel#descLabel {
                font-size: 16px;
                color: #666666;
            }
            QLabel#pathLabel {
                font-size: 11px;
                color: #888888;
            }
            QLabel#footerLabel {
                font-size: 11px;
                color: #999999;
            }
            QPushButton#settingsButton {
                background-color: #f0f0f0;
                border: 1px solid #dddddd;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 12px;
                color: #555555;
            }
            QPushButton#settingsButton:hover {
                background-color: #e6e6e6;
                border: 1px solid #cccccc;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-bottom-color: #cccccc;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 8px 15px;
                margin-right: 2px;
                color: #666666;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                color: #333333;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e0e0e0;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Health Monitoring System")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Integrated Focus, Posture and Face Recognition")
        desc_label.setObjectName("descLabel")
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # 選項卡部件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 主頁選項卡
        self.home_tab = QWidget()
        self.tab_widget.addTab(self.home_tab, "Home")
        
        # 主頁佈局
        home_layout = QVBoxLayout(self.home_tab)
        home_layout.setContentsMargins(20, 20, 20, 20)
        home_layout.setSpacing(20)
        
        # Function cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(30)
        cards_layout.setAlignment(Qt.AlignCenter)
        
        # Focus card
        self.focus_card = FunctionCard(
            "Focus Monitoring",
            "Monitor and improve your focus",
            "icons/focus_icon.png",
            "#4a7abc",
            "focus",
            lambda: self.tab_widget.setCurrentIndex(1)  # 切換到專注力選項卡
        )
        cards_layout.addWidget(self.focus_card)
        
        # Posture card
        self.posture_card = FunctionCard(
            "Posture Monitoring",
            "Monitor and improve your posture",
            "icons/posture_icon.png",
            "#e67e22",
            "posture",
            lambda: self.tab_widget.setCurrentIndex(2)  # 切換到坐姿選項卡
        )
        cards_layout.addWidget(self.posture_card)
        
        # Face Recognition card
        self.face_card = FunctionCard(
            "Face Attendance",
            "Recognize and identify faces",
            "icons/face_icon.png",
            "#27ae60",  # Green color
            "face",
            self.run_face_recognition  # Call function to run face recognition
        )
        cards_layout.addWidget(self.face_card)
        
        home_layout.addLayout(cards_layout)
        
        # Path display
        path_layout = QVBoxLayout()
        path_layout.setSpacing(5)
        
        self.focus_path_label = QLabel(f"Focus Monitoring Path: {self.get_path_display('focus_monitoring')}")
        self.focus_path_label.setObjectName("pathLabel")
        path_layout.addWidget(self.focus_path_label)
        
        self.posture_path_label = QLabel(f"Posture Monitoring Path: {self.get_path_display('posture_monitoring')}")
        self.posture_path_label.setObjectName("pathLabel")
        path_layout.addWidget(self.posture_path_label)
        
        self.face_path_label = QLabel(f"Face Recognition Path: {self.get_path_display('face_recognition')}")
        self.face_path_label.setObjectName("pathLabel")
        path_layout.addWidget(self.face_path_label)
        
        home_layout.addLayout(path_layout)
        
        # Settings button
        settings_button = QPushButton("Configure Paths")
        settings_button.setObjectName("settingsButton")
        settings_button.setFixedWidth(150)
        settings_button.setCursor(Qt.PointingHandCursor)
        settings_button.clicked.connect(self.open_settings)
        
        settings_layout = QHBoxLayout()
        settings_layout.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(settings_button)
        home_layout.addLayout(settings_layout)
        
        # 添加專注力監測選項卡
        self.focus_tab = FocusMonitorWidget(self.paths["focus_monitoring"])
        self.tab_widget.addTab(self.focus_tab, "Focus Monitoring")
        
        # 添加坐姿監測選項卡
        self.posture_tab = PostureMonitorWidget(self.paths["posture_monitoring"])
        self.tab_widget.addTab(self.posture_tab, "Posture Monitoring")
        
        # Footer
        footer_label = QLabel("© 2025 Health Monitoring System - Maintain Healthy Work Habits")
        footer_label.setObjectName("footerLabel")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
    
    def run_face_recognition(self):
        """Launch the face recognition application"""
        try:
            # Check if file exists
            if not os.path.exists(self.paths["face_recognition"]):
                QMessageBox.critical(self, "Error", f"Face recognition module not found: {self.paths['face_recognition']}")
                return
                
            # Run the face recognition module as a separate process
            subprocess.Popen([sys.executable, self.paths["face_recognition"]])
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch face recognition: {str(e)}")
    
    def load_config(self):
        """Load path settings from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.paths = json.load(f)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Warning",
                f"Unable to read configuration file: {str(e)}"
            )
    
    def save_config(self):
        """Save path settings to config file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.paths, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Warning",
                f"Unable to save configuration file: {str(e)}"
            )
    
    def get_path_display(self, key):
        """Get display text for path"""
        path = self.paths.get(key, "")
        if not path:
            return "Not set"
        
        # If path is too long, display truncated version
        if len(path) > 40:
            return f"...{path[-37:]}"
        return path
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.paths)
        result = dialog.exec()
        
        if result == QDialog.Accepted:
            old_paths = self.paths.copy()
            self.paths = dialog.get_paths()
            self.save_config()
            
            # Update path labels
            self.focus_path_label.setText(f"Focus Monitoring Path: {self.get_path_display('focus_monitoring')}")
            self.posture_path_label.setText(f"Posture Monitoring Path: {self.get_path_display('posture_monitoring')}")
            self.face_path_label.setText(f"Face Recognition Path: {self.get_path_display('face_recognition')}")
            
            # 如果路徑變更，更新監測選項卡
            if old_paths["focus_monitoring"] != self.paths["focus_monitoring"]:
                # 先停止現有的監測
                self.focus_tab.stop_monitoring()
                # 重新創建選項卡
                self.tab_widget.removeTab(1)
                self.focus_tab = FocusMonitorWidget(self.paths["focus_monitoring"])
                self.tab_widget.insertTab(1, self.focus_tab, "Focus Monitoring")
            
            if old_paths["posture_monitoring"] != self.paths["posture_monitoring"]:
                # 先停止現有的監測
                self.posture_tab.stop_monitoring()
                # 重新創建選項卡
                self.tab_widget.removeTab(2)
                self.posture_tab = PostureMonitorWidget(self.paths["posture_monitoring"])
                self.tab_widget.insertTab(2, self.posture_tab, "Posture Monitoring")
            
            QMessageBox.information(
                self,
                "Success",
                "Path settings saved"
            )
    
    def closeEvent(self, event):
        """在關閉應用程式前釋放資源"""
        # 停止所有監測
        if hasattr(self, 'focus_tab'):
            self.focus_tab.stop_monitoring()
        
        if hasattr(self, 'posture_tab'):
            self.posture_tab.stop_monitoring()
        
        # 接受關閉事件
        event.accept()


# 應用程式入口點，放在所有類定義之外
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MonitoringApp()
    main_window.show()
    sys.exit(app.exec())