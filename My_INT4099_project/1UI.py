import sys
import os
import json
import subprocess
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                              QDialog, QLineEdit, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QFont, QColor, QPainter, QBrush


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
        self.setFixedSize(220, 280)
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


class SettingsDialog(QDialog):
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.setup_ui()
    
    def setup_ui(self):
        self.setWindowTitle("路徑設置")
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
        focus_label = QLabel("專注力監測路徑:")
        focus_label.setFixedWidth(120)
        self.focus_entry = QLineEdit(self.config.get("focus_monitoring", ""))
        focus_button = QPushButton("瀏覽...")
        focus_button.setFixedWidth(80)
        focus_button.clicked.connect(lambda: self.browse_file("focus_monitoring"))
        
        focus_layout.addWidget(focus_label)
        focus_layout.addWidget(self.focus_entry)
        focus_layout.addWidget(focus_button)
        layout.addLayout(focus_layout)
        
        # Posture monitoring path
        posture_layout = QHBoxLayout()
        posture_label = QLabel("坐姿監測路徑:")
        posture_label.setFixedWidth(120)
        self.posture_entry = QLineEdit(self.config.get("posture_monitoring", ""))
        posture_button = QPushButton("瀏覽...")
        posture_button.setFixedWidth(80)
        posture_button.clicked.connect(lambda: self.browse_file("posture_monitoring"))
        
        posture_layout.addWidget(posture_label)
        posture_layout.addWidget(self.posture_entry)
        posture_layout.addWidget(posture_button)
        layout.addLayout(posture_layout)
        
        # Note
        note_label = QLabel("請選擇相應Python程式的路徑。路徑應指向.py文件。")
        note_label.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(note_label)
        
        # Spacer
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
    
    def browse_file(self, key):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            f"選擇{key.split('_')[0]}監測程式",
            "",
            "Python文件 (*.py);;所有文件 (*.*)"
        )
        
        if filepath:
            if key == "focus_monitoring":
                self.focus_entry.setText(filepath)
            elif key == "posture_monitoring":
                self.posture_entry.setText(filepath)
    
    def get_paths(self):
        return {
            "focus_monitoring": self.focus_entry.text(),
            "posture_monitoring": self.posture_entry.text()
        }


class MonitoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize configuration
        self.config_file = "monitoring_config.json"
        self.paths = {
            "focus_monitoring": "1concentration.py",
            "posture_monitoring": "1posture_monitor.py"
        }
        self.load_config()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Window setup
        self.setWindowTitle("健康監測系統")
        self.setMinimumSize(860, 600)
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
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Title
        title_label = QLabel("健康監測系統")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("請選擇您想啟動的監測功能")
        desc_label.setObjectName("descLabel")
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # Function cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(30)
        cards_layout.setAlignment(Qt.AlignCenter)
        
        # Focus card
        self.focus_card = FunctionCard(
            "專注力監測",
            "監測並提高您的專注力",
            "icons/focus_icon.png",
            "#4a7abc",
            "focus",
            self.start_focus_monitoring
        )
        cards_layout.addWidget(self.focus_card)
        
        # Posture card
        self.posture_card = FunctionCard(
            "坐姿監測",
            "監測並改善您的坐姿",
            "icons/posture_icon.png",
            "#e67e22",
            "posture",
            self.start_posture_monitoring
        )
        cards_layout.addWidget(self.posture_card)
        
        main_layout.addLayout(cards_layout)
        
        # Settings button
        settings_button = QPushButton("設置路徑")
        settings_button.setObjectName("settingsButton")
        settings_button.setFixedWidth(150)
        settings_button.setCursor(Qt.PointingHandCursor)
        settings_button.clicked.connect(self.open_settings)
        
        settings_layout = QHBoxLayout()
        settings_layout.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(settings_button)
        main_layout.addLayout(settings_layout)
        
        # Path display
        path_layout = QVBoxLayout()
        path_layout.setSpacing(5)
        
        self.focus_path_label = QLabel(f"專注力監測路徑: {self.get_path_display('focus_monitoring')}")
        self.focus_path_label.setObjectName("pathLabel")
        path_layout.addWidget(self.focus_path_label)
        
        self.posture_path_label = QLabel(f"坐姿監測路徑: {self.get_path_display('posture_monitoring')}")
        self.posture_path_label.setObjectName("pathLabel")
        path_layout.addWidget(self.posture_path_label)
        
        main_layout.addLayout(path_layout)
        
        # Add spacer
        main_layout.addStretch()
        
        # Footer
        footer_label = QLabel("© 2025 健康監測系統 - 保持健康的工作習慣")
        footer_label.setObjectName("footerLabel")
        footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer_label)
    
    def load_config(self):
        """Load path settings from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.paths = json.load(f)
        except Exception as e:
            QMessageBox.warning(
                self,
                "警告",
                f"無法讀取配置文件: {str(e)}"
            )
    
    def save_config(self):
        """Save path settings to config file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.paths, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(
                self,
                "警告",
                f"無法保存配置文件: {str(e)}"
            )
    
    def get_path_display(self, key):
        """Get display text for path"""
        path = self.paths.get(key, "")
        if not path:
            return "未設置"
        
        # If path is too long, display truncated version
        if len(path) > 40:
            return f"...{path[-37:]}"
        return path
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.paths)
        result = dialog.exec()
        
        if result == QDialog.Accepted:
            self.paths = dialog.get_paths()
            self.save_config()
            
            # Update path labels
            self.focus_path_label.setText(f"專注力監測路徑: {self.get_path_display('focus_monitoring')}")
            self.posture_path_label.setText(f"坐姿監測路徑: {self.get_path_display('posture_monitoring')}")
            
            QMessageBox.information(
                self,
                "成功",
                "路徑設置已保存"
            )
    
    def start_focus_monitoring(self):
        """Start focus monitoring function"""
        path = self.paths.get("focus_monitoring", "")
        
        if not path:
            QMessageBox.warning(
                self,
                "警告",
                "尚未設置專注力監測程式路徑，請先設置路徑"
            )
            self.open_settings()
            return
        
        if not os.path.exists(path):
            QMessageBox.critical(
                self,
                "錯誤",
                f"找不到路徑: {path}"
            )
            return
        
        # Show confirmation dialog
        response = QMessageBox.question(
            self,
            "確認",
            "您確定要啟動專注力監測嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            try:
                # Start program as independent process
                subprocess.Popen([sys.executable, path])
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "錯誤",
                    f"啟動失敗: {str(e)}"
                )
    
    def start_posture_monitoring(self):
        """Start posture monitoring function"""
        path = self.paths.get("posture_monitoring", "")
        
        if not path:
            QMessageBox.warning(
                self,
                "警告",
                "尚未設置坐姿監測程式路徑，請先設置路徑"
            )
            self.open_settings()
            return
        
        if not os.path.exists(path):
            QMessageBox.critical(
                self,
                "錯誤",
                f"找不到路徑: {path}"
            )
            return
        
        # Show confirmation dialog
        response = QMessageBox.question(
            self,
            "確認",
            "您確定要啟動坐姿監測嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            try:
                # Start program as independent process
                subprocess.Popen([sys.executable, path])
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "錯誤",
                    f"啟動失敗: {str(e)}"
                )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MonitoringApp()
    window.show()
    
    sys.exit(app.exec())