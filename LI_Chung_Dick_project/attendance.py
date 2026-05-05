import os
import csv
import pandas as pd
from datetime import datetime, date, timedelta
from PySide6.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QTableWidget, QTableWidgetItem, QComboBox, 
                              QDateEdit, QFileDialog, QMessageBox, QHeaderView,
                              QFrame, QSplitter, QApplication, QStyle)
from PySide6.QtCore import Qt, QDate, QSize, QTimer, QPoint, QPropertyAnimation, QEasingCurve, QRect
from PySide6.QtGui import QFont, QColor, QIcon, QBrush, QLinearGradient, QPalette, QCursor, QPainter, QPixmap, QTransform


class AttendanceRecord:
    """考勤記錄類"""
    
    def __init__(self, name, check_in_time=None):
        self.name = name
        self.check_in_time = check_in_time or datetime.now()
        self.date = self.check_in_time.date()
        
    def to_dict(self):
        """轉換為字典格式"""
        return {
            'name': self.name,
            'date': self.date.strftime('%Y-%m-%d'),
            'check_in_time': self.check_in_time.strftime('%H:%M:%S'),
        }


class AttendanceTracker:
    """考勤跟踪器類"""
    
    def __init__(self, attendance_dir="attendance_records"):
        self.attendance_dir = attendance_dir
        os.makedirs(attendance_dir, exist_ok=True)
        
        # 當前日期的考勤記錄
        self.today_records = {}
        
        # 今天的日期
        self.current_date = date.today()
        
        # 最小簽到間隔(秒)
        self.min_checkin_interval = 300  # 5分鐘
        
        # 載入今天的記錄
        self.load_today_records()
        
    def load_today_records(self):
        """載入今天的考勤記錄"""
        self.today_records = {}
        self.current_date = date.today()
        
        filename = os.path.join(self.attendance_dir, f"{self.current_date.strftime('%Y-%m-%d')}.csv")
        if not os.path.exists(filename):
            return
            
        try:
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                name = row['name']
                
                # 解析時間
                check_in_time = datetime.strptime(
                    f"{row['date']} {row['check_in_time']}", 
                    '%Y-%m-%d %H:%M:%S'
                )
                    
                record = AttendanceRecord(name, check_in_time)
                self.today_records[name] = record
                
        except Exception as e:
            print(f"載入今天的考勤記錄失敗: {e}")
            
    def record_attendance(self, name, confidence):
        """記錄出勤
        
        Returns:
            tuple: (是否是新記錄, 記錄類型 (check_in))
        """
        current_time = datetime.now()
        
        # 檢查是否已有今天的記錄
        if name in self.today_records:
            record = self.today_records[name]
            
            # 與上次簽到時間太近，不處理
            time_diff = (current_time - record.check_in_time).total_seconds()
            if time_diff < self.min_checkin_interval:
                return (False, None)
                
            # 已有記錄但超過最小間隔，創建新記錄（多次簽到）
            record = AttendanceRecord(name)
            self.today_records[name] = record
            self.save_records()
            return (True, "check_in")
        else:
            # 新的簽到記錄
            record = AttendanceRecord(name)
            self.today_records[name] = record
            self.save_records()
            return (True, "check_in")
            
    def save_records(self):
        """保存考勤記錄到CSV"""
        filename = os.path.join(self.attendance_dir, f"{self.current_date.strftime('%Y-%m-%d')}.csv")
        
        # 將記錄轉換為DataFrame
        records_data = [record.to_dict() for record in self.today_records.values()]
        df = pd.DataFrame(records_data)
        
        # 保存到CSV
        df.to_csv(filename, index=False)
        
    def get_today_records(self):
        """獲取今天的考勤記錄"""
        return list(self.today_records.values())
        
    def get_records_by_date(self, selected_date):
        """獲取指定日期的考勤記錄"""
        date_str = selected_date.strftime('%Y-%m-%d')
        filename = os.path.join(self.attendance_dir, f"{date_str}.csv")
        
        if not os.path.exists(filename):
            return []
            
        try:
            df = pd.read_csv(filename)
            records = []
            
            for _, row in df.iterrows():
                name = row['name']
                
                # 解析時間
                check_in_time = datetime.strptime(
                    f"{row['date']} {row['check_in_time']}", 
                    '%Y-%m-%d %H:%M:%S'
                )
                    
                record = AttendanceRecord(name, check_in_time)
                records.append(record)
                
            return records
            
        except Exception as e:
            print(f"載入{date_str}的考勤記錄失敗: {e}")
            return []
            
    def get_records_by_period(self, start_date, end_date):
        """獲取一段時間內的考勤記錄"""
        all_records = []
        
        current = start_date
        while current <= end_date:
            records = self.get_records_by_date(current)
            all_records.extend(records)
            current += timedelta(days=1)
            
        return all_records
        
    def export_attendance(self, filename, start_date, end_date):
        """導出考勤記錄"""
        records = self.get_records_by_period(start_date, end_date)
        
        if not records:
            return False
            
        # 將記錄轉換為DataFrame
        records_data = [record.to_dict() for record in records]
        df = pd.DataFrame(records_data)
        
        # 保存到CSV或Excel
        try:
            if filename.endswith('.csv'):
                df.to_csv(filename, index=False)
            elif filename.endswith('.xlsx'):
                # 檢查是否安裝了openpyxl
                try:
                    import openpyxl
                    df.to_excel(filename, index=False)
                except ImportError:
                    # 如果沒有安裝openpyxl，改為保存為CSV格式
                    print("未安裝openpyxl，無法導出Excel格式。改為保存為CSV格式。")
                    filename = filename.replace('.xlsx', '.csv')
                    df.to_csv(filename, index=False)
            else:
                return False
        except Exception as e:
            print(f"導出失敗: {e}")
            return False
            
        return True, filename  # 返回成功狀態和實際的文件名
    
    def delete_record(self, date_str, name, check_in_time_str):
        """刪除指定的考勤記錄
        
        Parameters:
            date_str (str): 日期，格式為 'YYYY-MM-DD'
            name (str): 姓名
            check_in_time_str (str): 簽到時間，格式為 'HH:MM:SS'
        
        Returns:
            bool: 是否刪除成功
        """
        # 構建文件名
        filename = os.path.join(self.attendance_dir, f"{date_str}.csv")
        
        if not os.path.exists(filename):
            return False
        
        try:
            # 讀取CSV文件
            df = pd.read_csv(filename)
            
            # 查找匹配的記錄
            mask = (df['name'] == name) & (df['check_in_time'] == check_in_time_str)
            
            # 如果沒有找到記錄
            if not mask.any():
                return False
            
            # 刪除匹配的記錄
            df = df[~mask]
            
            # 如果刪除後還有記錄，保存回文件
            if not df.empty:
                df.to_csv(filename, index=False)
            else:
                # 如果沒有記錄了，刪除文件
                os.remove(filename)
                
            # 如果刪除的是今天的記錄，更新內存中的記錄
            if date_str == self.current_date.strftime('%Y-%m-%d'):
                self.load_today_records()
                
            return True
            
        except Exception as e:
            print(f"刪除考勤記錄失敗: {e}")
            return False
    
    def delete_multiple_records(self, records_to_delete):
        """刪除多條考勤記錄
        
        Parameters:
            records_to_delete (list): 要刪除的記錄列表，每條記錄是(date_str, name, check_in_time_str)的元組
            
        Returns:
            int: 成功刪除的記錄數量
        """
        success_count = 0
        
        for date_str, name, check_in_time_str in records_to_delete:
            if self.delete_record(date_str, name, check_in_time_str):
                success_count += 1
                
        return success_count


# 自定義按鈕類，添加懸停效果
class HoverButton(QPushButton):
    def __init__(self, text, parent=None, primary=False, danger=False):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.primary = primary
        self.danger = danger
        self.hovered = False
        self.setFixedHeight(36)
        self.setFont(QFont("Microsoft YaHei", 10))
        
    def enterEvent(self, event):
        self.hovered = True
        self.update()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.hovered = False
        self.update()
        super().leaveEvent(event)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 設置不同類型按鈕的顏色
        if self.danger:
            base_color = QColor(220, 53, 69)
            hover_color = QColor(200, 35, 51)
            text_color = QColor(255, 255, 255)
        elif self.primary:
            base_color = QColor(65, 105, 225)  # 皇家藍
            hover_color = QColor(45, 85, 205)
            text_color = QColor(255, 255, 255)
        else:
            base_color = QColor(52, 58, 64)
            hover_color = QColor(73, 80, 87)
            text_color = QColor(255, 255, 255)
        
        # 設置按鈕顏色（根據懸停狀態）
        bg_color = hover_color if self.hovered else base_color
        
        # 繪製按鈕背景
        path = painter.drawRoundedRect(0, 0, self.width(), self.height(), 18, 18)
        painter.fillRect(0, 0, self.width(), self.height(), bg_color)
        
        # 繪製按鈕文本
        painter.setPen(text_color)
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())
        
        # 如果有圖標，繪製圖標
        if not self.icon().isNull():
            icon_rect = QRect(10, (self.height() - 16) // 2, 16, 16)
            self.icon().paint(painter, icon_rect)
            
        painter.end()


class AttendanceTabWidget(QWidget):
    """考勤標籤頁控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 創建考勤追踪器
        self.attendance_tracker = AttendanceTracker()
        
        # 是否在考勤模式
        self.is_attendance_mode = False
        
        # 上次識別到的人名
        self.last_recognized_name = None
        self.last_recognized_time = datetime.now()
        
        # 創建UI
        self.init_ui()
        
        # 設置樣式
        self.apply_styles()
        
        # 更新表格
        self.update_table()
        
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)
        
        # 標題區域
        title_frame = QFrame()
        title_frame.setObjectName("titleFrame")
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = QLabel("智能考勤管理系統")
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label)
        
        # 日期顯示
        date_label = QLabel(f"今天是: {QDate.currentDate().toString('yyyy年MM月dd日')}")
        date_label.setObjectName("dateLabel")
        title_layout.addWidget(date_label)
        
        main_layout.addWidget(title_frame)
        
        # 控制區域
        control_frame = QFrame()
        control_frame.setObjectName("controlFrame")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(20, 15, 20, 15)
        control_layout.setSpacing(15)
        
        # 開始/停止考勤按鈕
        self.toggle_attendance_btn = HoverButton("開始考勤", primary=True)
        self.toggle_attendance_btn.setCheckable(True)
        self.toggle_attendance_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.toggle_attendance_btn.clicked.connect(self.toggle_attendance_mode)
        control_layout.addWidget(self.toggle_attendance_btn)
        
        # 日期選擇
        date_frame = QFrame()
        date_layout = QHBoxLayout(date_frame)
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(8)
        
        date_label = QLabel("查看日期:")
        date_label.setObjectName("formLabel")
        date_layout.addWidget(date_label)
        
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.dateChanged.connect(self.date_changed)
        self.date_edit.setObjectName("dateEdit")
        date_layout.addWidget(self.date_edit)
        
        control_layout.addWidget(date_frame)
        
        # 刷新按鈕
        refresh_btn = HoverButton("刷新")
        refresh_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.clicked.connect(self.update_table)
        control_layout.addWidget(refresh_btn)
        
        # 導出按鈕
        export_btn = HoverButton("導出報表")
        export_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_DialogSaveButton))
        export_btn.clicked.connect(self.export_report)
        control_layout.addWidget(export_btn)
        
        # 刪除按鈕
        self.delete_btn = HoverButton("刪除記錄", danger=True)
        self.delete_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_TrashIcon))
        self.delete_btn.clicked.connect(self.delete_selected_records)
        control_layout.addWidget(self.delete_btn)
        
        main_layout.addWidget(control_frame)
        
        # 狀態標籤
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 10, 20, 10)
        
        status_icon = QLabel()
        status_icon.setPixmap(QApplication.style().standardPixmap(QStyle.SP_MessageBoxInformation))
        status_layout.addWidget(status_icon)
        
        self.status_label = QLabel("考勤系統就緒")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        main_layout.addWidget(status_frame)
        
        # 表格容器
        table_frame = QFrame()
        table_frame.setObjectName("tableFrame")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(20, 20, 20, 20)
        
        # 表格標題
        table_header = QFrame()
        table_header.setObjectName("tableHeader")
        table_header_layout = QHBoxLayout(table_header)
        table_header_layout.setContentsMargins(10, 10, 10, 10)
        
        table_title = QLabel("考勤記錄")
        table_title.setObjectName("sectionTitle")
        table_header_layout.addWidget(table_title)
        
        table_layout.addWidget(table_header)
        
        # 考勤表格
        self.attendance_table = QTableWidget()
        self.attendance_table.setObjectName("attendanceTable")
        self.attendance_table.setColumnCount(3)
        self.attendance_table.setHorizontalHeaderLabels([
            "姓名", "日期", "簽到時間"
        ])
        self.attendance_table.setAlternatingRowColors(True)
        self.attendance_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # 設置列寬自動調整
        header = self.attendance_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        table_layout.addWidget(self.attendance_table)
        
        main_layout.addWidget(table_frame, 1)  # 表格區域擴展填充
        
        # 底部狀態欄
        footer_frame = QFrame()
        footer_frame.setObjectName("footerFrame")
        footer_layout = QHBoxLayout(footer_frame)
        footer_layout.setContentsMargins(20, 10, 20, 10)
        
        footer_text = QLabel("© 2023 智能考勤管理系統")
        footer_text.setObjectName("footerText")
        footer_layout.addWidget(footer_text)
        
        main_layout.addWidget(footer_frame)
        
    def apply_styles(self):
        """應用樣式"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #E6E6E6;
                font-family: "Microsoft YaHei", Arial, sans-serif;
            }
            #titleFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A237E, stop:1 #3949AB);
                border-radius: 20px;
            }
            #titleLabel {
                font-size: 28px;
                font-weight: bold;
                color: white;
                background: transparent;
                margin-bottom: 5px;
            }
            #dateLabel {
                font-size: 14px;
                color: rgba(255, 255, 255, 0.9);
                background: transparent;
            }
            #controlFrame {
                background-color: #2D2D30;
                border-radius: 20px;
                border: 1px solid #3F3F46;
            }
            #formLabel {
                font-size: 14px;
                font-weight: bold;
                color: #CCCCCC;
                background: transparent;
            }
            #statusFrame {
                background-color: #2D2D30;
                border-radius: 15px;
                border: 1px solid #3F3F46;
            }
            #statusLabel {
                color: #CCCCCC;
                font-size: 14px;
                background: transparent;
            }
            #tableFrame {
                background-color: #2D2D30;
                border-radius: 20px;
                border: 1px solid #3F3F46;
            }
            #tableHeader {
                background-color: #252526;
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
                margin-bottom: 10px;
            }
            #sectionTitle {
                font-size: 18px;
                font-weight: bold;
                color: #E6E6E6;
                background: transparent;
            }
            #footerFrame {
                background-color: #2D2D30;
                border-radius: 15px;
                border: 1px solid #3F3F46;
            }
            #footerText {
                color: #CCCCCC;
                font-size: 13px;
                background: transparent;
            }
            QTableWidget {
                background-color: #252526;
                alternate-background-color: #2D2D30;
                gridline-color: #3F3F46;
                border: 1px solid #3F3F46;
                border-radius: 10px;
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #3949AB;
                color: white;
                padding: 10px;
                font-weight: bold;
                border: none;
                border-right: 1px solid #5C6BC0;
            }
            QHeaderView::section:first {
                border-top-left-radius: 8px;
            }
            QHeaderView::section:last {
                border-top-right-radius: 8px;
                border-right: none;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3F3F46;
            }
            QTableWidget::item:selected {
                background-color: rgba(65, 105, 225, 0.3);
                color: #FFFFFF;
            }
            QDateEdit {
                padding: 8px;
                background-color: #252526;
                color: #E6E6E6;
                border: 1px solid #3F3F46;
                border-radius: 10px;
                min-width: 120px;
            }
            QDateEdit:hover {
                border: 1px solid #4169E1;
            }
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #3F3F46;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QDateEdit::drop-down:hover {
                background-color: #333337;
            }
        """)
        
    def toggle_attendance_mode(self):
        """切換考勤模式"""
        self.is_attendance_mode = self.toggle_attendance_btn.isChecked()
        
        if self.is_attendance_mode:
            self.toggle_attendance_btn.setText("停止考勤")
            self.toggle_attendance_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaStop))
            self.status_label.setText("考勤模式已啟動，等待人臉識別...")
            
            # 確保載入的是今天的記錄
            self.attendance_tracker.load_today_records()
            
            # 更新日期為今天
            self.date_edit.setDate(QDate.currentDate())
        else:
            self.toggle_attendance_btn.setText("開始考勤")
            self.toggle_attendance_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
            self.status_label.setText("考勤模式已停止")
            
    def date_changed(self):
        """日期變更時更新表格"""
        self.update_table()
        
    def update_table(self):
        """更新考勤表格"""
        # 清空表格
        self.attendance_table.setRowCount(0)
        
        # 獲取選擇的日期
        qdate = self.date_edit.date()
        selected_date = date(qdate.year(), qdate.month(), qdate.day())
        
        # 加載記錄
        if selected_date == date.today():
            records = self.attendance_tracker.get_today_records()
        else:
            records = self.attendance_tracker.get_records_by_date(selected_date)
            
        # 填充表格
        for i, record in enumerate(records):
            self.attendance_table.insertRow(i)
            
            # 姓名
            name_item = QTableWidgetItem(record.name)
            name_item.setTextAlignment(Qt.AlignCenter)
            self.attendance_table.setItem(i, 0, name_item)
            
            # 日期
            date_str = record.date.strftime('%Y-%m-%d')
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(Qt.AlignCenter)
            self.attendance_table.setItem(i, 1, date_item)
            
            # 簽到時間
            check_in_str = record.check_in_time.strftime('%H:%M:%S')
            time_item = QTableWidgetItem(check_in_str)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.attendance_table.setItem(i, 2, time_item)
            
        # 更新表格標題數量
        records_count = self.attendance_table.rowCount()
        date_str = qdate.toString('yyyy年MM月dd日')
        table_title = self.findChild(QLabel, "sectionTitle")
        if table_title:
            table_title.setText(f"考勤記錄 - {date_str} ({records_count}人次)")
            
    def handle_recognition_result(self, name, confidence):
        """處理人臉識別結果"""
        if not self.is_attendance_mode or name == "未知" or confidence < 0.5:
            return
            
        # 檢查冷卻時間
        current_time = datetime.now()
        if (self.last_recognized_name == name and 
            (current_time - self.last_recognized_time).total_seconds() < 2):
            return
            
        # 更新最後識別時間和人名
        self.last_recognized_name = name
        self.last_recognized_time = current_time
        
        # 記錄考勤
        recorded, record_type = self.attendance_tracker.record_attendance(name, confidence)
        
        if not recorded:
            return
            
        # 更新狀態並添加動畫效果
        if record_type == "check_in":
            self.status_label.setText(f"{name} 已簽到 - {current_time.strftime('%H:%M:%S')}")
            
            # 顯示簽到成功提示
            self.show_check_in_success(name)
            
        # 更新表格
        self.update_table()
        
    def show_check_in_success(self, name):
        """顯示簽到成功提示"""
        # 創建一個消息框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("簽到成功")
        msg_box.setText(f"{name} 簽到成功！")
        msg_box.setIcon(QMessageBox.Information)
        
        # 設置樣式
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2D2D30;
                border-radius: 15px;
            }
            QLabel {
                color: #E6E6E6;
                font-size: 14px;
                padding: 10px;
                background: transparent;
            }
            QPushButton {
                background-color: #4169E1;
                color: white;
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3158D6;
            }
        """)
            
        # 自動關閉
        timer = QTimer()
        timer.singleShot(2000, msg_box.accept)
        
        # 顯示消息框
        msg_box.exec()
        
    def export_report(self):
        """導出考勤報表"""
        # 選擇時間範圍的對話框
        from PySide6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("選擇報表範圍")
        dialog.setMinimumWidth(350)
        
        # 設置對話框樣式
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2D2D30;
                border-radius: 15px;
            }
            QLabel {
                color: #E6E6E6;
                font-size: 14px;
                padding: 5px 0;
                background: transparent;
            }
            QDateEdit {
                padding: 8px;
                background-color: #252526;
                color: #E6E6E6;
                border: 1px solid #3F3F46;
                border-radius: 10px;
                min-width: 120px;
            }
            QDateEdit:hover {
                border: 1px solid #4169E1;
            }
            QPushButton {
                background-color: #4169E1;
                color: white;
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #3158D6;
            }
            QPushButton#cancel {
                background-color: #52565A;
                color: white;
            }
            QPushButton#cancel:hover {
                background-color: #636A71;
            }
        """)
        
        layout = QFormLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)
        
        # 開始日期
        start_date_edit = QDateEdit()
        start_date_edit.setDate(QDate.currentDate().addDays(-7))
        start_date_edit.setCalendarPopup(True)
        layout.addRow("開始日期:", start_date_edit)
        
        # 結束日期
        end_date_edit = QDateEdit()
        end_date_edit.setDate(QDate.currentDate())
        end_date_edit.setCalendarPopup(True)
        layout.addRow("結束日期:", end_date_edit)
        
        # 按鈕
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = button_box.button(QDialogButtonBox.Ok)
        ok_btn.setText("確定")
        cancel_btn = button_box.button(QDialogButtonBox.Cancel)
        cancel_btn.setText("取消")
        cancel_btn.setObjectName("cancel")
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # 顯示對話框
        if dialog.exec() != QDialog.Accepted:
            return
            
        # 獲取選擇的日期
        start_qdate = start_date_edit.date()
        end_qdate = end_date_edit.date()
        
        start_date = date(start_qdate.year(), start_qdate.month(), start_qdate.day())
        end_date = date(end_qdate.year(), end_qdate.month(), end_qdate.day())
        
        # 選擇保存位置，默認文件名為當天日期
        default_filename = f"考勤報表_{datetime.now().strftime('%Y%m%d')}"
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存報表", default_filename, "Excel 文件 (*.xlsx);;CSV 文件 (*.csv)"
        )
        
        if not filename:
            return
            
        # 導出報表
        result = self.attendance_tracker.export_attendance(filename, start_date, end_date)
        
        if result:
            actual_filename = result[1] if isinstance(result, tuple) else filename
            QMessageBox.information(self, "成功", f"報表已導出至 {actual_filename}")
        else:
            QMessageBox.warning(self, "錯誤", "報表導出失敗，可能沒有符合條件的記錄")
            
    def delete_selected_records(self):
        """刪除選中的考勤記錄"""
        # 獲取選中的行
        selected_rows = set(index.row() for index in self.attendance_table.selectedIndexes())
        
        if not selected_rows:
            QMessageBox.warning(self, "警告", "請先選擇要刪除的記錄")
            return
        
        # 確認是否刪除
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("確認刪除")
        msg_box.setText(f"確定要刪除選中的 {len(selected_rows)} 條記錄嗎？")
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        
        # 設置樣式
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #2D2D30;
                border-radius: 15px;
            }
            QLabel {
                color: #E6E6E6;
                font-size: 14px;
                padding: 10px;
                background: transparent;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 80px;
            }
            QPushButton[text="Yes"] {
                background-color: #DC3545;
                color: white;
            }
            QPushButton[text="Yes"]:hover {
                background-color: #C82333;
            }
            QPushButton[text="No"] {
                background-color: #52565A;
                color: white;
            }
            QPushButton[text="No"]:hover {
                background-color: #636A71;
            }
        """)
        
        reply = msg_box.exec()
        
        if reply != QMessageBox.Yes:
            return
        
        # 準備要刪除的記錄
        records_to_delete = []
        
        for row in sorted(selected_rows):
            name = self.attendance_table.item(row, 0).text()
            date_str = self.attendance_table.item(row, 1).text()
            check_in_time_str = self.attendance_table.item(row, 2).text()
            
            records_to_delete.append((date_str, name, check_in_time_str))
        
        # 執行刪除
        deleted_count = self.attendance_tracker.delete_multiple_records(records_to_delete)
        
        # 更新表格
        self.update_table()
        
        # 顯示結果
        success_msg = QMessageBox(self)
        success_msg.setWindowTitle("刪除結果")
        success_msg.setText(f"成功刪除 {deleted_count} 條記錄")
        success_msg.setIcon(QMessageBox.Information)
        success_msg.setStandardButtons(QMessageBox.Ok)
        
        # 設置樣式
        success_msg.setStyleSheet("""
            QMessageBox {
                background-color: #2D2D30;
                border-radius: 15px;
            }
            QLabel {
                color: #E6E6E6;
                font-size: 14px;
                padding: 10px;
                background: transparent;
            }
            QPushButton {
                background-color: #4169E1;
                color: white;
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3158D6;
            }
        """)
        
        success_msg.exec()


# 自定義TabWidget，添加風格化外觀
class StyledTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("styledTabWidget")
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3F3F46;
                border-radius: 15px;
                top: -1px;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: rgba(65, 105, 225, 0.7);
                color: white;
                border: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 10px 15px;
                margin-right: 5px;
                font-weight: bold;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: #4169E1;
                margin-bottom: -1px;
            }
            QTabBar::tab:hover:!selected {
                background-color: rgba(65, 105, 225, 0.9);
            }
        """)


# 修改FaceRecognitionApp類
def add_attendance_tab(app):
    """向主應用添加考勤標籤頁"""
    
    # 創建標籤頁控件
    app.tab_widget = StyledTabWidget()
    
    # 創建主頁控件（原有內容）
    main_tab = QWidget()
    main_layout = QVBoxLayout(main_tab)
    main_layout.setContentsMargins(20, 20, 20, 20)
    
    # 獲取原來中央控件的子控件
    central_widget = app.centralWidget()
    
    # 移動所有子控件到主頁標籤
    for child in central_widget.children():
        if isinstance(child, QWidget) and child != central_widget.layout():
            child.setParent(main_tab)
            main_layout.addWidget(child)
    
    # 添加主頁標籤
    app.tab_widget.addTab(main_tab, "人臉識別")
    
    # 創建考勤標籤頁
    app.attendance_tab = AttendanceTabWidget()
    app.tab_widget.addTab(app.attendance_tab, "考勤管理")
    
    # 設置中央控件
    app.setCentralWidget(app.tab_widget)
    
    # 連接識別結果信號
    original_update_recognition_results = app.update_recognition_results
    
    def new_update_recognition_results(frame, face_locations, names, confidences):
        """擴展的識別結果處理函數"""
        # 調用原函數
        original_update_recognition_results(frame, face_locations, names, confidences)
        
        # 如果有識別到人臉，通知考勤頁面
        if names and confidences:
            for name, confidence in zip(names, confidences):
                if name != "未知" and confidence > 0.5:
                    app.attendance_tab.handle_recognition_result(name, confidence)
                    
    # 替換原函數
    app.update_recognition_results = new_update_recognition_results
    
    # 設置程序整體樣式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1E1E1E;
        }
        QStatusBar {
            background-color: #1E1E1E;
            color: #CCCCCC;
        }
        QLabel {
            color: #E6E6E6;
        }
        QPushButton {
            background-color: #333337;
            color: #E6E6E6;
            border-radius: 10px;
            padding: 8px 15px;
            font-weight: bold;
            border: 1px solid #3F3F46;
        }
        QPushButton:hover {
            background-color: #3F3F46;
            border: 1px solid #5A5A5E;
        }
        QLineEdit {
            background-color: #252526;
            border: 1px solid #3F3F46;
            border-radius: 10px;
            padding: 8px;
            color: #E6E6E6;
        }
        QLineEdit:hover, QLineEdit:focus {
            border: 1px solid #4169E1;
        }
        QComboBox {
            background-color: #252526;
            border: 1px solid #3F3F46;
            border-radius: 10px;
            padding: 8px;
            color: #E6E6E6;
        }
        QComboBox:hover, QComboBox:focus {
            border: 1px solid #4169E1;
        }
        QSlider {
            background-color: transparent;
        }
        QSlider::groove:horizontal {
            background-color: #3F3F46;
            height: 4px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background-color: #4169E1;
            border: none;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background-color: #5A7CF0;
        }
        QListWidget {
            background-color: #252526;
            border: 1px solid #3F3F46;
            border-radius: 10px;
            padding: 5px;
            color: #E6E6E6;
        }
        QListWidget::item {
            padding: 5px;
            border-bottom: 1px solid #3F3F46;
        }
        QListWidget::item:selected {
            background-color: rgba(65, 105, 225, 0.3);
            color: #FFFFFF;
        }
        QListWidget::item:hover:!selected {
            background-color: rgba(65, 105, 225, 0.2);
        }
    """)