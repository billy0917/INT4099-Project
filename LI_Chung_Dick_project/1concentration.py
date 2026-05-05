"""
Focus Monitoring System
Uses computer webcam to monitor student focus levels
"""

import cv2
import numpy as np
import time
import datetime
import os
from collections import deque
import mediapipe as mp
import argparse
import threading  # 添加線程支持，防止聲音播放阻塞主程序

# 嘗試導入專門的聲音庫並提供備選方案
try:
    import winsound  # Windows專用
    use_winsound = True
except ImportError:
    use_winsound = False

try:
    from playsound import playsound
    use_playsound = True
except ImportError:
    use_playsound = False

class FocusMonitor:
    def __init__(self, 
                 focus_history_size=100, 
                 focus_threshold=0.7,
                 looking_down_threshold=3,
                 eye_aspect_ratio_threshold=0.3,
                 head_down_position=0.55,
                 sound_alert_enabled=True,  # 添加聲音提示開關
                 sound_cooldown=5):  # 聲音提示冷卻時間（秒）
        """
        初始化專注力監測系統
        
        參數:
            focus_history_size: 專注力歷史記錄大小，用於平滑專注力評分
            focus_threshold: 專注力閾值，低於此值被視為分心
            looking_down_threshold: 低頭多少秒後可能被視為分心
            eye_aspect_ratio_threshold: 眼睛縱橫比閾值，用於檢測眼睛是否閉合
            head_down_position: 頭部低下的位置閾值
            sound_alert_enabled: 是否啟用聲音提示
            sound_cooldown: 聲音提示的冷卻時間（秒）
        """
        # 初始化攝像頭
        self.cap = cv2.VideoCapture(0)
        
        # 初始化MediaPipe人臉網格檢測
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 初始化MediaPipe手部檢測
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 專注力評分相關參數
        self.focus_history = deque(maxlen=focus_history_size)
        self.focus_threshold = focus_threshold
        self.start_time = time.time()
        self.focus_score = 1.0  # 初始專注力為100%
        self.is_focused = True
        self.looking_down_start = None
        self.looking_down_threshold = looking_down_threshold  # 低頭超過此秒數才可能被視為分心
        self.eye_aspect_ratio_threshold = eye_aspect_ratio_threshold
        self.head_down_position = head_down_position
        
        # 聲音提示相關參數
        self.sound_alert_enabled = sound_alert_enabled
        self.sound_cooldown = sound_cooldown
        self.last_alert_time = 0
        self.was_focused = True  # 記錄上次是否專注的狀態
        
        # 檢測可用的聲音庫
        self.use_winsound = use_winsound
        self.use_playsound = use_playsound
        
        if not (self.use_winsound or self.use_playsound):
            print("警告：未找到可用的聲音播放庫，聲音提示功能將被禁用")
            self.sound_alert_enabled = False
        
        # 創建聲音文件目錄
        self.sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
        os.makedirs(self.sound_dir, exist_ok=True)
        
        # 定義聲音文件路徑
        self.alert_sound_path = os.path.join(self.sound_dir, "alert.mp3")
        self.wav_alert_sound_path = os.path.join(self.sound_dir, "alert.wav")  # 用於winsound
        
        if not (os.path.exists(self.alert_sound_path) or os.path.exists(self.wav_alert_sound_path)):
            self.create_default_sound_file()
        
        # 日誌記錄
        self.log_file = f"focus_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.log_file, 'w') as f:
            f.write("timestamp,focus_score,is_focused,head_position,eyes_open,hands_visible\n")
    
    def create_default_sound_file(self):
        """
        當聲音文件不存在時，提示用戶需要添加聲音文件
        """
        print(f"聲音提示文件不存在")
        if self.use_winsound:
            print(f"請在以下路徑放置名為'alert.wav'的聲音文件:")
            print(f"  {self.wav_alert_sound_path}")
        else:
            print(f"請在以下路徑放置名為'alert.mp3'的聲音文件:")
            print(f"  {self.alert_sound_path}")
        
        print("重要提示：")
        print("1. 文件路徑中避免使用特殊字符和空格")
        print("2. 如果使用Windows，推薦使用WAV格式聲音文件")
        
        # 為了確保程序能夠運行，我們創建一個簡單的文本文件作為占位符
        with open(self.alert_sound_path + ".txt", "w") as f:
            f.write("This is a placeholder. Please replace with an mp3 file for alert sound.")
    
    def play_alert_sound(self):
        """
        播放分心提示音效
        """
        if not self.sound_alert_enabled:
            return
        
        # 檢查冷卻時間
        current_time = time.time()
        if current_time - self.last_alert_time < self.sound_cooldown:
            return
        
        # 更新最後提示時間
        self.last_alert_time = current_time
        
        # 直接嘗試播放聲音，不使用額外的線程
        try:
            # 嘗試使用winsound（Windows專用，支持WAV）
            if self.use_winsound:
                if os.path.exists(self.wav_alert_sound_path):
                    print(f"使用winsound播放: {self.wav_alert_sound_path}")
                    winsound.PlaySound(self.wav_alert_sound_path, winsound.SND_FILENAME)
                else:
                    # 使用系統內置聲音
                    print("使用winsound播放系統聲音")
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            
            # 嘗試使用playsound（跨平台，支持MP3）
            elif self.use_playsound and os.path.exists(self.alert_sound_path):
                normalized_path = os.path.normpath(self.alert_sound_path)
                print(f"使用playsound播放: {normalized_path}")
                playsound(normalized_path)
            
            # 如果都不可用
            else:
                print("警告：無法播放提示音，請檢查聲音文件是否存在")
                
        except Exception as e:
            print(f"播放聲音時發生錯誤: {e}")
            print("提示：可能是文件路徑問題，請確保路徑中沒有特殊字符或空格")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks, face_landmarks, image_shape):
        """
        計算眼睛縱橫比（Eye Aspect Ratio, EAR）
        EAR = 眼睛高度 / 眼睛寬度
        EAR越小，眼睛越閉合
        
        參數:
            eye_landmarks: 眼睛的關鍵點索引列表
            face_landmarks: 臉部識別的關鍵點
            image_shape: 圖片形狀
        
        返回:
            eye_aspect_ratio: 眼睛縱橫比
        """
        h, w = image_shape[:2]
        
        # 獲取眼睛的垂直點
        top_point = face_landmarks.landmark[eye_landmarks[0]]
        bottom_point = face_landmarks.landmark[eye_landmarks[1]]
        
        # 獲取眼睛的水平點
        left_point = face_landmarks.landmark[eye_landmarks[2]]
        right_point = face_landmarks.landmark[eye_landmarks[3]]
        
        # 計算眼睛高度和寬度
        eye_height = abs((bottom_point.y - top_point.y) * h)
        eye_width = abs((right_point.x - left_point.x) * w)
        
        # 防止除零錯誤
        if eye_width == 0:
            return 0
        
        # 計算EAR
        ear = eye_height / eye_width
        return ear
    
    def detect_focus(self, frame):
        """
        檢測專注力的主要功能
        
        參數:
            frame: 視頻幀
            
        返回:
            frame: 處理後的視頻幀
            is_focused: 是否專注
            focus_score: 專注力評分
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # 檢測臉部
        face_results = self.face_mesh.process(frame_rgb)
        face_landmarks = face_results.multi_face_landmarks
        
        # 檢測手部
        hands_results = self.hands.process(frame_rgb)
        hands_landmarks = hands_results.multi_hand_landmarks
        
        # 初始化變數
        head_down = False
        eyes_closed = False
        hands_visible = False
        
        # 繪製手部關鍵點（如果存在）
        if hands_landmarks:
            for hand_landmarks in hands_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            hands_visible = True
        
        # 分析臉部特徵
        if face_landmarks:
            # 繪製臉部網格
            for face_landmark in face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
            
            face = face_landmarks[0]
            
            # 計算頭部姿態
            # MediaPipe臉部網格的關鍵點：1=鼻尖, 152=下巴, 10=前額
            nose = face.landmark[1]
            chin = face.landmark[152]
            forehead = face.landmark[10]
            
            # 通過計算頭部關鍵點的y座標比例來判斷頭部方向
            nose_y = nose.y
            
            # 如果鼻子位置低於閾值，視為低頭
            if nose_y > self.head_down_position:
                head_down = True
                cv2.putText(frame, "Head Down", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 如果開始低頭，記錄時間
                if self.looking_down_start is None:
                    self.looking_down_start = time.time()
            else:
                self.looking_down_start = None
            
            # 檢查眼睛是否閉合（使用眼睛縱橫比）
            # 左眼關鍵點 (上, 下, 左, 右)
            left_eye_landmarks = [159, 145, 33, 133]
            # 右眼關鍵點 (上, 下, 左, 右)
            right_eye_landmarks = [386, 374, 362, 263]
            
            left_ear = self.calculate_eye_aspect_ratio(left_eye_landmarks, face, frame.shape)
            right_ear = self.calculate_eye_aspect_ratio(right_eye_landmarks, face, frame.shape)
            
            # 計算平均EAR
            avg_ear = (left_ear + right_ear) / 2
            
            # 如果EAR低於閾值，視為眼睛閉合
            if avg_ear < self.eye_aspect_ratio_threshold:
                eyes_closed = True
                cv2.putText(frame, "Eyes Closed", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 當低頭時的特殊處理
        low_focus = False
        if head_down:
            # 如果低頭且有手在畫面中，可能是在寫字或做作業
            if hands_visible:
                low_focus = False
                cv2.putText(frame, "Writing/Working", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 如果低頭時間超過閾值且沒有手在畫面中，可能是分心或睡著了
                if self.looking_down_start and (time.time() - self.looking_down_start) > self.looking_down_threshold:
                    low_focus = True
                    cv2.putText(frame, "Distracted/Sleeping", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    low_focus = False
        else:
            # 如果眼睛閉合時間過長，可能是疲勞或分心
            if eyes_closed:
                low_focus = True
        
        # 更新專注力評分
        current_focus = 0.0 if low_focus else 1.0
        self.focus_history.append(current_focus)
        self.focus_score = sum(self.focus_history) / len(self.focus_history)
        
        # 判斷是否專注
        self.is_focused = self.focus_score >= self.focus_threshold
        
        # 僅當總的專注力分數(focus_score)完全為0.00時才播放音效
        if self.focus_score == 0.0:
            print("專注力分數為0.00，播放提示音...")
            self.play_alert_sound()
        
        # 更新上次專注狀態
        self.was_focused = self.is_focused
        
        # 記錄日誌
        with open(self.log_file, 'a') as f:
            f.write(f"{time.time()},{self.focus_score},{1 if self.is_focused else 0},{1 if head_down else 0},{1 if not eyes_closed else 0},{1 if hands_visible else 0}\n")
        
        return frame, self.is_focused, self.focus_score
    
    def visualize(self, frame, is_focused, focus_score):
        """
        視覺化專注力狀態
        
        參數:
            frame: 視頻幀
            is_focused: 是否專注
            focus_score: 專注力評分
            
        返回:
            frame: 添加了視覺化元素的視頻幀
        """
        h, w, _ = frame.shape
        
        # 添加專注力狀態條
        bar_height = 30
        bar_width = int(w * focus_score)
        
        # 根據專注力設定顏色
        if is_focused:
            color = (0, 255, 0)  # 綠色表示專注
            status = "Focused"
        else:
            color = (0, 0, 255)  # 紅色表示分心
            status = "Distracted"
        
        # 繪製專注力條
        cv2.rectangle(frame, (0, 0), (bar_width, bar_height), color, -1)
        cv2.rectangle(frame, (0, 0), (w, bar_height), (255, 255, 255), 1)
        
        # 添加文字說明
        cv2.putText(
            frame, 
            f"Focus: {status} ({focus_score:.2f})", 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # 添加專注時間
        focus_time = time.time() - self.start_time
        hours, remainder = divmod(focus_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"Session Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        cv2.putText(
            frame, 
            time_str, 
            (10, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # 顯示聲音提示狀態
        sound_status = "ON" if self.sound_alert_enabled else "OFF"
        cv2.putText(
            frame,
            f"Sound Alert: {sound_status}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        return frame
    
    def run(self):
        """
        運行專注力監測系統的主循環
        """
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("無法獲取視頻幀。")
                    break
                
                # 鏡像翻轉，使畫面更自然
                frame = cv2.flip(frame, 1)
                
                # 檢測並獲取專注力數據
                frame, is_focused, focus_score = self.detect_focus(frame)
                
                # 視覺化
                frame = self.visualize(frame, is_focused, focus_score)
                
                # 顯示結果
                cv2.imshow('Focus Monitor', frame)
                
                # 鍵盤控制
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):  # 按'q'退出
                    break
                elif key == ord('s'):  # 按's'切換聲音提示
                    self.sound_alert_enabled = not self.sound_alert_enabled
                    status = "開啟" if self.sound_alert_enabled else "關閉"
                    print(f"聲音提示已{status}")
        
        except Exception as e:
            print(f"發生錯誤: {e}")
        
        finally:
            # 釋放資源
            self.cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            self.hands.close()
            print(f"專注力監測日誌已保存到: {self.log_file}")
            print(f"總共監測時間: {time.time() - self.start_time:.2f} 秒")

def parse_arguments():
    """
    解析命令行參數
    
    返回:
        args: 參數字典
    """
    parser = argparse.ArgumentParser(description='專注力監測系統')
    
    parser.add_argument('--focus-threshold', type=float, default=0.7,
                        help='專注力閾值，低於此值被視為分心 (默認: 0.7)')
    
    parser.add_argument('--looking-down-threshold', type=float, default=3.0,
                        help='低頭多少秒後可能被視為分心 (默認: 3.0)')
    
    parser.add_argument('--eye-aspect-ratio-threshold', type=float, default=0.3,
                        help='眼睛縱橫比閾值，用於檢測眼睛是否閉合 (默認: 0.3)')
    
    parser.add_argument('--head-down-position', type=float, default=0.55,
                        help='頭部低下的位置閾值 (默認: 0.55)')
    
    parser.add_argument('--sound-alert', type=bool, default=True,
                        help='是否啟用聲音提示 (默認: True)')
                        
    parser.add_argument('--sound-cooldown', type=float, default=5.0,
                        help='聲音提示的冷卻時間，秒 (默認: 5.0)')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 解析命令行參數
    args = parse_arguments()
    
    print("Focus monitoring system starting...")
    print(f"Focus threshold: {args.focus_threshold}")
    print(f"Looking down threshold: {args.looking_down_threshold} seconds")
    print(f"Eye aspect ratio threshold: {args.eye_aspect_ratio_threshold}")
    print(f"Head down position threshold: {args.head_down_position}")
    print(f"Sound alert: {'Enabled' if args.sound_alert else 'Disabled'}")
    print(f"Sound cooldown: {args.sound_cooldown} seconds")
    print("Press 'q' to exit")
    print("Press 's' to toggle sound alert")
    
    # 創建並運行專注力監測器
    monitor = FocusMonitor(
        focus_threshold=args.focus_threshold,
        looking_down_threshold=args.looking_down_threshold,
        eye_aspect_ratio_threshold=args.eye_aspect_ratio_threshold,
        head_down_position=args.head_down_position,
        sound_alert_enabled=args.sound_alert,
        sound_cooldown=args.sound_cooldown
    )
    monitor.run()