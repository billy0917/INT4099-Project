import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import math

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def angle_between_vectors(v1, v2):
    """計算兩個二維向量之間的角度（以度為單位）"""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)

def calculate_angle(a, b, c):
    """
    計算三個點形成的角度
    
    參數:
        a: 第一個點 [x, y]
        b: 中心點 [x, y]
        c: 第三個點 [x, y]
    
    返回:
        角度(度)
    """
    angle_radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle_degrees = math.degrees(angle_radians)
    
    # 確保角度在0-180之間
    if angle_degrees < 0:
        angle_degrees += 360
    if angle_degrees > 180:
        angle_degrees = 360 - angle_degrees
        
    return angle_degrees

def analyze_sitting_posture(landmarks, image_width, image_height):
    """
    根據MediaPipe關鍵點分析坐姿 (側面視角)
    
    參數:
        landmarks: MediaPipe姿勢關鍵點結果
        image_width: 圖像寬度
        image_height: 圖像高度
    
    返回:
        包含坐姿分析結果的字典
    """
    # MediaPipe關鍵點索引對照:
    # 0: 鼻子
    # 7: 左耳, 8: 右耳
    # 11: 左肩, 12: 右肩
    # 23: 左臀, 24: 右臀
    # 25: 左膝, 26: 右膝
    # 27: 左踝, 28: 右踝
    
    if not landmarks:
        return {"posture_quality": "unknown", "issues": ["no landmarks detected"]}
    
    # 初始化結果字典
    posture_result = {
        "posture_quality": "unknown",
        "posture_type": "unknown",
        "metrics": {},
        "issues": [],
        "suggestions": [],
        "keypoint_detection": {}
    }
    
    try:
        # 檢測哪一側的關鍵點可見性更好
        right_ear = landmarks.landmark[8]
        left_ear = landmarks.landmark[7]
        right_shoulder = landmarks.landmark[12]
        left_shoulder = landmarks.landmark[11]
        right_hip = landmarks.landmark[24]
        left_hip = landmarks.landmark[23]
        
        # 透過X座標判斷人面向的方向（左側或右側）
        # 假設較小的X值表示更靠近相機
        right_side_avg_x = (right_ear.x + right_shoulder.x + right_hip.x) / 3
        left_side_avg_x = (left_ear.x + left_shoulder.x + left_hip.x) / 3
        
        # 判斷面向哪一側 (true代表面向右側，false代表面向左側)
        person_facing_right = right_side_avg_x > left_side_avg_x
        
        # 基於面向方向選擇要分析的側面
        # 如果面向右側，我們分析左側輪廓（相機的視角會看到左側）
        # 如果面向左側，我們分析右側輪廓
        use_right_side = not person_facing_right
        posture_result["analyzed_side"] = "right" if use_right_side else "left"
        
        # 根據選定的側面選擇關鍵點
        ear = landmarks.landmark[8] if use_right_side else landmarks.landmark[7]
        shoulder = landmarks.landmark[12] if use_right_side else landmarks.landmark[11]
        hip = landmarks.landmark[24] if use_right_side else landmarks.landmark[23]
        knee = landmarks.landmark[26] if use_right_side else landmarks.landmark[25]
        nose = landmarks.landmark[0]  # 鼻子作為頭部位置的備選
        
        # 檢查關鍵點可見性
        ear_visible = ear.visibility > 0.5
        shoulder_visible = shoulder.visibility > 0.5
        hip_visible = hip.visibility > 0.5
        knee_visible = knee.visibility > 0.5
        nose_visible = nose.visibility > 0.5
        
        # 記錄每個關鍵點的檢測狀態
        posture_result["keypoint_detection"] = {
            "ear": {"visible": ear_visible, "confidence": float(ear.visibility)},
            "shoulder": {"visible": shoulder_visible, "confidence": float(shoulder.visibility)},
            "hip": {"visible": hip_visible, "confidence": float(hip.visibility)},
            "knee": {"visible": knee_visible, "confidence": float(knee.visibility)},
            "nose": {"visible": nose_visible, "confidence": float(nose.visibility)}
        }
        
        # 初始化測量結果
        angle_metrics = {}
        
        # 轉換為像素坐標用於角度計算
        if shoulder_visible and hip_visible:
            shoulder_px = np.array([int(shoulder.x * image_width), int(shoulder.y * image_height)])
            hip_px = np.array([int(hip.x * image_width), int(hip.y * image_height)])
            
            # 計算脊柱角度（與垂直線的夾角）
            spine_vector = shoulder_px - hip_px
            vertical_vector = np.array([0, -1])
            spine_angle = angle_between_vectors(spine_vector, vertical_vector)
            angle_metrics["spine_angle"] = float(spine_angle)
            
            # 垂直參考點（與肩同高但垂直於臀部）
            vertical_ref = np.array([hip_px[0], shoulder_px[1]])
            
            # 使用三點法計算脊柱角度（備選方法）
            spine_angle_alt = calculate_angle(vertical_ref, hip_px, shoulder_px)
            angle_metrics["spine_angle_alt"] = float(spine_angle_alt)
        
        if ear_visible and shoulder_visible:
            ear_px = np.array([int(ear.x * image_width), int(ear.y * image_height)])
            shoulder_px = np.array([int(shoulder.x * image_width), int(shoulder.y * image_height)])
            
            # 計算頸部角度（與垂直線的夾角）
            neck_vector = ear_px - shoulder_px
            vertical_vector = np.array([0, -1])
            neck_angle = angle_between_vectors(neck_vector, vertical_vector)
            angle_metrics["neck_angle"] = float(neck_angle)
        
        if hip_visible and knee_visible:
            hip_px = np.array([int(hip.x * image_width), int(hip.y * image_height)])
            knee_px = np.array([int(knee.x * image_width), int(knee.y * image_height)])
            
            # 計算大腿角度（與水平線的夾角）
            thigh_vector = hip_px - knee_px
            horizontal_vector = np.array([1, 0])
            thigh_angle = angle_between_vectors(thigh_vector, horizontal_vector)
            angle_metrics["thigh_angle"] = float(thigh_angle)
        
        # 如果沒有耳朵關鍵點但有鼻子，使用鼻子計算頭部位置
        if not ear_visible and nose_visible and shoulder_visible:
            nose_px = np.array([int(nose.x * image_width), int(nose.y * image_height)])
            shoulder_px = np.array([int(shoulder.x * image_width), int(shoulder.y * image_height)])
            
            head_vector = nose_px - shoulder_px
            head_angle = angle_between_vectors(head_vector, vertical_vector)
            angle_metrics["head_angle"] = float(head_angle)
        
        # 保存所有角度測量結果
        posture_result["metrics"] = angle_metrics
        
        # 確定坐姿質量和類型
        if angle_metrics:  # 確保至少計算了一個角度
            # 初始化評分系統
            posture_score = 0
            total_metrics = 0
            posture_issues = []
            
            # 評估脊柱角度
            if "spine_angle" in angle_metrics:
                total_metrics += 1
                spine_angle = angle_metrics["spine_angle"]
                
                # 理想脊柱角度應接近垂直（0-10度）
                if abs(spine_angle) < 10:
                    posture_score += 1
                    posture_result["spine_status"] = "good (upright)"
                elif spine_angle > 10:  # 向後傾斜
                    posture_issues.append("leaning back")
                    posture_result["spine_status"] = "leaning back"
                else:  # 向前傾斜
                    posture_issues.append("hunching forward")
                    posture_result["spine_status"] = "hunching forward"
            
            # 評估頸部角度
            if "neck_angle" in angle_metrics:
                total_metrics += 1
                neck_angle = angle_metrics["neck_angle"]
                
                # 理想的頸部角度約為15度
                if abs(neck_angle - 15) < 10:
                    posture_score += 1
                    posture_result["neck_status"] = "good alignment"
                elif neck_angle > 25:  # 頭部過度前傾
                    posture_issues.append("forward head posture")
                    posture_result["neck_status"] = "forward head posture"
                else:  # 頸部向後傾斜
                    posture_issues.append("neck tilted back")
                    posture_result["neck_status"] = "tilted back"
            
            # 評估大腿角度（如果有）
            if "thigh_angle" in angle_metrics:
                total_metrics += 1
                thigh_angle = angle_metrics["thigh_angle"]
                
                # 理想坐姿時大腿應相對水平
                if abs(thigh_angle - 90) < 15:
                    posture_score += 1
                    posture_result["thigh_status"] = "good (horizontal)"
                else:
                    posture_issues.append("improper thigh position")
                    posture_result["thigh_status"] = "improper position"
            
            # 如果我們使用了鼻子來估計頭部位置
            if "head_angle" in angle_metrics and not "neck_angle" in angle_metrics:
                total_metrics += 1
                head_angle = angle_metrics["head_angle"]
                
                # 使用頭部角度近似評估
                if abs(head_angle - 15) < 10:
                    posture_score += 1
                    posture_result["head_status"] = "good alignment"
                elif head_angle > 25:
                    posture_issues.append("forward head posture")
                    posture_result["head_status"] = "forward head posture"
                else:
                    posture_issues.append("head tilted back")
                    posture_result["head_status"] = "tilted back"
            
            # 計算最終坐姿質量
            if total_metrics > 0:
                posture_ratio = posture_score / total_metrics
                
                if posture_ratio >= 0.7:
                    posture_result["posture_quality"] = "good"
                elif posture_ratio >= 0.4:
                    posture_result["posture_quality"] = "fair"
                else:
                    posture_result["posture_quality"] = "poor"
                
                # 根據檢測到的問題確定坐姿類型
                if "hunching forward" in posture_issues:
                    posture_result["posture_type"] = "hunched forward"
                elif "leaning back" in posture_issues:
                    posture_result["posture_type"] = "leaning back"
                elif "forward head posture" in posture_issues:
                    posture_result["posture_type"] = "forward head"
                elif posture_result["posture_quality"] == "good":
                    posture_result["posture_type"] = "upright"
                else:
                    posture_result["posture_type"] = "mixed issues"
                
                # 保存所有檢測到的問題
                posture_result["issues"] = posture_issues
                
                # 為每個問題添加建議
                for issue in posture_issues:
                    if issue == "leaning back":
                        posture_result["suggestions"].append("move closer to desk")
                    elif issue == "hunching forward":
                        posture_result["suggestions"].append("sit back with spine aligned to chair")
                    elif issue == "forward head posture":
                        posture_result["suggestions"].append("pull head back to align with shoulders")
                    elif issue == "neck tilted back":
                        posture_result["suggestions"].append("adjust screen height to eye level")
                    elif issue == "improper thigh position":
                        posture_result["suggestions"].append("adjust chair height for proper thigh position")
            else:
                # 如果無法計算任何角度，給出一般建議
                posture_result["issues"].append("insufficient key points detected")
                posture_result["suggestions"].append("try to reposition for better visibility")
    
    except Exception as e:
        posture_result["error"] = str(e)
        print(f"Error analyzing posture: {e}")
    
    return posture_result

def visualize_posture_results(frame, posture_result, landmarks=None):
    """
    在圖像上可視化坐姿分析結果
    
    參數:
        frame: 原始圖像幀
        posture_result: 坐姿分析結果
        landmarks: MediaPipe檢測到的關鍵點 (可選)
        
    返回:
        帶有坐姿信息的標註幀
    """
    frame_h, frame_w = frame.shape[:2]
    
    # 決定資訊板位置 (上方中間)
    panel_x = frame_w // 4
    panel_y = 20
    
    # 計算資訊板大小
    info_width = frame_w // 2
    metrics_count = len(posture_result.get("metrics", {}))
    issues_count = len(posture_result.get("issues", []))
    
    # 計算資訊板的高度 - 基本高度加上每個指標和問題的額外行
    info_height = 130 + (metrics_count * 25) + (issues_count * 25)
    
    # 繪製半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + info_width, panel_y + info_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 繪製邊框
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + info_width, panel_y + info_height), 
                 (200, 200, 200), 2)
    
    # 開始在資訊板上繪製文本
    current_y = panel_y + 30
    
    # 繪製標題
    posture_quality = posture_result.get("posture_quality", "unknown")
    quality_color = (0, 255, 0) if posture_quality == "good" else \
                   (0, 165, 255) if posture_quality == "fair" else (0, 0, 255)
    
    cv2.putText(frame, f"Posture Analysis", 
              (panel_x + 10, current_y), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    current_y += 30
    
    # 繪製坐姿質量
    cv2.putText(frame, f"Quality: {posture_quality.upper()}", 
              (panel_x + 10, current_y), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
    current_y += 30
    
    # 繪製坐姿類型
    posture_type = posture_result.get("posture_type", "unknown")
    cv2.putText(frame, f"Type: {posture_type}", 
              (panel_x + 10, current_y), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    current_y += 30
    
    # 分析側面
    if "analyzed_side" in posture_result:
        cv2.putText(frame, f"Analyzed: {posture_result['analyzed_side']} side", 
                  (panel_x + 10, current_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        current_y += 25
    
    # 顯示主要角度測量
    if "metrics" in posture_result:
        metrics = posture_result["metrics"]
        for metric_name, value in metrics.items():
            if metric_name == "spine_angle":
                status = posture_result.get("spine_status", "")
                status_text = f" - {status}" if status else ""
                cv2.putText(frame, f"Spine: {value:.1f}°{status_text}", 
                          (panel_x + 10, current_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                current_y += 25
            elif metric_name == "neck_angle":
                status = posture_result.get("neck_status", "")
                status_text = f" - {status}" if status else ""
                cv2.putText(frame, f"Neck: {value:.1f}°{status_text}", 
                          (panel_x + 10, current_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                current_y += 25
            elif metric_name == "thigh_angle":
                status = posture_result.get("thigh_status", "")
                status_text = f" - {status}" if status else ""
                cv2.putText(frame, f"Thigh: {value:.1f}°{status_text}", 
                          (panel_x + 10, current_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                current_y += 25
            elif metric_name == "head_angle":
                status = posture_result.get("head_status", "")
                status_text = f" - {status}" if status else ""
                cv2.putText(frame, f"Head: {value:.1f}°{status_text}", 
                          (panel_x + 10, current_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                current_y += 25
    
    # 繪製問題（如果有）
    if "issues" in posture_result and posture_result["issues"]:
        cv2.putText(frame, "Issues:", 
                  (panel_x + 10, current_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        current_y += 25
        
        for j, issue in enumerate(posture_result["issues"]):
            # 檢查是否有足夠空間，防止文本超出幀
            if current_y >= frame_h - 10:
                break
                
            cv2.putText(frame, f"- {issue}", 
                      (panel_x + 20, current_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            current_y += 25
    
    # 如果有提供關鍵點，繪製詳細的姿勢線條
    if landmarks:
        try:
            # 獲取使用的側面
            side = "right" if posture_result.get("analyzed_side", "") == "right" else "left"
            
            # 根據選擇的側面確定索引
            ear_idx = 8 if side == "right" else 7
            shoulder_idx = 12 if side == "right" else 11
            hip_idx = 24 if side == "right" else 23
            knee_idx = 26 if side == "right" else 25
            
            # 耳朵到肩膀線（頸部）
            if landmarks.landmark[ear_idx].visibility > 0.5 and landmarks.landmark[shoulder_idx].visibility > 0.5:
                ear_coords = (int(landmarks.landmark[ear_idx].x * frame_w), 
                             int(landmarks.landmark[ear_idx].y * frame_h))
                shoulder_coords = (int(landmarks.landmark[shoulder_idx].x * frame_w), 
                                 int(landmarks.landmark[shoulder_idx].y * frame_h))
                cv2.line(frame, ear_coords, shoulder_coords, (0, 255, 0), 2)
                
                # 在頸部中點顯示角度
                if "neck_angle" in posture_result.get("metrics", {}):
                    neck_angle = posture_result["metrics"]["neck_angle"]
                    midpoint = ((ear_coords[0] + shoulder_coords[0]) // 2, 
                               (ear_coords[1] + shoulder_coords[1]) // 2)
                    cv2.putText(frame, f"{neck_angle:.1f}°", midpoint, 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 肩膀到臀部線（脊柱）
            if landmarks.landmark[shoulder_idx].visibility > 0.5 and landmarks.landmark[hip_idx].visibility > 0.5:
                shoulder_coords = (int(landmarks.landmark[shoulder_idx].x * frame_w), 
                                 int(landmarks.landmark[shoulder_idx].y * frame_h))
                hip_coords = (int(landmarks.landmark[hip_idx].x * frame_w), 
                             int(landmarks.landmark[hip_idx].y * frame_h))
                cv2.line(frame, shoulder_coords, hip_coords, (0, 0, 255), 2)
                
                # 在脊柱中點顯示角度
                if "spine_angle" in posture_result.get("metrics", {}):
                    spine_angle = posture_result["metrics"]["spine_angle"]
                    midpoint = ((shoulder_coords[0] + hip_coords[0]) // 2, 
                               (shoulder_coords[1] + hip_coords[1]) // 2)
                    cv2.putText(frame, f"{spine_angle:.1f}°", midpoint, 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 繪製垂直參考線
                ref_x = (shoulder_coords[0] + hip_coords[0]) // 2
                cv2.line(frame, 
                       (ref_x, shoulder_coords[1] - 50), 
                       (ref_x, hip_coords[1] + 50), 
                       (255, 255, 255), 1, cv2.LINE_DASH)
            
            # 臀部到膝蓋線（大腿）
            if landmarks.landmark[hip_idx].visibility > 0.5 and landmarks.landmark[knee_idx].visibility > 0.5:
                hip_coords = (int(landmarks.landmark[hip_idx].x * frame_w), 
                             int(landmarks.landmark[hip_idx].y * frame_h))
                knee_coords = (int(landmarks.landmark[knee_idx].x * frame_w), 
                              int(landmarks.landmark[knee_idx].y * frame_h))
                cv2.line(frame, hip_coords, knee_coords, (255, 0, 255), 2)
                
                # 在大腿中點顯示角度
                if "thigh_angle" in posture_result.get("metrics", {}):
                    thigh_angle = posture_result["metrics"]["thigh_angle"]
                    midpoint = ((hip_coords[0] + knee_coords[0]) // 2, 
                               (hip_coords[1] + knee_coords[1]) // 2)
                    cv2.putText(frame, f"{thigh_angle:.1f}°", midpoint, 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
        except Exception as e:
            print(f"Error drawing posture lines: {e}")
    
    return frame

def pose_detection_with_posture(width=1280, height=720):
    """
    使用MediaPipe和網絡攝像頭進行實時坐姿分析
    
    參數:
        width: 攝像頭寬度
        height: 攝像頭高度
    """
    # 打開網絡攝像頭
    cap = cv2.VideoCapture(0)
    
    # 設置攝像頭分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 檢查是否成功設置了分辨率
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Set resolution: {width}x{height}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    
    # 初始化MediaPipe姿勢檢測器
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1) as pose:
        
        # 用於幀率計算
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while cap.isOpened():
            # 讀取幀
            success, frame = cap.read()
            if not success:
                print("Error: Unable to capture frame.")
                break
            
            # 更新FPS計數器
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # 為了改善性能，可選擇性地將圖像標記為不可寫入
            frame.flags.writeable = False
            
            # 將BGR圖像轉換為RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 處理圖像
            results = pose.process(frame_rgb)
            
            # 可以再次修改圖像
            frame.flags.writeable = True
            
            # 獲取圖像尺寸
            image_height, image_width, _ = frame.shape
            
            # 如果檢測到姿勢關鍵點
            if results.pose_landmarks:
                # 進行坐姿分析
                posture_result = analyze_sitting_posture(
                    results.pose_landmarks, image_width, image_height)
                
                # 繪製姿勢關鍵點和連接線
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # 可視化坐姿分析結果
                frame = visualize_posture_results(frame, posture_result, results.pose_landmarks)
            else:
                # 如果未檢測到關鍵點，顯示消息
                cv2.putText(frame, "No pose detected", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 添加FPS信息
            cv2.putText(frame, f"FPS: {fps}", (20, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 顯示圖像
            cv2.imshow('MediaPipe Pose with Posture Analysis', frame)
            
            # 如果按下'q'則退出循環
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

def process_image_with_posture(image_path):
    """
    使用MediaPipe處理圖像進行姿態檢測和坐姿分析
    
    參數:
        image_path: 圖像文件路徑
        
    返回:
        帶有姿態和坐姿分析的標註圖像
    """
    # 讀取圖像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return None
    
    # 獲取圖像尺寸
    image_height, image_width, _ = image.shape
    
    # 初始化MediaPipe姿勢檢測器
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        model_complexity=2) as pose:
        
        # 將BGR圖像轉換為RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 處理圖像
        results = pose.process(image_rgb)
        
        # 創建輸出圖像副本
        annotated_image = image.copy()
        
        # 如果檢測到姿勢關鍵點
        if results.pose_landmarks:
            # 進行坐姿分析
            posture_result = analyze_sitting_posture(
                results.pose_landmarks, image_width, image_height)
            
            # 繪製姿勢關鍵點和連接線
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # 可視化坐姿分析結果
            annotated_image = visualize_posture_results(
                annotated_image, posture_result, results.pose_landmarks)
            
            return annotated_image, posture_result
        else:
            print("No pose detected in the image.")
            # 如果未檢測到關鍵點，顯示消息
            cv2.putText(annotated_image, "No pose detected", (50, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_image, {"posture_quality": "unknown", "issues": ["no pose detected"]}

def main():
    parser = argparse.ArgumentParser(description="Pose Detection and Posture Analysis with MediaPipe")
    parser.add_argument("--mode", type=str, default="webcam", choices=["webcam", "image"],
                       help="Run mode: webcam or image")
    parser.add_argument("--image", type=str, help="Image path for image mode")
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    
    args = parser.parse_args()
    
    if args.mode == "webcam":
        pose_detection_with_posture(width=args.width, height=args.height)
    else:
        if args.image is None:
            print("Error: Please provide an image path using --image when using image mode.")
        else:
            annotated_image, posture_result = process_image_with_posture(args.image)
            
            if annotated_image is not None:
                # 顯示結果
                cv2.imshow("MediaPipe Posture Analysis", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 打印坐姿分析結果
                print("\nPosture Analysis Results:")
                print(f"  Quality: {posture_result.get('posture_quality', 'unknown')}")
                print(f"  Type: {posture_result.get('posture_type', 'unknown')}")
                
                # 打印角度測量
                if "metrics" in posture_result:
                    print("\nAngle Measurements:")
                    for metric, value in posture_result["metrics"].items():
                        print(f"  - {metric}: {value:.1f}°")
                
                # 打印問題和建議
                if posture_result.get("issues"):
                    print("\nDetected Issues:")
                    for issue in posture_result["issues"]:
                        print(f"  - {issue}")
                    
                    if posture_result.get("suggestions"):
                        print("\nSuggestions:")
                        for suggestion in posture_result["suggestions"]:
                            print(f"  - {suggestion}")
                else:
                    print("\nNo posture issues detected.")

if __name__ == "__main__":
    main()