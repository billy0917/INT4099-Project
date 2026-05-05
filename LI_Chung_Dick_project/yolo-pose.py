import cv2
import torch
import numpy as np
from ultralytics import YOLO

def pose_detection(width=1920, height=1080):
    # 載入 YOLOv8 pose 模型
    from ultralytics import YOLO
    model = YOLO('yolov8n-pose.pt')
    
    # 開啟網絡攝像頭
    cap = cv2.VideoCapture(0)
    
    # 設置攝像頭分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 檢查是否成功設置了分辨率
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"設置的分辨率: {width}x{height}")
    print(f"實際分辨率: {actual_width}x{actual_height}")
    
    if not cap.isOpened():
        print("錯誤: 無法開啟網絡攝像頭。")
        return
    
    while True:
        # 讀取幀
        success, frame = cap.read()
        if not success:
            print("錯誤: 無法獲取幀。")
            break
            
        # 在幀上運行 YOLOv8 姿態估計
        results = model(frame)
        
        # 在幀上可視化結果
        annotated_frame = results[0].plot()
        
        # 顯示註釋幀
        cv2.imshow("YOLOv8 姿態檢測", annotated_frame)
        
        # 如果按下 'q' 則退出循環
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 釋放網絡攝像頭並關閉窗口
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    # Load the YOLOv8 pose model
    model = YOLO('yolov8n-pose.pt')
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Run YOLOv8 pose estimation on the image
    results = model(image)
    
    # Get keypoints
    keypoints = results[0].keypoints.data.cpu().numpy()
    
    # Visualize the results on the image
    annotated_image = results[0].plot()
    
    return annotated_image, keypoints

if __name__ == "__main__":
    # For webcam-based pose detection
    pose_detection(width=1920, height=1080)
    
 