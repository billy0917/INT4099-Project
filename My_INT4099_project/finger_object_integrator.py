import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import subprocess
import sys
import threading

# 導入食指識別器
from finger_recognizer import FingerObjectRecognizer
# 導入Ultralytics YOLO物體訓練器
from ultralytics_yolo_trainer import UltralyticsTrainer

# 檢查並創建獨立訓練腳本
def ensure_standalone_trainer_exists():
    """確保獨立訓練腳本存在"""
    standalone_script = "standalone_yolo_trainer.py"
    
    if not os.path.exists(standalone_script):
        print(f"創建獨立訓練腳本: {standalone_script}")
        with open(standalone_script, "w", encoding="utf-8") as f:
            f.write("""\"\"\"
獨立的YOLO訓練腳本 - 可從主程序調用
使用方法: python standalone_yolo_trainer.py
\"\"\"

import os
import sys
import yaml
import time
import shutil
from pathlib import Path

def train_yolo(base_folder="ultralytics_yolo", log_file="training_log.txt"):
    \"\"\"獨立訓練YOLO模型，並將結果保存到指定位置\"\"\"
    start_time = time.time()
    success = False
    
    print(f"開始YOLO模型訓練，日誌將保存到 {log_file}")
    
    with open(log_file, "w", encoding="utf-8") as log:
        # 記錄訓練開始
        log.write(f"=== YOLO訓練開始於 {time.ctime()} ===\\n")
        
        try:
            # 設置路徑
            data_folder = os.path.join(base_folder, "data")
            model_folder = os.path.join(base_folder, "model")
            trained_model_path = os.path.join(model_folder, "best.pt")
            data_yaml_path = os.path.join(data_folder, "dataset.yaml")
            labels_path = os.path.join(model_folder, "labels.txt")
            
            # 檢查數據集配置文件
            if not os.path.exists(data_yaml_path):
                log.write(f"錯誤: 數據集配置文件不存在: {data_yaml_path}\\n")
                return False
            
            # 檢查標籤文件
            if not os.path.exists(labels_path):
                log.write(f"錯誤: 標籤文件不存在: {labels_path}\\n")
                return False
            
            # 載入標籤
            with open(labels_path, 'r', encoding="utf-8") as f:
                labels = [line.strip() for line in f.readlines()]
            
            log.write(f"已載入標籤: {labels}\\n")
            
            # 檢查訓練圖像
            images_train = os.path.join(data_folder, "images", "train")
            images_val = os.path.join(data_folder, "images", "val")
            
            train_images_count = len(os.listdir(images_train)) if os.path.exists(images_train) else 0
            val_images_count = len(os.listdir(images_val)) if os.path.exists(images_val) else 0
            
            log.write(f"訓練圖像: {train_images_count}張, 驗證圖像: {val_images_count}張\\n")
            
            if train_images_count + val_images_count < 5:
                log.write(f"錯誤: 訓練和驗證圖像總數過少: {train_images_count + val_images_count}\\n")
                return False
            
            # 導入必要的庫
            try:
                from ultralytics import YOLO
                log.write("成功導入Ultralytics YOLO\\n")
            except ImportError as e:
                log.write(f"錯誤: 無法導入Ultralytics模組: {e}\\n")
                log.write("嘗試安裝Ultralytics...\\n")
                
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    from ultralytics import YOLO
                    log.write("成功安裝並導入Ultralytics YOLO\\n")
                except Exception as install_error:
                    log.write(f"錯誤: 安裝Ultralytics失敗: {install_error}\\n")
                    return False
            
            # 檢查是否可以導入PyTorch
            try:
                import torch
                log.write(f"PyTorch版本: {torch.__version__}\\n")
                if torch.cuda.is_available():
                    log.write(f"CUDA可用: {torch.cuda.get_device_name(0)}\\n")
                else:
                    log.write("CUDA不可用，將使用CPU訓練（可能較慢）\\n")
            except ImportError:
                log.write("警告: 無法導入PyTorch，訓練可能無法進行\\n")
            
            # 執行訓練
            log.write("開始訓練模型...\\n")
            try:
                # 載入預訓練模型
                model = YOLO('yolov8n.pt')
                
                # 配置訓練參數
                results = model.train(
                    data=data_yaml_path,
                    epochs=50,
                    imgsz=640,
                    batch=4,
                    patience=10,
                    project=model_folder,
                    name='train',
                    exist_ok=True
                )
                
                log.write("模型訓練完成!\\n")
                success = True
                
                # 檢查並複製模型文件
                trained_model = os.path.join(model_folder, 'train', 'weights', 'best.pt')
                if os.path.exists(trained_model):
                    log.write(f"找到訓練完成的模型: {trained_model}\\n")
                    file_size_mb = os.path.getsize(trained_model) / (1024 * 1024)
                    log.write(f"模型文件大小: {file_size_mb:.2f} MB\\n")
                    
                    # 複製到最終位置
                    try:
                        shutil.copy2(trained_model, trained_model_path)
                        log.write(f"已將模型複製到: {trained_model_path}\\n")
                    except Exception as copy_error:
                        log.write(f"警告: 複製模型文件時出錯: {copy_error}\\n")
                        log.write(f"請手動將 {trained_model} 複製到 {trained_model_path}\\n")
                else:
                    log.write(f"警告: 找不到訓練完成的模型文件: {trained_model}\\n")
                    # 嘗試查找last.pt
                    last_model = os.path.join(model_folder, 'train', 'weights', 'last.pt')
                    if os.path.exists(last_model):
                        log.write(f"找到last.pt模型: {last_model}\\n")
                        try:
                            shutil.copy2(last_model, trained_model_path)
                            log.write(f"已將last.pt模型複製到: {trained_model_path}\\n")
                        except Exception as copy_error:
                            log.write(f"警告: 複製last.pt文件時出錯: {copy_error}\\n")
                    else:
                        log.write("錯誤: 未找到任何訓練輸出模型\\n")
                        success = False
            
            except Exception as train_error:
                log.write(f"訓練過程中發生錯誤: {train_error}\\n")
                import traceback
                log.write(traceback.format_exc())
                success = False
        
        except Exception as e:
            log.write(f"訓練準備過程中發生錯誤: {e}\\n")
            import traceback
            log.write(traceback.format_exc())
            success = False
        
        # 記錄訓練結束
        end_time = time.time()
        duration = end_time - start_time
        log.write(f"=== YOLO訓練結束於 {time.ctime()} ===\\n")
        log.write(f"總耗時: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)\\n")
        log.write(f"訓練結果: {'成功' if success else '失敗'}\\n")
    
    print(f"YOLO訓練{'成功' if success else '失敗'}，查看日誌獲取詳情: {log_file}")
    return success

if __name__ == "__main__":
    # 如果直接運行此腳本
    base_folder = "ultralytics_yolo"
    log_file = "yolo_training_log.txt"
    
    # 從命令行參數中獲取設置
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    if len(sys.argv) > 2:
        log_file = sys.argv[2]
    
    # 運行訓練
    result = train_yolo(base_folder, log_file)
    
    # 顯示結果
    if result:
        print("訓練成功完成!")
    else:
        print("訓練失敗，請查看日誌文件獲取詳情。")
    
    # 等待用戶按鍵退出
    input("按Enter鍵退出...")
""")
    return os.path.exists(standalone_script)

# 檢查並創建日誌查看器
def ensure_log_reader_exists():
    """確保日誌查看器腳本存在"""
    log_reader = "log_reader.py"
    
    if not os.path.exists(log_reader):
        print(f"創建日誌查看器腳本: {log_reader}")
        with open(log_reader, "w", encoding="utf-8") as f:
            f.write("""\"\"\"
YOLO訓練日誌查看器
提供實時監控訓練進度的GUI界面
\"\"\"

import os
import tkinter as tk
from tkinter import scrolledtext
import threading
import time

def show_training_status(log_file="yolo_training_log.txt"):
    \"\"\"顯示YOLO訓練狀態與日誌\"\"\"
    if not os.path.exists(log_file):
        print(f"訓練日誌文件不存在: {log_file}")
        print("可能尚未開始訓練，或訓練日誌保存在其他位置")
        return
    
    # 創建簡單的GUI顯示訓練日誌
    root = tk.Tk()
    root.title("YOLO訓練狀態監視器")
    root.geometry("900x700")
    
    # 添加標頭標籤
    header = tk.Label(root, text="YOLO模型訓練狀態", font=("Arial", 16))
    header.pack(pady=10)
    
    # 顯示日誌文件路徑
    path_label = tk.Label(root, text=f"日誌檔案: {os.path.abspath(log_file)}", font=("Arial", 10))
    path_label.pack(pady=5)
    
    # 創建狀態標籤
    status_label = tk.Label(root, text="正在載入...", font=("Arial", 12))
    status_label.pack(pady=5)
    
    # 創建滾動文本區
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 10))
    text_area.pack(expand=True, fill='both', padx=10, pady=10)
    
    # 創建按鈕框架
    button_frame = tk.Frame(root)
    button_frame.pack(fill='x', pady=10)
    
    # 添加刷新按鈕
    refresh_button = tk.Button(button_frame, text="刷新", width=10)
    refresh_button.pack(side=tk.LEFT, padx=10)
    
    # 添加關閉按鈕
    close_button = tk.Button(button_frame, text="關閉", width=10, command=root.destroy)
    close_button.pack(side=tk.RIGHT, padx=10)
    
    # 進度條框架
    progress_frame = tk.Frame(root, height=30)
    progress_frame.pack(fill='x', padx=10, pady=5)
    
    # 訓練進度條背景
    progress_bg = tk.Canvas(progress_frame, height=20, bg='lightgray')
    progress_bg.pack(fill='x')
    
    # 進度條
    progress_bar = progress_bg.create_rectangle(0, 0, 0, 20, fill='green')
    
    # 進度文本
    progress_text = progress_bg.create_text(450, 10, text="", fill="black")
    
    # 讀取日誌文件的函數
    def update_log():
        last_size = 0
        training_complete = False
        epoch_pattern = "Epoch"
        max_epochs = 50  # 默認最大輪數
        current_epoch = 0
        
        while True:
            try:
                if os.path.exists(log_file):
                    current_size = os.path.getsize(log_file)
                    
                    if current_size > last_size or refresh_button.get() == 1:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # 更新文本區域
                            text_area.delete(1.0, tk.END)
                            text_area.insert(tk.END, content)
                            text_area.see(tk.END)  # 滾動到底部
                            
                            # 解析進度
                            lines = content.split('\\n')
                            for line in lines:
                                # 檢查是否包含訓練輪數信息
                                if epoch_pattern in line and "epochs=" in line:
                                    try:
                                        # 嘗試提取總輪數
                                        epochs_part = line.split("epochs=")[1].split(",")[0]
                                        max_epochs = int(epochs_part)
                                    except:
                                        pass
                                
                                # 檢查當前輪數
                                if "epoch=" in line and "completed" in line:
                                    try:
                                        # 提取當前輪數
                                        current_part = line.split("epoch=")[1].split()[0]
                                        current_epoch = int(current_part)
                                    except:
                                        pass
                            
                            # 更新進度條
                            if max_epochs > 0 and current_epoch > 0:
                                progress_percent = min(100, (current_epoch / max_epochs) * 100)
                                canvas_width = progress_bg.winfo_width()
                                progress_width = (progress_percent / 100) * canvas_width
                                progress_bg.coords(progress_bar, 0, 0, progress_width, 20)
                                progress_bg.itemconfig(progress_text, text=f"進度: {current_epoch}/{max_epochs} ({progress_percent:.1f}%)")
                            
                            # 檢查是否訓練完成
                            if "=== YOLO訓練結束於" in content:
                                if "訓練結果: 成功" in content:
                                    status_label.config(text="訓練已成功完成！", fg="green")
                                    training_complete = True
                                else:
                                    status_label.config(text="訓練失敗，請查看日誌獲取詳情", fg="red")
                                    training_complete = True
                            else:
                                status_label.config(text="訓練進行中...", fg="blue")
                        
                        last_size = current_size
                    
                    # 如果訓練已完成，不再頻繁更新
                    if training_complete:
                        time.sleep(2)
                    else:
                        time.sleep(0.5)
                else:
                    status_label.config(text=f"找不到日誌文件: {log_file}", fg="red")
                    time.sleep(2)
            except Exception as e:
                text_area.insert(tk.END, f"\\n讀取日誌時出錯: {e}\\n")
                time.sleep(2)
            
            # 如果窗口被關閉，停止更新
            if not root.winfo_exists():
                return
    
    # 刷新按鈕事件
    def refresh_logs():
        status_label.config(text="正在刷新...", fg="blue")
        # 通過設置標誌來觸發刷新
        refresh_button.set(1)
        # 0.1秒後重置標誌
        root.after(100, lambda: refresh_button.set(0))
    
    # 設置刷新按鈕的事件和狀態變量
    refresh_button.config(command=refresh_logs)
    refresh_button.set = lambda v: refresh_button.config(state=tk.NORMAL if v==0 else tk.DISABLED)
    refresh_button.get = lambda: 1 if refresh_button.cget('state') == tk.DISABLED else 0
    
    # 在單獨的線程中更新日誌
    log_thread = threading.Thread(target=update_log, daemon=True)
    log_thread.start()
    
    # 運行主循環
    root.mainloop()

if __name__ == "__main__":
    # 可以通過命令行參數指定日誌文件
    import sys
    log_file = "yolo_training_log.txt"
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    show_training_status(log_file)
""")
    return os.path.exists(log_reader)

# 定義顯示訓練日誌的函數
def show_training_logs():
    """在獨立的進程中顯示訓練日誌"""
    try:
        # 檢查日誌文件是否存在
        log_file = "yolo_training_log.txt"
        if not os.path.exists(log_file):
            print(f"訓練日誌文件不存在: {log_file}")
            messagebox.showinfo("日誌不存在", f"找不到訓練日誌文件: {log_file}\n可能尚未開始訓練。")
            return
        
        # 確保日誌查看器存在
        if ensure_log_reader_exists():
            # 在單獨的進程中啟動日誌查看器
            subprocess.Popen([sys.executable, "log_reader.py", log_file])
        else:
            print("無法創建日誌查看器腳本")
            messagebox.showerror("錯誤", "無法創建日誌查看器腳本")
    except Exception as e:
        print(f"啟動日誌查看器時出錯: {e}")
        messagebox.showerror("錯誤", f"啟動日誌查看器時出錯:\n{e}")

def list_and_select_cameras():
    """列出所有可用摄像头并让用户选择"""
    available_cameras = []
    camera_info = []
    
    print("正在检测所有可用摄像头...")
    
    # 测试更多可能的摄像头索引（包括DroidCam可能使用的索引）
    for i in range(10):  # 检查索引0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                info = f"摄像头 {i}: {width}x{height} @ {fps:.1f}fps"
                
                # 显示一帧图像，帮助用户识别每个摄像头
                if frame is not None:
                    # 在图像上显示摄像头索引
                    cv2.putText(frame, f"Camera {i}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 显示预览窗口
                    cv2.imshow(f"Camera {i} Preview", frame)
                    cv2.waitKey(1000)  # 显示1秒
                    cv2.destroyWindow(f"Camera {i} Preview")
                
                print(info)
                available_cameras.append(i)
                camera_info.append(info)
            cap.release()
    
    if not available_cameras:
        print("没有检测到任何摄像头！")
        return 0
    
    if len(available_cameras) == 1:
        print(f"只检测到一个摄像头 (索引 {available_cameras[0]})")
        return available_cameras[0]
    
    # 让用户选择一个摄像头
    print("\n请选择要使用的摄像头:")
    for i, info in enumerate(camera_info):
        print(f"{i+1}. {info}")
    
    try:
        choice = int(input(f"请输入选项 (1-{len(available_cameras)}): "))
        if 1 <= choice <= len(available_cameras):
            selected_index = available_cameras[choice-1]
            print(f"已选择摄像头 {selected_index}")
            return selected_index
        else:
            print(f"无效选择，使用默认摄像头 {available_cameras[0]}")
            return available_cameras[0]
    except ValueError:
        print(f"无效输入，使用默认摄像头 {available_cameras[0]}")
        return available_cameras[0]

def main():
    # 确保脚本文件存在
    ensure_standalone_trainer_exists()
    ensure_log_reader_exists()
    
    # 列出并选择摄像头
    camera_index = list_and_select_cameras()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_index}")
        return
    
    print(f"成功打开摄像头 {camera_index}")
    
    # 初始化識別器
    finger_recognizer = FingerObjectRecognizer()
    object_trainer = UltralyticsTrainer()
    
    # 訓練相關變量
    is_training_mode = False
    current_label = ""
    captures_count = 0
    training_in_progress = False
    log_file = "yolo_training_log.txt"
    
    # 檢查模型是否存在
    model_path = 'ultralytics_yolo/model/best.pt'
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Available labels: {object_trainer.labels}")
    
    # 檢查是否有訓練正在進行（通過日誌文件）
    if os.path.exists(log_file):
        # 檢查日誌文件最後修改時間
        log_mod_time = os.path.getmtime(log_file)
        current_time = time.time()
        # 如果日誌文件在過去10分鐘內被修改，可能有訓練正在進行
        if current_time - log_mod_time < 600:  # 10分鐘 = 600秒
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "=== YOLO訓練開始於" in content and "=== YOLO訓練結束於" not in content:
                    training_in_progress = True
                    print("檢測到訓練可能正在進行中，按 'v' 查看日誌")
    
    # 創建隱藏的Tkinter根窗口，用於文件對話框
    root = tk.Tk()
    root.withdraw()
    
    print("=== Finger Object Recognition System (Ultralytics YOLO Version) ===")
    print("Press 'm' to switch training/recognition mode")
    print("Press 'l' to set training label")
    print("Press 'c' to capture training image")
    print("Press 'u' to upload images for training")
    print("Press 'f' to import images from folder")
    print("Press 't' to train model")
    print("Press 'v' to view training logs")
    print("Press 'q' to exit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read camera frame")
            continue
        
        # 鏡像翻轉
        frame = cv2.flip(frame, 1)
        
        # 處理畫面（食指識別部分）
        output_frame = finger_recognizer.process_frame(frame)
        
        # 檢查訓練狀態
        if os.path.exists(log_file) and training_in_progress:
            # 檢查日誌是否包含訓練結束的標記
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "=== YOLO訓練結束於" in content:
                        print("訓練已完成，檢查結果")
                        training_in_progress = False
                        # 如果有新訓練的模型，嘗試加載
                        if os.path.exists(model_path):
                            print("發現訓練完成的模型，嘗試加載")
                            object_trainer.load_model()
            except Exception as e:
                print(f"檢查訓練狀態時出錯: {e}")
        
        # 在訓練模式下顯示訓練信息
        if is_training_mode:
            cv2.putText(output_frame, "Training Mode (YOLO)", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(output_frame, f"Label: {current_label}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, f"Captured: {captures_count}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示訓練狀態
            if training_in_progress:
                cv2.putText(output_frame, "Training in progress...", (10, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(output_frame, "Press 'v' to view logs", (10, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 如果有框選區域，修改文本
            if finger_recognizer.recognition_active and finger_recognizer.recognize_box:
                x1, y1, x2, y2 = finger_recognizer.recognize_box
                # 覆蓋原來的"Recognize Obj"文本
                cv2.rectangle(output_frame, (x1-2, y1-30), (x1+150, y1-5), (0,0,0), -1)
                cv2.putText(output_frame, "Press 'c' to capture", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        elif object_trainer.model is not None:
            # 顯示識別模式標記
            cv2.putText(output_frame, "Recognition Mode (YOLO)", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 如果在識別模式下且模型已訓練，顯示識別結果
            if finger_recognizer.recognition_active and finger_recognizer.recognize_box:
                x1, y1, x2, y2 = finger_recognizer.recognize_box
                roi = frame[y1:y2, x1:x2]
                
                # Debug info - 顯示ROI大小
                if roi.size > 0:
                    roi_text = f"ROI: {roi.shape[1]}x{roi.shape[0]}"
                    cv2.putText(output_frame, roi_text, (10, 160), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    label, confidence = object_trainer.predict(roi)
                    
                    # 始終顯示預測結果，即使置信度較低
                    cv2.rectangle(output_frame, (x1-2, y1-30), (x1+250, y1-5), (0,0,0), -1)
                    if label:
                        cv2.putText(output_frame, f"{label} ({confidence:.2f})", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 0, 255), 2)
                    else:
                        cv2.putText(output_frame, f"No detection ({confidence:.2f})", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (255, 0, 255), 2)
        
        # 顯示置信度閾值
        cv2.putText(output_frame, f"Confidence threshold: {object_trainer.conf_threshold:.2f}", 
                   (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 顯示結果
        cv2.imshow('Finger Object Recognition System (Ultralytics YOLO Version)', output_frame)
        
        # 處理按鍵
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):  # 退出
            break
        elif key == ord('m'):  # 切換模式
            is_training_mode = not is_training_mode
            print(f"{'Training' if is_training_mode else 'Recognition'} mode activated")
        elif key == ord('l') and is_training_mode:  # 設置標籤
            current_label = input("Enter object label: ")
            print(f"Current label set to: {current_label}")
        elif key == ord('c') and is_training_mode:  # 捕獲圖像
            if not finger_recognizer.recognize_box:
                print("Please select an object with your index fingers first")
                continue
            if not current_label:
                print("Please set a label first")
                continue
            
            # 提取框選區域
            x1, y1, x2, y2 = finger_recognizer.recognize_box
            roi = frame[y1:y2, x1:x2]
            
            # 保存訓練圖像
            count = object_trainer.save_image(roi, current_label, [0, 0, roi.shape[1], roi.shape[0]])
            captures_count = count
            print(f"Captured image {count} for '{current_label}'")
        elif key == ord('u') and is_training_mode:  # 上傳圖像
            # 暫停攝像頭
            cv2.waitKey(1)
            
            # 讓用戶選擇圖像文件
            filetypes = [
                ('Image files', '*.jpg;*.jpeg;*.png;*.bmp'),
                ('All files', '*.*')
            ]
            file_paths = filedialog.askopenfilenames(
                title="Select images to upload",
                filetypes=filetypes
            )
            
            if file_paths:
                # 如果未設置標籤，請用戶輸入
                if not current_label:
                    current_label = simpledialog.askstring("Label", "Enter object label:", parent=root)
                    if not current_label:
                        print("No label set, upload canceled")
                        continue
                
                # 上傳圖像進行訓練
                count = object_trainer.train_from_images(file_paths, current_label)
                if count > 0:
                    captures_count = sum(1 for f in os.listdir(object_trainer.images_train) if f.startswith(f"{current_label}_"))
                    print(f"Upload successful: {count} images with label '{current_label}'")
        elif key == ord('f') and is_training_mode:  # 從資料夾導入
            # 暫停攝像頭
            cv2.waitKey(1)
            
            # 讓用戶選擇資料夾
            folder_path = filedialog.askdirectory(
                title="Select folder containing images"
            )
            
            if folder_path:
                # 如果未設置標籤，請用戶輸入
                if not current_label:
                    current_label = simpledialog.askstring("Label", "Enter object label (leave empty to use folder name):", parent=root)
                
                # 導入資料夾中的圖像
                count = object_trainer.import_from_folder(folder_path, current_label)
                if count > 0:
                    used_label = current_label if current_label else os.path.basename(folder_path)
                    captures_count = sum(1 for f in os.listdir(object_trainer.images_train) if f.startswith(f"{used_label}_"))
                    print(f"Import successful: {count} images with label '{used_label}'")
                    if not current_label:
                        current_label = os.path.basename(folder_path)
        elif key == ord('t') and is_training_mode:  # 訓練模型
            if training_in_progress:
                print("訓練已在進行中，請等待完成或查看日誌")
                messagebox.showinfo("Training in Progress", "Training is already in progress. Press 'v' to view logs.")
                continue
            
            print("Starting YOLO model training...")
            
            # 使用獨立進程啟動訓練
            try:
                # 啟動訓練進程
                try:
                    # 嘗試使用CREATE_NEW_CONSOLE標誌（僅限Windows）
                    training_process = subprocess.Popen(
                        [sys.executable, "standalone_yolo_trainer.py", "ultralytics_yolo", log_file],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                except (AttributeError, TypeError):
                    # 非Windows系統或CREATE_NEW_CONSOLE不可用
                    training_process = subprocess.Popen(
                        [sys.executable, "standalone_yolo_trainer.py", "ultralytics_yolo", log_file],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                
                print(f"訓練進程已啟動 (PID: {training_process.pid})")
                print(f"訓練日誌將保存到: {log_file}")
                print("您可以繼續使用程序，訓練在後台進行")
                print("按 'v' 查看訓練日誌和狀態")
                
                # 標記訓練正在進行
                training_in_progress = True
                
                # 顯示消息框
                messagebox.showinfo("Training Started", 
                                    "YOLO training has started in a separate process.\n"
                                    "You can continue using the application.\n"
                                    "Press 'v' to view training logs and status.")
                
            except Exception as e:
                print(f"啟動訓練時出錯: {e}")
                import traceback
                traceback.print_exc()
        elif key == ord('v'):  # 查看訓練日誌
            # 啟動日誌查看器
            show_training_logs()
    
    cap.release()
    finger_recognizer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()