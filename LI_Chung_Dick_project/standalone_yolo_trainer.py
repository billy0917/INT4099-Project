"""
獨立的YOLO訓練腳本 - 可從主程序調用
使用方法: python standalone_yolo_trainer.py
"""

import os
import sys
import yaml
import time
import shutil
from pathlib import Path

def train_yolo(base_folder="ultralytics_yolo", log_file="training_log.txt"):
    """獨立訓練YOLO模型，並將結果保存到指定位置"""
    start_time = time.time()
    success = False
    
    print(f"開始YOLO模型訓練，日誌將保存到 {log_file}")
    
    with open(log_file, "w", encoding="utf-8") as log:
        # 記錄訓練開始
        log.write(f"=== YOLO訓練開始於 {time.ctime()} ===\n")
        
        try:
            # 設置路徑
            data_folder = os.path.join(base_folder, "data")
            model_folder = os.path.join(base_folder, "model")
            trained_model_path = os.path.join(model_folder, "best.pt")
            data_yaml_path = os.path.join(data_folder, "dataset.yaml")
            labels_path = os.path.join(model_folder, "labels.txt")
            
            # 檢查數據集配置文件
            if not os.path.exists(data_yaml_path):
                log.write(f"錯誤: 數據集配置文件不存在: {data_yaml_path}\n")
                return False
            
            # 檢查標籤文件
            if not os.path.exists(labels_path):
                log.write(f"錯誤: 標籤文件不存在: {labels_path}\n")
                return False
            
            # 載入標籤
            with open(labels_path, 'r', encoding="utf-8") as f:
                labels = [line.strip() for line in f.readlines()]
            
            log.write(f"已載入標籤: {labels}\n")
            
            # 檢查訓練圖像
            images_train = os.path.join(data_folder, "images", "train")
            images_val = os.path.join(data_folder, "images", "val")
            
            train_images_count = len(os.listdir(images_train)) if os.path.exists(images_train) else 0
            val_images_count = len(os.listdir(images_val)) if os.path.exists(images_val) else 0
            
            log.write(f"訓練圖像: {train_images_count}張, 驗證圖像: {val_images_count}張\n")
            
            if train_images_count + val_images_count < 5:
                log.write(f"錯誤: 訓練和驗證圖像總數過少: {train_images_count + val_images_count}\n")
                return False
            
            # 導入必要的庫
            try:
                from ultralytics import YOLO
                log.write("成功導入Ultralytics YOLO\n")
            except ImportError as e:
                log.write(f"錯誤: 無法導入Ultralytics模組: {e}\n")
                log.write("嘗試安裝Ultralytics...\n")
                
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    from ultralytics import YOLO
                    log.write("成功安裝並導入Ultralytics YOLO\n")
                except Exception as install_error:
                    log.write(f"錯誤: 安裝Ultralytics失敗: {install_error}\n")
                    return False
            
            # 檢查是否可以導入PyTorch
            try:
                import torch
                log.write(f"PyTorch版本: {torch.__version__}\n")
                if torch.cuda.is_available():
                    log.write(f"CUDA可用: {torch.cuda.get_device_name(0)}\n")
                else:
                    log.write("CUDA不可用，將使用CPU訓練（可能較慢）\n")
            except ImportError:
                log.write("警告: 無法導入PyTorch，訓練可能無法進行\n")
            
            # 執行訓練
            log.write("開始訓練模型...\n")
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
                
                log.write("模型訓練完成!\n")
                success = True
                
                # 檢查並複製模型文件
                trained_model = os.path.join(model_folder, 'train', 'weights', 'best.pt')
                if os.path.exists(trained_model):
                    log.write(f"找到訓練完成的模型: {trained_model}\n")
                    file_size_mb = os.path.getsize(trained_model) / (1024 * 1024)
                    log.write(f"模型文件大小: {file_size_mb:.2f} MB\n")
                    
                    # 複製到最終位置
                    try:
                        shutil.copy2(trained_model, trained_model_path)
                        log.write(f"已將模型複製到: {trained_model_path}\n")
                    except Exception as copy_error:
                        log.write(f"警告: 複製模型文件時出錯: {copy_error}\n")
                        log.write(f"請手動將 {trained_model} 複製到 {trained_model_path}\n")
                else:
                    log.write(f"警告: 找不到訓練完成的模型文件: {trained_model}\n")
                    # 嘗試查找last.pt
                    last_model = os.path.join(model_folder, 'train', 'weights', 'last.pt')
                    if os.path.exists(last_model):
                        log.write(f"找到last.pt模型: {last_model}\n")
                        try:
                            shutil.copy2(last_model, trained_model_path)
                            log.write(f"已將last.pt模型複製到: {trained_model_path}\n")
                        except Exception as copy_error:
                            log.write(f"警告: 複製last.pt文件時出錯: {copy_error}\n")
                    else:
                        log.write("錯誤: 未找到任何訓練輸出模型\n")
                        success = False
            
            except Exception as train_error:
                log.write(f"訓練過程中發生錯誤: {train_error}\n")
                import traceback
                log.write(traceback.format_exc())
                success = False
        
        except Exception as e:
            log.write(f"訓練準備過程中發生錯誤: {e}\n")
            import traceback
            log.write(traceback.format_exc())
            success = False
        
        # 記錄訓練結束
        end_time = time.time()
        duration = end_time - start_time
        log.write(f"=== YOLO訓練結束於 {time.ctime()} ===\n")
        log.write(f"總耗時: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)\n")
        log.write(f"訓練結果: {'成功' if success else '失敗'}\n")
    
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