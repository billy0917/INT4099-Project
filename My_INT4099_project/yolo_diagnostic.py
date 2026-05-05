import os
import sys
import subprocess
import platform
import importlib.util
import shutil
import tempfile

def check_ultralytics_installation():
    """檢查Ultralytics套件安裝情況"""
    print("== 檢查Ultralytics安裝 ==")
    try:
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
        
        # 檢查YOLO模型是否可以加載
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            print("成功加載基礎YOLOv8n模型")
        except Exception as e:
            print(f"無法加載YOLO模型: {e}")
        
        return True
    except ImportError:
        print("未安裝Ultralytics套件")
        return False
    except Exception as e:
        print(f"檢查Ultralytics安裝時出錯: {e}")
        return False

def check_system_resources():
    """檢查系統資源"""
    print("== 檢查系統資源 ==")
    import psutil
    
    # CPU信息
    print(f"CPU核心數: {os.cpu_count()}")
    print(f"CPU使用率: {psutil.cpu_percent()}%")
    
    # 記憶體信息
    mem = psutil.virtual_memory()
    print(f"總記憶體: {mem.total / (1024**3):.2f} GB")
    print(f"可用記憶體: {mem.available / (1024**3):.2f} GB")
    print(f"記憶體使用率: {mem.percent}%")
    
    # 磁盤信息
    disk = psutil.disk_usage('/')
    print(f"磁盤總空間: {disk.total / (1024**3):.2f} GB")
    print(f"磁盤可用空間: {disk.free / (1024**3):.2f} GB")
    print(f"磁盤使用率: {disk.percent}%")
    
    # GPU信息
    try:
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                print(f"找到CUDA設備: {torch.cuda.get_device_name(0)}")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            else:
                print("未找到CUDA設備")
        except ImportError:
            print("未安裝PyTorch")
        
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"找到TensorFlow GPU設備: {len(gpus)}個")
                for gpu in gpus:
                    print(f"  - {gpu.name}")
            else:
                print("TensorFlow未找到GPU設備")
        except ImportError:
            print("未安裝TensorFlow")
            
        if not gpu_available:
            print("警告: 未檢測到GPU。訓練可能會非常慢。")
    except Exception as e:
        print(f"檢查GPU時出錯: {e}")

def test_subprocess():
    """測試子進程功能"""
    print("== 測試子進程功能 ==")
    
    # 創建臨時文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # 嘗試使用子進程運行一個簡單的Python程序
        cmd = [sys.executable, "-c", "print('子進程測試成功')"]
        
        print("使用subprocess.Popen:")
        try:
            with open(temp_path, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                process.wait()
                print(f"退出代碼: {process.returncode}")
            
            with open(temp_path, 'r') as f:
                output = f.read().strip()
                print(f"輸出: {output}")
            
            if "子進程測試成功" in output:
                print("Popen測試通過")
            else:
                print("Popen測試失敗: 未找到預期輸出")
        except Exception as e:
            print(f"Popen測試失敗: {e}")
        
        print("\n使用subprocess.run:")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"退出代碼: {result.returncode}")
            print(f"輸出: {result.stdout}")
            
            if "子進程測試成功" in result.stdout:
                print("Run測試通過")
            else:
                print("Run測試失敗: 未找到預期輸出")
        except Exception as e:
            print(f"Run測試失敗: {e}")
    finally:
        # 刪除臨時文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def check_training_folders(base_folder="ultralytics_yolo"):
    """檢查訓練資料夾"""
    print("== 檢查訓練資料夾 ==")
    
    data_folder = os.path.join(base_folder, "data")
    model_folder = os.path.join(base_folder, "model")
    images_train = os.path.join(data_folder, "images", "train")
    images_val = os.path.join(data_folder, "images", "val")
    labels_train = os.path.join(data_folder, "labels", "train")
    labels_val = os.path.join(data_folder, "labels", "val")
    
    # 檢查資料夾是否存在
    folders = {
        "主資料夾": base_folder,
        "數據資料夾": data_folder,
        "模型資料夾": model_folder,
        "訓練圖像資料夾": images_train,
        "驗證圖像資料夾": images_val,
        "訓練標籤資料夾": labels_train,
        "驗證標籤資料夾": labels_val
    }
    
    for name, folder in folders.items():
        if os.path.exists(folder):
            print(f"{name} 存在: {folder}")
            if name in ["訓練圖像資料夾", "驗證圖像資料夾"]:
                try:
                    file_count = len(os.listdir(folder))
                    print(f"  - 包含 {file_count} 個文件")
                except Exception as e:
                    print(f"  - 無法讀取文件數量: {e}")
        else:
            print(f"{name} 不存在: {folder}")
    
    # 檢查標籤文件
    labels_path = os.path.join(model_folder, "labels.txt")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
                print(f"標籤文件存在，包含 {len(labels)} 個標籤: {labels}")
        except Exception as e:
            print(f"無法讀取標籤文件: {e}")
    else:
        print(f"標籤文件不存在: {labels_path}")
    
    # 檢查模型文件
    model_path = os.path.join(model_folder, "best.pt")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"模型文件存在: {model_path} ({size_mb:.2f} MB)")
    else:
        print(f"模型文件不存在: {model_path}")

def run_diagnostics():
    """運行所有診斷測試"""
    print("===== YOLO訓練診斷工具 =====")
    print(f"日期時間: {os.popen('date').read().strip()}")
    print(f"Python版本: {sys.version}")
    print(f"操作系統: {platform.platform()}")
    print("\n")
    
    check_ultralytics_installation()
    print("\n")
    
    check_system_resources()
    print("\n")
    
    test_subprocess()
    print("\n")
    
    check_training_folders()
    print("\n")
    
    print("診斷完成")

if __name__ == "__main__":
    run_diagnostics()