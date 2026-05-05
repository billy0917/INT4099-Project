"""
YOLO訓練日誌查看器
提供實時監控訓練進度的GUI界面
"""

import os
import tkinter as tk
from tkinter import scrolledtext
import threading
import time

def show_training_status(log_file="yolo_training_log.txt"):
    """顯示YOLO訓練狀態與日誌"""
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
                            lines = content.split('\n')
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
                text_area.insert(tk.END, f"\n讀取日誌時出錯: {e}\n")
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