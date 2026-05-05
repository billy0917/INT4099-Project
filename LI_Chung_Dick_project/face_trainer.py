import os
import cv2
import numpy as np
import face_recognition
import pickle
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tkinter import Tk, filedialog, simpledialog, messagebox, Label, Button, Frame, Listbox, Scrollbar, VERTICAL, END
import threading
import shutil

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("face_trainer.log"),
                             logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ImageProcessing:
    """圖像處理工具類"""
    
    @staticmethod
    def normalize_brightness(image):
        """規範化圖像亮度"""
        # 轉換到 LAB 顏色空間
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 對 L 通道應用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 合併通道
        limg = cv2.merge((cl, a, b))
        
        # 轉回 RGB
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final


class FaceTrainer:
    """人臉訓練器"""
    
    def __init__(self):
        self.data_dir = "face_data"
        self.encodings_dir = os.path.join(self.data_dir, "encodings")
        
        # 確保目錄存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.encodings_dir, exist_ok=True)
        
        # 已知的人臉編碼和對應的名稱
        self.known_faces = {}  # 名稱 -> 編碼列表
        self.load_known_faces()
        
    def load_known_faces(self):
        """載入已知的人臉編碼"""
        if not os.path.exists(self.encodings_dir):
            return
            
        # 遍歷編碼目錄
        for filename in os.listdir(self.encodings_dir):
            if filename.endswith(".pkl"):
                name = os.path.splitext(filename)[0]
                encoding_file = os.path.join(self.encodings_dir, filename)
                
                try:
                    # 載入編碼
                    with open(encoding_file, 'rb') as f:
                        encodings = pickle.load(f)
                    self.known_faces[name] = encodings
                    logger.info(f"已載入 {name} 的 {len(encodings)} 個面部編碼")
                except Exception as e:
                    logger.error(f"載入編碼錯誤: {e}")
                    
        logger.info(f"共載入 {len(self.known_faces)} 個人的面部數據")
    
    def process_image(self, image_path, name):
        """處理單個圖像並提取人臉編碼"""
        try:
            # 讀取圖像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"無法讀取圖像: {image_path}")
                return None
                
            # 轉換為RGB（face_recognition使用RGB）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 預處理圖像
            processed_image = ImageProcessing.normalize_brightness(rgb_image)
            
            # 檢測人臉位置
            face_locations = face_recognition.face_locations(processed_image, model="hog")
            
            if not face_locations:
                logger.warning(f"圖像中未檢測到人臉: {image_path}")
                return None
                
            # 提取人臉編碼（只使用第一個檢測到的人臉）
            face_encodings = face_recognition.face_encodings(processed_image, [face_locations[0]])
            
            if not face_encodings:
                logger.warning(f"無法提取人臉編碼: {image_path}")
                return None
                
            # 獲取人臉範圍
            top, right, bottom, left = face_locations[0]
            
            # 保存處理後的人臉圖像以供參考
            face_image = processed_image[top:bottom, left:right]
            
            # 創建人臉目錄
            person_dir = os.path.join(self.data_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.jpg"
            output_path = os.path.join(person_dir, image_filename)
            
            # 保存人臉圖像
            cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            
            logger.info(f"已處理和保存人臉: {output_path}")
            
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"處理圖像時出錯 {image_path}: {e}")
            return None
    
    def process_image_folder(self, folder_path, name, ui_callback=None):
        """處理文件夾中的所有圖像"""
        if not os.path.exists(folder_path):
            logger.error(f"文件夾不存在: {folder_path}")
            return False
            
        # 獲取所有圖像文件
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
                    
        if not image_files:
            logger.warning(f"文件夾中沒有圖像: {folder_path}")
            return False
            
        logger.info(f"找到 {len(image_files)} 個圖像文件用於 {name}")
        
        # 處理每個圖像並收集編碼
        encodings = []
        processed_count = 0
        
        # 使用多進程處理圖像
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # 啟動所有圖像處理任務
            future_to_image = {
                executor.submit(self.process_image, image_path, name): image_path
                for image_path in image_files
            }
            
            # 收集結果
            total = len(future_to_image)
            for i, future in enumerate(future_to_image):
                encoding = future.result()
                if encoding is not None:
                    encodings.append(encoding)
                    processed_count += 1
                
                # 更新進度
                progress = (i + 1) / total * 100
                if ui_callback:
                    ui_callback(name, progress, f"處理中... {i+1}/{total}")
        
        # 檢查是否有足夠的編碼
        if processed_count == 0:
            logger.warning(f"沒有成功處理任何圖像用於 {name}")
            return False
            
        # 保存編碼
        if encodings:
            # 合併現有編碼
            if name in self.known_faces:
                encodings.extend(self.known_faces[name])
                logger.info(f"合併了 {len(self.known_faces[name])} 個現有編碼")
                
            self.known_faces[name] = encodings
            
            # 保存為pickle文件
            encoding_file = os.path.join(self.encodings_dir, f"{name}.pkl")
            with open(encoding_file, 'wb') as f:
                pickle.dump(encodings, f)
                
            # 也保存為numpy文件作為備份
            np_file = os.path.join(self.encodings_dir, f"{name}.npy")
            np.save(np_file, np.array(encodings))
            
            logger.info(f"已保存 {len(encodings)} 個編碼用於 {name}")
            
            if ui_callback:
                ui_callback(name, 100, f"完成! 處理了 {processed_count}/{len(image_files)} 張圖像")
                
            return True
        
        return False
    
    def remove_person(self, name):
        """刪除某人的所有編碼和訓練圖像"""
        if name not in self.known_faces:
            logger.warning(f"{name} 不在已知人臉中")
            return False
            
        # 刪除編碼文件
        pkl_file = os.path.join(self.encodings_dir, f"{name}.pkl")
        np_file = os.path.join(self.encodings_dir, f"{name}.npy")
        
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
        if os.path.exists(np_file):
            os.remove(np_file)
            
        # 刪除訓練圖像目錄
        person_dir = os.path.join(self.data_dir, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            
        # 從內存中移除
        del self.known_faces[name]
        
        logger.info(f"已刪除 {name} 的所有數據")
        return True
        
    def get_known_people(self):
        """獲取所有已知人名"""
        return list(self.known_faces.keys())
        
    def get_person_info(self, name):
        """獲取某人的編碼數量和訓練圖像數量"""
        if name not in self.known_faces:
            return 0, 0
            
        encoding_count = len(self.known_faces[name])
        
        # 計算訓練圖像數量
        image_count = 0
        person_dir = os.path.join(self.data_dir, name)
        if os.path.exists(person_dir):
            image_count = len([f for f in os.listdir(person_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                              
        return encoding_count, image_count


class FaceTrainerUI:
    """人臉訓練器界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("人臉識別預訓練工具")
        self.root.geometry("800x600")
        
        self.trainer = FaceTrainer()
        
        self.init_ui()
        self.refresh_person_list()
        
    def init_ui(self):
        """初始化界面"""
        # 主框架
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 左側框架 - 人員列表
        left_frame = Frame(main_frame, width=300)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # 右側框架 - 操作和信息
        right_frame = Frame(main_frame, width=500)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # 人員列表
        Label(left_frame, text="已訓練的人臉:").pack(anchor="w")
        
        list_frame = Frame(left_frame)
        list_frame.pack(fill="both", expand=True)
        
        self.person_listbox = Listbox(list_frame)
        self.person_listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = Scrollbar(list_frame, orient=VERTICAL)
        scrollbar.pack(side="right", fill="y")
        
        self.person_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.person_listbox.yview)
        
        # 左側按鈕
        left_btn_frame = Frame(left_frame)
        left_btn_frame.pack(fill="x", pady=5)
        
        Button(left_btn_frame, text="查看詳情", command=self.view_person_details).pack(side="left", padx=5)
        Button(left_btn_frame, text="刪除選中", command=self.delete_person).pack(side="left", padx=5)
        Button(left_btn_frame, text="刷新列表", command=self.refresh_person_list).pack(side="left", padx=5)
        
        # 右側 - 訓練控件
        Label(right_frame, text="添加新人臉訓練數據", font=("Arial", 12, "bold")).pack(anchor="w", pady=10)
        
        # 名稱輸入
        name_frame = Frame(right_frame)
        name_frame.pack(fill="x", pady=5)
        
        Label(name_frame, text="姓名:").pack(side="left")
        self.name_entry = simpledialog.askstring("輸入姓名", "請輸入要訓練的人的姓名:")
        if not self.name_entry:
            self.name_entry = "未命名"
        Label(name_frame, text=self.name_entry).pack(side="left", padx=5)
        Button(name_frame, text="更改姓名", command=self.change_name).pack(side="left", padx=5)
        
        # 訓練按鈕
        train_frame = Frame(right_frame)
        train_frame.pack(fill="x", pady=10)
        
        Button(train_frame, text="選擇圖像文件夾", command=self.select_folder).pack(side="left", padx=5)
        Button(train_frame, text="選擇多個圖像文件", command=self.select_files).pack(side="left", padx=5)
        
        # 狀態顯示
        self.status_label = Label(right_frame, text="就緒", anchor="w", justify="left")
        self.status_label.pack(fill="x", pady=10)
        
        # 操作說明
        help_text = """
        使用說明:
        1. 輸入要訓練的人的姓名
        2. 選擇包含該人臉部圖像的文件夾或多個圖像文件
        3. 系統會自動處理圖像，提取人臉特徵，並保存訓練數據
        4. 訓練數據將保存在face_data目錄中，可供人臉識別程序使用
        
        注意:
        - 每個人建議至少提供5張不同角度的臉部圖像
        - 圖像中應該只有一個人臉（被訓練者的人臉）
        - 圖像質量越好，識別效果越佳
        - 訓練完成後，您可以在人臉識別程序中直接使用這些數據
        """
        
        help_label = Label(right_frame, text=help_text, justify="left", anchor="w")
        help_label.pack(fill="x", pady=10)
        
    def change_name(self):
        """更改訓練的人名"""
        new_name = simpledialog.askstring("更改姓名", "請輸入新姓名:", initialvalue=self.name_entry)
        if new_name and new_name.strip():
            self.name_entry = new_name.strip()
            messagebox.showinfo("成功", f"已更改姓名為: {self.name_entry}")
            
    def select_folder(self):
        """選擇圖像文件夾"""
        folder_path = filedialog.askdirectory(title="選擇包含人臉圖像的文件夾")
        if not folder_path:
            return
            
        self.status_label.config(text=f"正在處理文件夾: {folder_path}")
        self.root.update()
        
        # 在新線程中處理圖像，避免UI凍結
        def process_thread():
            success = self.trainer.process_image_folder(
                folder_path, 
                self.name_entry,
                lambda name, progress, status: self.root.after(0, lambda: self.update_progress(name, progress, status))
            )
            
            if success:
                self.root.after(0, lambda: messagebox.showinfo("成功", 
                               f"已成功處理 {self.name_entry} 的人臉圖像"))
            else:
                self.root.after(0, lambda: messagebox.showerror("錯誤", 
                              f"處理 {self.name_entry} 的人臉圖像時遇到問題"))
                
            self.root.after(0, self.refresh_person_list)
                
        threading.Thread(target=process_thread).start()
        
    def select_files(self):
        """選擇多個圖像文件"""
        file_paths = filedialog.askopenfilenames(
            title="選擇人臉圖像", 
            filetypes=[("圖像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_paths:
            return
            
        # 創建臨時目錄
        temp_dir = os.path.join("temp", self.name_entry)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 複製文件到臨時目錄
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(temp_dir, filename))
            
        self.status_label.config(text=f"正在處理 {len(file_paths)} 個圖像文件")
        self.root.update()
        
        # 在新線程中處理圖像
        def process_thread():
            success = self.trainer.process_image_folder(
                temp_dir, 
                self.name_entry,
                lambda name, progress, status: self.root.after(0, lambda: self.update_progress(name, progress, status))
            )
            
            # 清理臨時目錄
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if success:
                self.root.after(0, lambda: messagebox.showinfo("成功", 
                               f"已成功處理 {self.name_entry} 的 {len(file_paths)} 個人臉圖像"))
            else:
                self.root.after(0, lambda: messagebox.showerror("錯誤", 
                              f"處理 {self.name_entry} 的人臉圖像時遇到問題"))
                
            self.root.after(0, self.refresh_person_list)
                
        threading.Thread(target=process_thread).start()
        
    def update_progress(self, name, progress, status):
        """更新進度信息"""
        self.status_label.config(text=f"{name}: {status} ({progress:.1f}%)")
        
    def refresh_person_list(self):
        """刷新人員列表"""
        self.person_listbox.delete(0, END)
        
        people = self.trainer.get_known_people()
        for name in people:
            encoding_count, image_count = self.trainer.get_person_info(name)
            self.person_listbox.insert(END, f"{name} ({encoding_count}編碼/{image_count}圖像)")
            
    def view_person_details(self):
        """查看人員詳情"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選擇一個人")
            return
            
        # 獲取姓名（去除計數部分）
        full_text = self.person_listbox.get(selection[0])
        name = full_text.split(" (")[0]
        
        encoding_count, image_count = self.trainer.get_person_info(name)
        
        detail_text = f"姓名: {name}\n"
        detail_text += f"編碼數量: {encoding_count}\n"
        detail_text += f"訓練圖像數量: {image_count}\n\n"
        
        if image_count > 0:
            person_dir = os.path.join(self.trainer.data_dir, name)
            detail_text += f"訓練圖像目錄: {os.path.abspath(person_dir)}"
            
        messagebox.showinfo(f"{name} 的詳細信息", detail_text)
        
    def delete_person(self):
        """刪除選中的人"""
        selection = self.person_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選擇一個人")
            return
            
        # 獲取姓名（去除計數部分）
        full_text = self.person_listbox.get(selection[0])
        name = full_text.split(" (")[0]
        
        confirm = messagebox.askyesno("確認", f"確定要刪除 {name} 的所有訓練數據嗎？")
        if confirm:
            success = self.trainer.remove_person(name)
            if success:
                messagebox.showinfo("成功", f"已刪除 {name} 的所有訓練數據")
                self.refresh_person_list()
            else:
                messagebox.showerror("錯誤", f"刪除 {name} 的訓練數據時出錯")


if __name__ == "__main__":
    root = Tk()
    app = FaceTrainerUI(root)
    root.mainloop()