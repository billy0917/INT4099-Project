def diagnose_model(trainer):
    """診斷模型並列印詳細資訊"""
    import os
    
    print("\n=== YOLO模型診斷 ===")
    model_path = trainer.trained_model_path
    print(f"模型路徑: {model_path}")
    print(f"模型存在: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"模型檔案大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    print(f"標籤數量: {len(trainer.labels)}")
    print(f"標籤列表: {trainer.labels}")
    
    # 檢查訓練資料
    train_imgs = len(os.listdir(trainer.images_train))
    val_imgs = len(os.listdir(trainer.images_val))
    print(f"訓練圖像: {train_imgs}, 驗證圖像: {val_imgs}")
    
    if trainer.model is None:
        print("模型未載入! 請確認訓練是否成功")
    else:
        print("模型已載入! 嘗試使用測試圖像")
        
        # 創建測試圖像
        import numpy as np
        import cv2
        test_img = np.ones((300, 300, 3), dtype=np.uint8) * 128  # 灰色測試圖像
        
        try:
            results = trainer.model.predict(test_img, conf=0.01, verbose=False)  # 使用極低的置信度閾值
            boxes = results[0].boxes
            if len(boxes) > 0:
                print(f"測試圖像檢測到 {len(boxes)} 個物體")
                print(f"最高置信度: {boxes.conf.max().item():.4f}")
            else:
                print("測試圖像未檢測到物體 - 這是正常的")
            print("預測功能正常運作!")
        except Exception as e:
            print(f"預測測試失敗: {e}")
    
    print("=== 診斷完成 ===\n")