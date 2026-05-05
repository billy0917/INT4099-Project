def improved_predict(trainer, image):
    """改善的預測功能，提供更多調試資訊"""
    import cv2
    import numpy as np
    
    if trainer.model is None:
        print("模型未載入，無法進行預測")
        return None, 0.0
    
    # 檢查圖像是否為空
    if image is None or image.size == 0:
        print("圖像為空，無法預測")
        return None, 0.0
    
    # 檢查圖像尺寸是否合適
    h, w = image.shape[:2]
    if h < 20 or w < 20:
        print(f"圖像太小 ({w}x{h})，可能導致識別失敗")
    
    # 確保圖像至少是正方形的
    min_dim = min(h, w)
    if h > 3*w or w > 3*h:
        print("警告: 圖像比例極不平衡，可能影響識別")
        
    # 嘗試增強圖像
    try:
        # 調整亮度和對比度
        alpha = 1.2  # 對比度
        beta = 10    # 亮度
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 嘗試使用增強的圖像預測
        print(f"使用增強圖像預測 (亮度+{beta}, 對比度x{alpha})")
        results = trainer.model.predict(enhanced, conf=trainer.conf_threshold, verbose=False)
        
        # 獲取結果
        if len(results[0].boxes) == 0:
            print("增強圖像未檢測到物體，嘗試原始圖像...")
            # 如果增強的圖像沒有檢測到任何物體，嘗試原始圖像
            results = trainer.model.predict(image, conf=trainer.conf_threshold, verbose=False)
        
        # 獲取結果
        boxes = results[0].boxes
        if len(boxes) == 0:
            print(f"無檢測結果 (置信度閾值: {trainer.conf_threshold})")
            return None, 0.0
            
        # 獲取最高置信度的檢測
        max_conf_idx = boxes.conf.argmax().item()
        confidence = boxes.conf[max_conf_idx].item()
        class_id = int(boxes.cls[max_conf_idx].item())
        
        # 獲取對應的標籤名稱
        if class_id < len(trainer.labels):
            label = trainer.labels[class_id]
            print(f"檢測到: {label} (置信度: {confidence:.2f})")
        else:
            label = f"Class_{class_id}"
            print(f"檢測到未知類別 ID {class_id} (置信度: {confidence:.2f})")
        
        return label, confidence
        
    except Exception as e:
        print(f"預測錯誤: {e}")
        return None, 0.0