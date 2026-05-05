# Intelligent Vision-Based Monitoring & Attendance System (INT4099 Project)

This project is a comprehensive desktop application built with Python and PySide6 that acts as a smart student or office monitoring assistant. It integrates multiple AI and computer vision models to provide real-time attendance tracking, posture evaluation, concentration monitoring, and custom object detection.

## ✨ Key Features

- **Facial Recognition & Auto-Attendance** 
  Uses `face_recognition` and OpenCV to identify enrolled users in real-time and automatically logs their daily attendance into structured `.csv` reports.
- **Real-time Posture & Focus Monitoring** 
  Utilizes `MediaPipe` pose estimation to track skeletal landmarks. By calculating precise spatial geometries (e.g., neck inclination and torso angles), the system evaluates posture health and uses timers with audio alerts (`winsound`) to prevent distraction and slouching.
- **Custom Object & Pose Detection** 
  Integrates **YOLOv8** (`ultralytics`) for custom object detection and YOLO-Pose tracking. The repository includes custom training scripts (`ultralytics_yolo_trainer.py`) used to fine-tune the model on specific datasets.
- **Responsive GUI Architecture** 
  Features a complete desktop application interface built with **PySide6**. By decoupling the intensive camera and AI processing streams into separate threads (Multi-threading), the system guarantees a smooth, non-blocking user experience.

## 🛠️ Tech Stack

- **Programming Language**: Python 3.x
- **Computer Vision & AI**: OpenCV, MediaPipe, Ultralytics YOLOv8, `face_recognition` (dlib)
- **Frontend / GUI**: PySide6 (Qt framework)
- **Concurrency**: Python `threading` & `QThread`
- **Data Management**: CSV, JSON, NumPy arrays

## 📄 License and Disclaimer
This repository is an academic project originally created for INT4099.
