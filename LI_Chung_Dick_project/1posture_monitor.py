import cv2
import time
import math as m
import mediapipe as mp
import argparse
import winsound  # Import winsound for alert sounds

class PostureMonitor:
    def __init__(self, 
                 offset_threshold=100, 
                 neck_angle_threshold=25, 
                 torso_angle_threshold=10, 
                 time_threshold=180,
                 alert_threshold=2):  # Add new parameter for alert sound threshold (2 seconds)
        """
        Initialize PostureMonitor with thresholds for posture detection.
        
        Args:
            offset_threshold: Threshold for shoulder alignment
            neck_angle_threshold: Threshold for neck inclination angle
            torso_angle_threshold: Threshold for torso inclination angle
            time_threshold: Time threshold for triggering a posture alert
            alert_threshold: Time threshold for playing alert sound (in seconds)
        """
        self.offset_threshold = offset_threshold
        self.neck_angle_threshold = neck_angle_threshold
        self.torso_angle_threshold = torso_angle_threshold
        self.time_threshold = time_threshold
        self.alert_threshold = alert_threshold  # Add alert threshold
        
        # Initialize MediaPipe pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Frame counters
        self.good_frames = 0
        self.bad_frames = 0
        
        # Alert flag to prevent repeated alerts
        self.alert_triggered = False
        
        # Font and colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'blue': (255, 127, 0),
            'red': (50, 50, 255),
            'green': (127, 255, 0),
            'dark_blue': (127, 20, 0),
            'light_green': (127, 233, 100),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255),
            'white': (255, 255, 255)
        }
        
        # Video capture and pose detection
        self.cap = None
        self.pose = None
    
    def initialize(self, video_path=0):
        """
        Initialize video capture and pose detection.
        
        Args:
            video_path: Path to video file or camera index (default: 0 for webcam)
        """
        # Initialize mediapipe pose
        self.pose = self.mp_pose.Pose()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        
        # Reset frame counters
        self.good_frames = 0
        self.bad_frames = 0
        self.alert_triggered = False
    
    def findDistance(self, x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            x1, y1: Coordinates of the first point
            x2, y2: Coordinates of the second point
            
        Returns:
            Distance between the two points
        """
        dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist
    
    def findAngle(self, x1, y1, x2, y2):
        """
        Calculate the angle between two points with respect to the y-axis.
        
        Args:
            x1, y1: Coordinates of the first point
            x2, y2: Coordinates of the second point
            
        Returns:
            Angle in degrees
        """
        try:
            theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
            degree = int(180/m.pi) * theta
            return degree
        except:
            return 0
    
    def play_alert_sound(self):
        """Play alert sound for bad posture."""
        # Play a beep sound at 1000Hz for 500ms
        winsound.Beep(1000, 500)
    
    def detect_posture(self, frame):
        """
        Detect posture in a video frame.
        
        Args:
            frame: Video frame to analyze
            
        Returns:
            tuple: (processed_frame, posture_data)
                processed_frame: Frame with annotations
                posture_data: Dictionary with posture information
        """
        if frame is None:
            return None, None
            
        # Get height and width
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        keypoints = self.pose.process(frame_rgb)
        
        # If no landmarks detected, return original frame
        if not keypoints.pose_landmarks:
            return frame, None
        
        # Get landmark positions
        lm = keypoints.pose_landmarks
        lmPose = self.mp_pose.PoseLandmark
        
        # Left shoulder
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        
        # Left ear
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        
        # Left hip
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        
        # Calculate distance between shoulders
        offset = self.findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        
        # Calculate angles
        neck_inclination = self.findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = self.findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        
        # Determine if posture is good
        is_good_posture = (neck_inclination < self.neck_angle_threshold and 
                           torso_inclination < self.torso_angle_threshold)
        
        # Calculate fps
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30  # Default to 30 fps if unavailable
        
        # Update frame counters and handle alert
        if is_good_posture:
            self.bad_frames = 0
            self.good_frames += 1
            line_color = self.colors['green']
            text_color = self.colors['light_green']
            posture_quality = "good"
            
            # Reset alert flag when posture returns to good
            self.alert_triggered = False
        else:
            self.good_frames = 0
            self.bad_frames += 1
            line_color = self.colors['red']
            text_color = self.colors['red']
            posture_quality = "bad"
            
            # Calculate bad posture time
            bad_time = (1 / fps) * self.bad_frames
            
            # Trigger alert if bad posture exceeds alert threshold and alert hasn't been triggered yet
            if bad_time >= self.alert_threshold and not self.alert_triggered:
                self.play_alert_sound()
                self.alert_triggered = True  # Set flag to prevent repeated alerts
                cv2.putText(frame, "POSTURE ALERT!", (w//2 - 150, h//2), 
                           self.font, 1.2, self.colors['red'], 3)
        
        # Calculate time in current posture
        good_time = (1 / fps) * self.good_frames
        bad_time = (1 / fps) * self.bad_frames
        
        # Draw landmarks and lines
        # Shoulder alignment text
        if offset < self.offset_threshold:
            cv2.putText(frame, f"{int(offset)} Shoulders aligned", (w - 280, 30), 
                       self.font, 0.6, self.colors['green'], 2)
        else:
            cv2.putText(frame, f"{int(offset)} Shoulders not aligned", (w - 280, 30), 
                       self.font, 0.6, self.colors['red'], 2)
        
        # Draw circles at landmarks
        cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, self.colors['white'], 2)
        cv2.circle(frame, (l_ear_x, l_ear_y), 7, self.colors['white'], 2)
        cv2.circle(frame, (l_shldr_x, l_shldr_y - 100), 7, self.colors['white'], 2)
        cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, self.colors['pink'], -1)
        cv2.circle(frame, (l_hip_x, l_hip_y), 7, self.colors['yellow'], -1)
        cv2.circle(frame, (l_hip_x, l_hip_y - 100), 7, self.colors['yellow'], -1)
        
        # Draw angle text
        angle_text_neck = f'Neck inclination: {int(neck_inclination)}'
        angle_text_torso = f'Torso inclination: {int(torso_inclination)}'
        
        cv2.putText(frame, angle_text_neck, (10, 30), self.font, 0.6, text_color, 2)
        cv2.putText(frame, angle_text_torso, (10, 60), self.font, 0.6, text_color, 2)
        cv2.putText(frame, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), 
                   self.font, 0.9, text_color, 2)
        cv2.putText(frame, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), 
                   self.font, 0.9, text_color, 2)
        
        # Draw lines
        cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), line_color, 2)
        cv2.line(frame, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), line_color, 2)
        cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), line_color, 2)
        cv2.line(frame, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), line_color, 2)
        
        # Display posture time
        if good_time > 0:
            time_string = f'Good Posture Time : {round(good_time, 1)}s'
            cv2.putText(frame, time_string, (10, h - 20), self.font, 0.9, self.colors['green'], 2)
        else:
            time_string = f'Bad Posture Time : {round(bad_time, 1)}s'
            cv2.putText(frame, time_string, (10, h - 20), self.font, 0.9, self.colors['red'], 2)
        
        # Create posture data dictionary
        posture_data = {
            "posture_quality": posture_quality,
            "neck_inclination": neck_inclination,
            "torso_inclination": torso_inclination,
            "shoulder_offset": offset,
            "good_time": good_time,
            "bad_time": bad_time,
            "needs_warning": bad_time > self.time_threshold,
            "alert_triggered": self.alert_triggered
        }
        
        return frame, posture_data
    
    def release(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()

# For backwards compatibility with command line usage
def parse_arguments():
    parser = argparse.ArgumentParser(description='Posture Monitor with MediaPipe')
    parser.add_argument('--video', type=str, default=0, help='Path to the input video file. If not provided, the webcam will be used.')
    parser.add_argument('--offset-threshold', type=int, default=100, help='Threshold value for shoulder alignment.')
    parser.add_argument('--neck-angle-threshold', type=int, default=25, help='Threshold value for neck inclination angle.')
    parser.add_argument('--torso-angle-threshold', type=int, default=10, help='Threshold value for torso inclination angle.')
    parser.add_argument('--time-threshold', type=int, default=180, help='Time threshold for triggering a posture alert.')
    parser.add_argument('--alert-threshold', type=int, default=2, help='Time threshold for playing alert sound (in seconds).')
    return parser.parse_args()

# Main function for command line usage
def main():
    args = parse_arguments()
    
    print("Arguments:")
    print(f"Video: {args.video}")
    print(f"Offset Threshold: {args.offset_threshold}")
    print(f"Neck Angle Threshold: {args.neck_angle_threshold}")
    print(f"Torso Angle Threshold: {args.torso_angle_threshold}")
    print(f"Time Threshold: {args.time_threshold}")
    print(f"Alert Threshold: {args.alert_threshold}")
    
    # Create and initialize posture monitor
    monitor = PostureMonitor(
        offset_threshold=args.offset_threshold,
        neck_angle_threshold=args.neck_angle_threshold,
        torso_angle_threshold=args.torso_angle_threshold,
        time_threshold=args.time_threshold,
        alert_threshold=args.alert_threshold
    )
    monitor.initialize(args.video)
    
    try:
        while True:
            ret, frame = monitor.cap.read()
            if not ret:
                break
                
            # Flip the frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Detect posture
            processed_frame, posture_data = monitor.detect_posture(frame)
            
            # Display the frame
            cv2.imshow('MediaPipe Pose', processed_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        monitor.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()