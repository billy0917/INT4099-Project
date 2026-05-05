import cv2
import mediapipe as mp
import numpy as np
import time

class FingerObjectRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Parameters for recognition
        self.still_time_required = 2.0  # seconds
        self.movement_threshold = 10  # pixels
        self.index_finger_positions = []
        self.last_movement_time = time.time()
        self.recognition_active = False
        self.recognize_box = None
        
        # Animation parameters
        self.recognition_start_time = None
        self.animation_radius = 30
        self.animation_thickness = 5
        
        # Colors for visualization
        self.colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'pink': (255, 0, 255),
            'cyan': (255, 255, 0)
        }

    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Process the frame to detect hands
        results = self.hands.process(rgb_frame)
        
        # Create a copy for drawing
        output_frame = frame.copy()
        
        # Track index finger positions
        index_fingers = []
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            # Display hand count
            cv2.putText(output_frame, f"Hand Num: {hand_count}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['blue'], 2)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Get index finger tip position (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Draw a circle at the index finger tip
                cv2.circle(output_frame, (index_x, index_y), 10, self.colors['blue'], -1)
                
                # Add to our list of index fingers
                index_fingers.append((index_x, index_y))
                
                # Draw colored trajectories (similar to the images)
                # This is a simplified version - in a real app you'd track history
                for i, connection in enumerate(self.mp_hands.HAND_CONNECTIONS):
                    color = [self.colors['red'], self.colors['green'], 
                             self.colors['blue'], self.colors['yellow']][i % 4]
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_pos = hand_landmarks.landmark[start_idx]
                    end_pos = hand_landmarks.landmark[end_idx]
                    start_point = (int(start_pos.x * w), int(start_pos.y * h))
                    end_point = (int(end_pos.x * w), int(end_pos.y * h))
                    cv2.line(output_frame, start_point, end_point, color, 2)
        
        # Process two index fingers for recognition
        if len(index_fingers) == 2:
            # Check if fingers have moved significantly
            if self.index_finger_positions:
                prev_left, prev_right = self.index_finger_positions
                curr_left, curr_right = index_fingers
                
                # Calculate movement
                left_movement = np.sqrt((curr_left[0] - prev_left[0])**2 + (curr_left[1] - prev_left[1])**2)
                right_movement = np.sqrt((curr_right[0] - prev_right[0])**2 + (curr_right[1] - prev_right[1])**2)
                
                if left_movement > self.movement_threshold or right_movement > self.movement_threshold:
                    # Reset timer if there's significant movement
                    self.last_movement_time = time.time()
                    self.recognition_start_time = None
                    self.recognition_active = False
                    self.recognize_box = None
                else:
                    # Hands are still
                    if self.recognition_start_time is None:
                        # Start the recognition timer
                        self.recognition_start_time = time.time()
                    
                    # Calculate the elapsed time
                    elapsed_time = time.time() - self.recognition_start_time
                    
                    # Draw the progress animation on both index fingers
                    progress = min(1.0, elapsed_time / self.still_time_required)
                    
                    # Draw progress circles on index fingers
                    for finger_pos in index_fingers:
                        # Draw blue circle as base
                        cv2.circle(output_frame, finger_pos, self.animation_radius, self.colors['blue'], 2)
                        
                        # Draw progress arc
                        if progress > 0:
                            start_angle = -90  # Start from top
                            end_angle = start_angle + (360 * progress)
                            cv2.ellipse(output_frame, finger_pos, 
                                      (self.animation_radius, self.animation_radius),
                                      0, start_angle, end_angle, self.colors['green'], 
                                      self.animation_thickness)
                    
                    # Add a visual connection between fingers when they're still
                    cv2.line(output_frame, index_fingers[0], index_fingers[1], 
                           self.colors['yellow'], 2)
                    
                    # Activate recognition if fingers have been still long enough
                    if not self.recognition_active and elapsed_time > self.still_time_required:
                        self.recognition_active = True
                        
                        # Create recognition box between the two index fingers
                        left_x, left_y = min(index_fingers[0][0], index_fingers[1][0]), min(index_fingers[0][1], index_fingers[1][1])
                        right_x, right_y = max(index_fingers[0][0], index_fingers[1][0]), max(index_fingers[0][1], index_fingers[1][1])
                        
                        # Add some padding to the box
                        padding = 20
                        left_x -= padding
                        left_y -= padding
                        right_x += padding
                        right_y += padding
                        
                        # Ensure coordinates are within frame
                        left_x = max(0, left_x)
                        left_y = max(0, left_y)
                        right_x = min(w, right_x)
                        right_y = min(h, right_y)
                        
                        self.recognize_box = (left_x, left_y, right_x, right_y)
            
            # Update finger positions
            self.index_finger_positions = index_fingers
        else:
            # Reset if we don't have exactly two fingers
            self.index_finger_positions = []
            self.recognition_active = False
            self.recognize_box = None
            self.last_movement_time = time.time()
            self.recognition_start_time = None
        
        # Draw recognition box if active
        if self.recognition_active and self.recognize_box:
            x1, y1, x2, y2 = self.recognize_box
            # Draw pink rectangle
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), self.colors['pink'], 2)
            # Add "Recognize Obj" text
            cv2.putText(output_frame, "Recognize Obj", (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['pink'], 2)
        
        return output_frame

    def release(self):
        self.hands.close()

def main():
    cap = cv2.VideoCapture(0)  # Use webcam (you may need to change the index)
    recognizer = FingerObjectRecognizer()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Mirror the frame horizontally for a more intuitive experience
        frame = cv2.flip(frame, 1)
        
        # Process the frame
        output_frame = recognizer.process_frame(frame)
        
        # Display the result
        cv2.imshow('Index Finger Object Recognition', output_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    recognizer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()