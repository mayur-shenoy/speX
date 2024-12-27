import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import pytesseract
import os
import time
import mediapipe as mp
import logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spex_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class PerformanceMonitor:
    def __init__(self):
        self.fps_history = []
        self.start_time = time.time()
        
    def update(self, frame_time):
        self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
            
    def get_average_fps(self):
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

class TextDetector:
    def __init__(self):
        self.min_confidence = 60
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Tesseract: {e}")

    def detect(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            texts = []

            for i, conf in enumerate(data['conf']):
                try:
                    if float(conf) > self.min_confidence:
                        text = data['text'][i].strip()
                        if text:
                            x, y, w, h = (
                                data['left'][i],
                                data['top'][i],
                                data['width'][i],
                                data['height'][i]
                            )
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            texts.append({
                                "text": text,
                                "confidence": conf,
                                "position": (x, y, w, h)
                            })
                except ValueError:
                    continue

            return frame, texts
            
        except Exception as e:
            logger.error(f"Error in text detection: {e}")
            return frame, []

class ObjectDetector:
    def __init__(self):
        try:
            self.model = YOLO('yolov8n.pt')
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        try:
            if self.model is None:
                return frame, []
                
            results = self.model(frame)
            detections = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if float(box.conf[0]) < 0.5:
                        continue
                        
                    cls = int(box.cls[0])
                    name = self.model.names[cls]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 0, 0), 2)
                    
                    detections.append({
                        "name": name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2)
                    })
                    
            return frame, detections
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return frame, []


class ObjectTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
        self.max_lost_frames = 30
        
    def update(self, frame, detections):
        try:
            tracked_objects = []
            lost_ids = []
            current_time = time.time()
            
            # Update existing trackers
            for obj_id, tracker_info in self.trackers.items():
                success, bbox = tracker_info['tracker'].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    tracked_objects.append({
                        'id': obj_id,
                        'name': tracker_info['name'],
                        'bbox': (x, y, x+w, y+h),
                        'tracked': True,
                        'age': current_time - tracker_info['start_time']
                    })
                    tracker_info['lost_frames'] = 0
                else:
                    tracker_info['lost_frames'] += 1
                    if tracker_info['lost_frames'] > self.max_lost_frames:
                        lost_ids.append(obj_id)

            # Remove lost trackers
            for obj_id in lost_ids:
                del self.trackers[obj_id]

            # Initialize new trackers
            for det in detections:
                if 'bbox' in det:
                    x1, y1, x2, y2 = det['bbox']
                    tracker = cv2.TrackerCSRT_create()
                    success = tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                    if success:
                        self.trackers[self.next_id] = {
                            'tracker': tracker,
                            'name': det['name'],
                            'lost_frames': 0,
                            'start_time': current_time
                        }
                        self.next_id += 1

            return tracked_objects
            
        except Exception as e:
            print(f"Error in object tracking: {e}")
            return []

class SceneAnalyzer:
    def __init__(self):
        self.last_scene = {}
        self.distance_threshold = 1.0
        self.hazard_objects = ["knife", "scissors", "fire", "stairs", "hole", "sharp", "hot"]
        self.important_objects = ["door", "chair", "table", "window", "phone", "person"]
        self.last_warning_time = {}
        self.warning_cooldown = 3.0
        
    def analyze_scene(self, detections, frame_dims):
        try:
            objects = {}
            scene_desc = []
            hazards = []
            navigation_points = []
            current_time = time.time()
            
            for det in detections:
                if 'name' in det:
                    bbox = det.get('bbox') or det.get('position')
                    if bbox:
                        name = det['name'].lower()
                        dist = self.estimate_distance(bbox, frame_dims)
                        direction = self.get_direction(bbox, frame_dims)
                        priority = self.get_object_priority(name, dist)
                        
                        objects[name] = {
                            'distance': dist,
                            'direction': direction,
                            'bbox': bbox,
                            'priority': priority
                        }
                        
                        # Handle hazards with cooldown
                        if name in self.hazard_objects and self.should_warn(name, current_time, dist):
                            hazards.append(f"Caution: {name} {direction} at {dist:.1f} meters")
                            self.last_warning_time[name] = current_time
                        
                        # Track important objects
                        if name in self.important_objects:
                            navigation_points.append(f"{name} {direction}")
            
            # Build prioritized description
            if hazards:
                scene_desc.extend(hazards)
            
            if navigation_points:
                scene_desc.extend(navigation_points)
            
            # Report significant changes
            for obj_name, info in objects.items():
                if self.is_significant_change(obj_name, info):
                    movement = self.describe_movement(obj_name, info)
                    if movement:
                        scene_desc.append(movement)
            
            self.last_scene = objects
            return ". ".join(scene_desc) if scene_desc else "Clear path ahead"
            
        except Exception as e:
            logger.error(f"Error in scene analysis: {e}")
            return "Error analyzing scene"

    def estimate_distance(self, bbox, frame_dims):
        """Estimate distance based on bounding box size"""
        _, _, w, h = bbox
        frame_height = frame_dims[0]
        # Rough estimate - can be calibrated for better accuracy
        return frame_height / (h + 0.0001)

    def get_direction(self, bbox, frame_dims):
        """Get directional position of object in frame"""
        x, y, w, h = bbox
        center_x = x + w/2
        frame_width = frame_dims[1]
        
        # Divide frame into thirds
        if center_x < frame_width/3:
            return "left"
        elif center_x > 2*frame_width/3:
            return "right"
        else:
            return "center"

    def should_warn(self, obj_name, current_time, distance):
        last_warning = self.last_warning_time.get(obj_name, 0)
        return (current_time - last_warning > self.warning_cooldown and 
                distance < self.distance_threshold)

    def get_object_priority(self, name, distance):
        if name in self.hazard_objects:
            return 1
        if name in self.important_objects:
            return 2
        return 3 + distance

    def is_significant_change(self, obj_name, current_info):
        if obj_name not in self.last_scene:
            return True
        old_info = self.last_scene[obj_name]
        dist_change = abs(current_info['distance'] - old_info['distance'])
        return dist_change > 0.5

    def describe_movement(self, obj_name, current_info):
        if obj_name not in self.last_scene:
            return None
        old_info = self.last_scene[obj_name]
        if current_info['distance'] < old_info['distance']:
            return f"{obj_name} moving closer"
        if current_info['distance'] > old_info['distance']:
            return f"{obj_name} moving away"
        return None


class FaceModule:
   def __init__(self):
       self.known_faces = {}
       self.known_face_encodings = []
       self.known_face_names = []
       self.faces_dir = os.path.join(os.getcwd(), "data", "faces")
       os.makedirs(self.faces_dir, exist_ok=True)
       self.load_known_faces()

   def load_known_faces(self):
       if not os.path.exists(self.faces_dir):
           print("No faces directory found.")
           return

       for file in os.listdir(self.faces_dir):
           if file.endswith('.npy'):
               name = os.path.splitext(file)[0]
               face_path = os.path.join(self.faces_dir, file)
               try:
                   face_encoding = np.load(face_path)
                   self.add_face(name, face_encoding, save=False)
                   print(f"Loaded face data for {name}")
               except Exception as e:
                   print(f"Error loading face {file}: {e}")

   def save_face(self, name, face_encoding):
       face_path = os.path.join(self.faces_dir, f"{name}.npy")
       try:
           np.save(face_path, face_encoding)
           print(f"Saved face data for {name}")
       except Exception as e:
           print(f"Error saving face: {e}")

   def add_face(self, name, face_encoding, save=True):
       self.known_face_encodings.append(face_encoding)
       self.known_face_names.append(name)
       self.known_faces[name] = face_encoding
       if save:
           self.save_face(name, face_encoding)

#    def learn_face(self, frame, name):
#        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        face_locations = face_recognition.face_locations(rgb_frame)
       
#        if face_locations:
#            face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]
#            self.add_face(name, face_encoding)
#            return True
#        return False

   
   def learn_face_enhanced(self, frame, name):
       try:
           rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           face_locations = face_recognition.face_locations(rgb_frame)
           
           if not face_locations:
               return False, "No face detected"
           if len(face_locations) > 1:
               return False, "Multiple faces detected"
               
           # Quality check
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
           if blur_value < 100:
               return False, "Image too blurry"

           face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]
           
           # Check if face already exists
           if self.known_face_encodings:
               matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
               if True in matches:
                   existing_name = self.known_face_names[matches.index(True)]
                   return False, f"Face already exists as {existing_name}"

           self.add_face(name, face_encoding)
           return True, "Face learned successfully"
       except Exception as e:
           print(f"Error learning face: {e}")
           return False, "Error during face learning"
       

   def recognize_faces(self, frame):
       small_frame = cv2.resize(frame, (480, 360))
       rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

       face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
       if not face_locations:
           return frame, []

       face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
       face_detections = []

       scale_x = frame.shape[1] / 480
       scale_y = frame.shape[0] / 360

       for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
           name = "Unknown"
           if self.known_face_encodings:
               matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
               if True in matches:
                   name = self.known_face_names[matches.index(True)]

           scaled_left = int(left * scale_x)
           scaled_top = int(top * scale_y)
           scaled_right = int(right * scale_x)
           scaled_bottom = int(bottom * scale_y)

           cv2.rectangle(frame, (scaled_left, scaled_top), 
                       (scaled_right, scaled_bottom), (0, 255, 0), 2)
           cv2.putText(frame, name, (scaled_left, scaled_top - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

           face_detections.append({
               "name": name,
               "bbox": (scaled_left, scaled_top, scaled_right, scaled_bottom)
           })

       return frame, face_detections

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_history = []  # For gesture smoothing

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self.detect_gesture(hand_landmarks)
                if gesture:
                    gestures.append(gesture)
                    # Add visualization
                    h, w, _ = frame.shape
                    x = int(hand_landmarks.landmark[0].x * w)
                    y = int(hand_landmarks.landmark[0].y * h)
                    cv2.putText(frame, f"Gesture: {gesture}", 
                              (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        return frame, gestures

    def detect_gesture(self, hand_landmarks):
        try:
            fingers_extended = self.get_finger_states(hand_landmarks)
            palm_direction = self.get_palm_direction(hand_landmarks)
            
            # Improved gesture detection logic
            if all(fingers_extended):
                return "open_palm" if palm_direction == "up" else "stop"
                
            elif not any(fingers_extended[1:]) and fingers_extended[0]:
                return "thumbs_up" if palm_direction == "up" else "thumbs_down"
                
            elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):
                return "peace"
                
            elif not any(fingers_extended):
                return "fist"
                
            elif fingers_extended[1] and not any(fingers_extended[2:]):
                return "pointing"
                
            return None
        except Exception as e:
            print(f"Error detecting gesture: {e}")
            return None

    def get_finger_states(self, landmarks):
        try:
            fingers = []
            
            # Thumb (using angle calculation)
            thumb_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y])
            thumb_ip = np.array([landmarks.landmark[3].x, landmarks.landmark[3].y])
            thumb_mcp = np.array([landmarks.landmark[2].x, landmarks.landmark[2].y])
            
            angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)
            fingers.append(angle > 150)  # Extended if angle > 150 degrees
            
            # Other fingers
            for tip_id in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky tips
                tip_y = landmarks.landmark[tip_id].y
                pip_y = landmarks.landmark[tip_id - 2].y  # PIP joint
                fingers.append(tip_y < pip_y - 0.04)  # Added threshold
                
            return fingers
        except Exception as e:
            print(f"Error getting finger states: {e}")
            return [False] * 5

    def get_palm_direction(self, landmarks):
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        
        if wrist.y > middle_mcp.y:
            return "up"
        elif wrist.y < middle_mcp.y:
            return "down"
        return "forward"

    def calculate_angle(self, p1, p2, p3):
        try:
            v1 = p1 - p2
            v2 = p3 - p2
            
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            
            return np.degrees(angle)
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0
           

    def get_navigation_command(self, gestures, detections):
        if not gestures:
            return None
        current_gesture = gestures[0]
        commands = {
            'open_palm': 'proceed',
            'fist': 'stop',
            'peace': 'turn',
            'thumbs up': 'confirm'
        }
        return commands.get(current_gesture)

    def _is_fist(self, landmarks):
        return all(landmarks.landmark[i].y > landmarks.landmark[i-3].y for i in [8,12,16,20])

    def _is_peace(self, landmarks):
        index_up = landmarks.landmark[8].y < landmarks.landmark[5].y
        middle_up = landmarks.landmark[12].y < landmarks.landmark[9].y
        others_down = all(landmarks.landmark[i].y > landmarks.landmark[i-3].y for i in [16,20])
        return index_up and middle_up and others_down

    def _is_thumbs_up(self, landmarks):
        thumb_up = landmarks.landmark[4].y < landmarks.landmark[2].y
        fingers_down = all(landmarks.landmark[i].y > landmarks.landmark[i-3].y for i in [8,12,16,20])
        return thumb_up and fingers_down

    def _is_open_palm(self, landmarks):
        return all(landmarks.landmark[i].y < landmarks.landmark[i-3].y for i in [8,12,16,20])


class VisionModule:
    def __init__(self):
        self.face_module = FaceModule()
        self.object_detector = ObjectDetector()
        self.text_detector = TextDetector()
        self.gesture_detector = GestureDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.object_tracker = ObjectTracker()
        self.performance_monitor = PerformanceMonitor()

    def detect_faces(self, frame):
        return self.face_module.recognize_faces(frame)

    def detect_objects(self, frame):
        return self.object_detector.detect(frame)

    def detect_text(self, frame):
        return self.text_detector.detect(frame)

    def detect_gestures(self, frame):
        return self.gesture_detector.detect(frame)

    def analyze_scene(self, detections, frame_dims):
        return self.scene_analyzer.analyze_scene(detections, frame_dims)

    def track_objects(self, frame, detections):
        tracked_objects = self.object_tracker.update(frame, detections)
        
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{obj['name']} ID:{obj['id']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame, tracked_objects

    def process_frame_complete(self, frame, mode=None):
        try:
            start_time = time.time()
            results = {
                'frame': frame,
                'faces': [],
                'objects': [],
                'text': [],
                'gestures': [],
                'navigation': None
            }

            # Always process faces and gestures
            frame, faces = self.face_module.recognize_faces(frame)
            results['faces'] = faces
            
            # Always process gestures (lightweight)
            frame, gestures = self.gesture_detector.detect(frame)
            results['gestures'] = gestures

            # Process mode-specific detections
            if mode == "objects":
                frame, objects = self.object_detector.detect(frame)
                results['objects'] = objects
                
                # Update object tracking
                frame, tracked_objects = self.track_objects(frame, objects)
                
                # Analyze scene with tracked objects
                scene_description = self.scene_analyzer.analyze_scene(tracked_objects, frame.shape[:2])
                results['navigation'] = scene_description
                
            elif mode == "text":
                frame, texts = self.text_detector.detect(frame)
                results['text'] = texts

            # Performance monitoring
            frame_time = time.time() - start_time
            self.performance_monitor.update(frame_time)
            avg_fps = self.performance_monitor.get_average_fps()
            
            # Display information
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if gestures:
                cv2.putText(frame, f"Gestures: {', '.join(gestures)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)

            results['frame'] = frame
            return results
        
        except Exception as e:
            logger.error(f"Error in processing frame: {e}")
            return {
                'frame': frame,
                'faces': [],
                'objects': [],
                'text': [],
                'gestures': [],
                'navigation': "Error in processing"
            }
        