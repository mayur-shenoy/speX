import cv2
import time
from modules.audio import AudioModule
from modules.vision import VisionModule
from modules.basic_features import BasicFeaturesModule
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spex_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpexAssistant:
    def __init__(self):
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.vision = VisionModule()
            self.audio = AudioModule()
            self.basic_features = BasicFeaturesModule()
            self.fps_limit = 30
            self.current_mode = None
            
            self.mode_triggers = {
                "faces": ["face", "faces", "facial", "recognize", "who"],
                "objects": ["object", "objects", "detect", "thing", "things", "what", "see"],
                "text": ["text", "read", "reading", "written", "write", "ocr"],
                "gestures": ["gesture", "gestures", "hand", "hands", "motion"]
            }
            
            self.basic_triggers = {
                "time": ["time", "hour", "clock"],
                "weather": ["weather", "temperature", "forecast", "outside"],
                "location": ["where am i", "location", "place", "city"]
            }
            
            self.navigation_triggers = ["navigate", "guide", "where is", "take me"]
            logger.info("SpexAssistant initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SpexAssistant: {e}")
            raise

    def start(self):
        logger.info("Starting Spex Assistant")
        self.audio.speak("Hello, I am Spex, your companion. I'm ready to help you.")
        try:
            self.main_loop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up resources")
        self.camera.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        try:
            results = self.vision.process_frame_complete(frame, self.current_mode)
            
            frame = results['frame']
            detections = []
            
            # Combine all detections
            detections.extend(results['faces'])
            detections.extend(results['objects'])
            detections.extend(results['text'])
            
            # Handle gestures
            if results['gestures']:
                nav_command = self.vision.gesture_detector.get_navigation_command(
                    results['gestures'], detections)
                if nav_command:
                    self.audio.speak(nav_command)

            return frame, detections, results
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return frame, [], {'frame': frame, 'faces': [], 'objects': [], 'text': [], 'gestures': [], 'navigation': None}

    def describe_scene(self, frame, detections):
        try:
            scene_description = self.vision.scene_analyzer.analyze_scene(detections, frame.shape[:2])
            basic_description = self.describe_detections(detections)
            
            return f"{basic_description}. {scene_description}"
        except Exception as e:
            logger.error(f"Error in describe_scene: {e}")
            return "I'm having trouble analyzing the scene"

    def describe_detections(self, detections):
        if not detections:
            return "I don't see anything notable right now"

        try:
            description_parts = []
            faces = [d for d in detections if "name" in d]
            objects = [d for d in detections if "confidence" in d]
            texts = [d for d in detections if "text" in d]

            if faces:
                known_faces = [f for f in faces if f['name'] != "Unknown"]
                unknown_faces = [f for f in faces if f['name'] == "Unknown"]

                if known_faces:
                    names = ", ".join(set(f['name'] for f in known_faces))
                    description_parts.append(f"I recognize {names}")
                if unknown_faces:
                    description_parts.append(
                        f"I see {len(unknown_faces)} unknown {'person' if len(unknown_faces) == 1 else 'people'}")

            if objects:
                obj_counts = {}
                for obj in objects:
                    obj_counts[obj['name']] = obj_counts.get(obj['name'], 0) + 1
                obj_desc = ", ".join(f"{count} {name}" for name, count in obj_counts.items())
                description_parts.append(f"I see {obj_desc}")

            if texts:
                unique_texts = set(t['text'] for t in texts)
                if len(unique_texts) <= 3:
                    description_parts.append(f"I can read: {', '.join(unique_texts)}")
                else:
                    description_parts.append("I can see some text")

            return ". ".join(description_parts)
        except Exception as e:
            logger.error(f"Error in describe_detections: {e}")
            return "Error processing detections"

    def handle_mode_command(self, command):
        try:
            if any(phrase in command for phrase in ["modes off", "mode off", "turn off", "disable"]):
                self.current_mode = None
                self.audio.speak("All modes deactivated")
                return True

            mode_words = ["mode", "switch to", "change to"]
            if any(word in command for word in mode_words):
                for mode, triggers in self.mode_triggers.items():
                    if any(trigger in command for trigger in triggers):
                        self.current_mode = mode
                        self.audio.speak(f"{mode} mode activated")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error in handle_mode_command: {e}")
            return False

    def handle_face_learning(self, frame):
        try:
            self.audio.speak("Who should I learn?")
            name = self.audio.listen()
            if name:
                success, msg = self.vision.face_module.learn_face_enhanced(frame, name)
                if success:
                    self.audio.speak(f"I've learned {name}'s face. Please turn your head slightly")
                    time.sleep(2)
                    ret, frame = self.camera.read()
                    if ret:
                        success, _ = self.vision.face_module.learn_face_enhanced(frame, name)
                        if success:
                            self.audio.speak("Face learning complete")
                        else:
                            self.audio.speak("Couldn't get a clear view of your face from another angle")
                else:
                    self.audio.speak(msg)
        except Exception as e:
            logger.error(f"Error in handle_face_learning: {e}")
            self.audio.speak("Sorry, there was an error learning the face")

    def handle_command(self, command, frame, detections, results):
        try:
            if "stop" in command:
                self.audio.speak("Goodbye!")
                return True
                
            elif any(word in command for word in ["mode", "switch", "change"]):
                self.handle_mode_command(command)
                
            elif "what do you see" in command or "describe scene" in command:
                description = self.describe_scene(frame, detections)
                self.audio.speak(description)
                
            elif "learn face" in command:
                self.handle_face_learning(frame)
                
            elif any(nav in command for nav in self.navigation_triggers):
                if results['navigation']:
                    self.audio.speak(results['navigation'])
                else:
                    nav_description = self.vision.scene_analyzer.analyze_scene(
                        detections, frame.shape[:2])
                    self.audio.speak(nav_description)
                    
            elif "read" in command and results['text']:
                text_content = " ".join(d['text'] for d in results['text'])
                self.audio.speak(f"The text says: {text_content}")
                
            elif any(phrase in command for trigger_list in self.basic_triggers.values()
                     for phrase in trigger_list):
                self.handle_basic_command(command)
                
            return False
                
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            return False

    def handle_basic_command(self, command):
        try:
            if any(trigger in command for trigger in self.basic_triggers["time"]):
                response = self.basic_features.get_time()
                self.audio.speak(response)
            elif any(trigger in command for trigger in self.basic_triggers["weather"]):
                response = self.basic_features.get_weather()
                self.audio.speak(response)
            elif any(trigger in command for trigger in self.basic_triggers["location"]):
                response, _ = self.basic_features.get_location()
                self.audio.speak(response)
        except Exception as e:
            logger.error(f"Error in handle_basic_command: {e}")
            self.audio.speak("Sorry, I couldn't process that command")

    def main_loop(self):
        last_time = time.time()
        try:
            while True:
                current_time = time.time()
                if current_time - last_time < 1.0 / self.fps_limit:
                    continue
                last_time = current_time

                ret, frame = self.camera.read()
                if not ret:
                    continue

                # Process frame with complete vision pipeline
                frame, detections, results = self.process_frame(frame)

                # Display mode and FPS
                mode_text = f"Mode: {self.current_mode if self.current_mode else 'None'}"
                cv2.putText(frame, mode_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if results['gestures']:
                    cv2.putText(frame, f"Gestures: {', '.join(results['gestures'])}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Spex Vision", frame)

                # Handle audio commands
                command = self.audio.listen()
                if command:
                    should_stop = self.handle_command(command, frame, detections, results)
                    if should_stop:
                        break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            logger.error(f"Error in main_loop: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    assistant = SpexAssistant()
    assistant.start()