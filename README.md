
# **SpeX** - AI-Powered Smart Glasses for Visually Impaired  

**SpeX** is an AI-powered smart glasses solution designed to help visually impaired individuals navigate the world independently. By embedding advanced AI technologies, SpeX acts as the "eyes" for its users, providing real-time visual assistance, auditory feedback, geographical tracking, distance measurement, and real-time alerts for enhanced mobility, safety, and social interaction.  

---  

## **Problem Statement**  

Visually impaired individuals face significant challenges in navigating their environment, interacting with others, and accessing textual information. Traditional tools lack real-time visual, geographical, and distance-based assistance, as well as proactive alerts, limiting independence. SpeX addresses these challenges by integrating AI-driven object detection, face recognition, text-to-speech conversion, real-time tracking, distance measurement, and alert systems within a wearable smart glasses framework.  

---  

## **Objective**  

SpeX aims to empower visually impaired users by providing a wearable solution that offers:  

- Real-time object detection  
- Face recognition for social interactions  
- Text recognition and reading  
- Real-time geographical tracking for location awareness  
- Distance measurement for obstacle detection and navigation  
- Real-time alerts for sudden changes or potential dangers in the environment  
- Voice-based interaction for seamless assistance  

This AI-driven system promotes independence, safety, and social integration.  

---  

## **Key Features**  

- **Object Detection (YOLO)**: Real-time identification and localization of objects in the environment, helping users navigate obstacles and identify points of interest.  
- **Face Recognition (OpenCV)**: Detects and recognizes faces for improved social interaction and safety.  
- **Text Recognition (Tesseract OCR)**: Converts printed text (from signs, labels, books, etc.) into spoken words.  
- **Real-Time Geographical Tracking**: Provides real-time location data and tracking using geolocation technologies for navigation and situational awareness.  
- **Distance Measurement**: Measures how far an object or obstacle is, providing real-time feedback for safe navigation.  
- **Real-Time Alerts**: Proactively warns users of sudden environmental changes, obstacles, or potential hazards to ensure safety.  
- **Voice Interaction**: Seamlessly integrates voice commands to allow users to interact with the assistant hands-free, enabling a natural experience.  

---  

## **Technologies Used**  

### **Hardware**  
- **Webcam**: Captures real-time visual input.  
- **Microphone**: Captures voice commands for interaction.  
- **Speaker/Headphones**: Outputs auditory feedback and alerts.  
- **Raspberry Pi 0**: Powers the system with cost-effective hardware.  
- **Bluetooth Earphones**: For portable, hands-free audio.  
- **Ultrasonic Sensors**: Measures the distance to obstacles and provides real-time feedback and alerts.  

### **Software**  
- **Numpy**: Numerical computations and data manipulation.  
- **OpenCV**: Real-time image and video processing for face and object recognition.  
- **Mediapipe**: Provides real-time ML-based detection and tracking of gestures and faces.  
- **Pytesseract**: Extracts text from images for real-time reading.  
- **Transformers**: Pre-trained models for NLP tasks such as text summarization.  
- **Torch**: Deep learning framework used for training and implementing AI models.  
- **Ultralytics**: YOLO-based object detection to help identify objects in real-time.  
- **Face_recognition**: Detects and recognizes faces using deep learning models.  
- **Speechrecognition**: Converts speech to text, enabling voice command processing.  
- **Pytz**: Handles time zone operations for real-time assistance.  
- **Geopy**: Provides geolocation and mapping functionalities for real-time tracking.  
- **Pyttsx3**: Converts text to speech for delivering auditory feedback and alerts.  

---  

## **Installation & Setup**  

1. Ensure you have **Python 3.11** installed for optimal stability and performance.  

2. Clone the repository:  
   ```bash  
   git clone https://github.com/harshendram/speXweb.git  
   ```  

3. Navigate to the project directory:  
   ```bash  
   cd speXweb  
   ```  

4. Install required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

5. Run the application:  
   ```bash  
   python speX.py  
   ```  

---  

## **Usage**  

SpeX provides real-time assistance through voice commands and proactive alerts. Here's how you can interact with the system:  

- **"What is the distance to [object]?"** – Measures the distance to the specified object or obstacle.  
- **"Identify objects around me."** – Detect objects in the user's vicinity.  
- **"Who is that?"** – Recognize and identify faces.  
- **"Read this text."** – Convert visible text into speech.  
- **"Where am I?"** – Retrieve real-time geographical location data.  
- **Proactive Alerts** – Notifies users of sudden changes or hazards (e.g., "Obstacle detected 2 meters ahead!").  

---  

## **Impact**  

SpeX greatly enhances the independence of visually impaired individuals by providing:  

- **Autonomous Navigation**: Users can avoid obstacles, measure distances, and identify objects in real-time.  
- **Social Interaction**: With face recognition, SpeX helps users recognize friends, family, or others.  
- **Information Access**: Text recognition empowers users to read printed materials, signs, and more.  
- **Geographical Awareness**: Real-time tracking ensures users are aware of their location and can navigate safely.  
- **Safety Alerts**: Proactive alerts enhance situational awareness, preventing potential accidents.  

---  

## **Scalability**  

SpeX is built to scale and evolve:  

- **Language Expansion**: The AI assistant can be extended to support more languages.  
- **Advanced Recognition**: Future improvements in AI models will lead to better accuracy in object and face detection.  
- **Assistive Technology Integration**: SpeX can integrate with other assistive technologies to broaden its capabilities.  

---  

## **Links**  

- [GitHub Repository]()  

---  
