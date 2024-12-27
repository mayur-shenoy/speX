import speech_recognition as sr
import pyttsx3

class AudioModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000

    def listen(self):
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Command: {command}")
                return command
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            return ""

    def speak(self, text):
        print(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()