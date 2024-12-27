import os
import requests
from datetime import datetime
import pytz
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

load_dotenv()

class BasicFeaturesModule:
    def __init__(self):
        self.weather_api_key = os.getenv('WEATHER_API_KEY', '27e1b00e469a4fdd424a70f09f3dfca8')
        self.geolocator = Nominatim(user_agent="spex_assistant")
        self.location_cache = None
        self.weather_cache = None
        self.weather_cache_time = None
        self.cache_duration = 300  # 5 minutes cache

    def get_time(self):
        current_time = datetime.now().strftime("%I:%M %p")
        current_date = datetime.now().strftime("%A, %B %d")
        return f"It is {current_time} on {current_date}"

    def get_location(self, force_refresh=False):
        if not force_refresh and self.location_cache:
            return self.location_cache

        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                city = data.get('city', 'Unknown')
                region = data.get('region', '')
                country = data.get('country_name', '')
                lat = data.get('latitude')
                lon = data.get('longitude')
                
                location_info = f"You are in {city}, {region}, {country}"
                self.location_cache = (location_info, (lat, lon))
                return self.location_cache
            
        except Exception as e:
            print(f"Location error: {e}")
        
        return "I couldn't determine the location", None

    def get_weather(self, force_refresh=False):
        current_time = datetime.now().timestamp()
        
        # Return cached weather if valid
        if not force_refresh and self.weather_cache and self.weather_cache_time:
            if current_time - self.weather_cache_time < self.cache_duration:
                return self.weather_cache

        try:
            # Get location first
            _, coords = self.get_location()
            if not coords:
                return "I couldn't get the weather information without location data"

            lat, lon = coords
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                temp = round(data['main']['temp'])
                feels_like = round(data['main']['feels_like'])
                condition = data['weather'][0]['description']
                humidity = data['main']['humidity']
                wind_speed = data['wind']['speed']

                weather_info = (
                    f"The temperature is {temp} degrees Celsius, "
                    f"feels like {feels_like} degrees. "
                    f"The conditions are {condition}, with "
                    f"{humidity}% humidity and wind speed of {wind_speed} meters per second"
                )

                self.weather_cache = weather_info
                self.weather_cache_time = current_time
                return weather_info

        except Exception as e:
            print(f"Weather error: {e}")
        
        return "I'm sorry, I couldn't get the weather information"

    def get_simple_time(self):
        """Returns just the time without the date"""
        return datetime.now().strftime("%I:%M %p")

    def get_date(self):
        """Returns just the date"""
        return datetime.now().strftime("%A, %B %d, %Y")

    def get_short_weather(self):
        """Returns a shorter weather update"""
        try:
            _, coords = self.get_location()
            if coords:
                lat, lon = coords
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    temp = round(data['main']['temp'])
                    condition = data['weather'][0]['description']
                    return f"It's {temp} degrees with {condition}"
        except Exception:
            pass
        return "Weather information unavailable"


# New NavigationModule class for navigation functionality
class NavigationModule:
    def __init__(self):
        self.focal_length = 1000  # Will be calibrated
        self.known_width = 0.6  # Average human width in meters

    def estimate_distance(self, bbox_width, frame_width):
        # Distance = (Known width × Focal length) / Pixel width
        distance = (self.known_width * self.focal_length) / (bbox_width * frame_width)
        return round(distance, 2)

    def get_direction(self, center_x, frame_width):
        frame_center = frame_width / 2
        threshold = frame_width * 0.1
        
        if abs(center_x - frame_center) < threshold:
            return "straight ahead"
        elif center_x < frame_center:
            return "to your left"
        else:
            return "to your right"

    def describe_position(self, detections, frame_width):
        for detection in detections:
            if 'bbox' in detection:
                x1, y1, x2, y2 = detection['bbox']
                width = x2 - x1
                center_x = x1 + (width / 2)
                
                distance = self.estimate_distance(width/frame_width, frame_width)
                direction = self.get_direction(center_x, frame_width)
                
                return f"{detection['name']} is {direction}, approximately {distance} meters away"
        return "No objects detected for navigation"
