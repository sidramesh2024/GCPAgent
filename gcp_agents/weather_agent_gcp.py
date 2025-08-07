"""
Google Cloud Platform Weather Agent using Gemini and function calling.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from models_gcp import TripContext, WeatherAnalysis


# Enhanced location-based mock weather (no API calls needed)


def _get_location_based_mock_weather(location: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Generate realistic mock weather based on location."""
    location_lower = location.lower()
    
    # Location-specific weather patterns
    if any(city in location_lower for city in ["toronto", "canada", "montreal", "vancouver"]):
        # Canada - cooler, more rain
        temp_range = (8, 22)
        precipitation = 40
        conditions = ["partly cloudy", "light rain", "overcast"]
    elif any(city in location_lower for city in ["san francisco", "california", "los angeles", "seattle"]):
        # West Coast US - mild, less rain in summer
        temp_range = (15, 25)
        precipitation = 20
        conditions = ["sunny", "partly cloudy", "clear sky"]
    elif any(city in location_lower for city in ["new york", "boston", "chicago", "philadelphia"]):
        # East Coast US - variable
        temp_range = (12, 28)
        precipitation = 35
        conditions = ["partly cloudy", "sunny", "light rain"]
    elif any(city in location_lower for city in ["london", "paris", "berlin", "amsterdam"]):
        # Europe - variable, often cloudy
        temp_range = (10, 23)
        precipitation = 45
        conditions = ["overcast", "light rain", "partly cloudy"]
    elif any(city in location_lower for city in ["tokyo", "seoul", "beijing", "shanghai"]):
        # East Asia - variable
        temp_range = (18, 30)
        precipitation = 50
        conditions = ["humid", "partly cloudy", "moderate rain"]
    elif any(city in location_lower for city in ["sydney", "melbourne", "brisbane"]):
        # Australia - depends on season
        temp_range = (16, 26)
        precipitation = 25
        conditions = ["sunny", "partly cloudy", "clear sky"]
    else:
        # Default temperate climate
        temp_range = (15, 25)
        precipitation = 30
        conditions = ["partly cloudy", "sunny", "light rain"]
    
    # Generate forecast for date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    forecast = []
    
    current_date = start
    day_count = 0
    while current_date <= end:
        # Add small daily variation (±3°C for realism)
        temp_variation = (day_count % 5) - 2  # Variation between -2 and +2
        temp_min = temp_range[0] + temp_variation
        temp_max = temp_range[1] + temp_variation
        
        # Add small precipitation variation (±10%)
        precip_variation = (day_count % 7) - 3  # Variation between -3 and +3
        precip = precipitation + precip_variation
        
        condition = conditions[day_count % len(conditions)]
        
        forecast.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "temperature_min": max(temp_min, -10),  # Reasonable minimum
            "temperature_max": min(temp_max, 45),   # Reasonable maximum  
            "precipitation_chance": max(0, min(100, precip)),
            "condition": condition
        })
        current_date += timedelta(days=1)
        day_count += 1
    
    return {
        "location": location,
        "forecast": forecast
    }


# Mock weather functions (kept for fallback)
def get_current_weather(location: str) -> Dict[str, Any]:
    """Mock function to get current weather for a location."""
    return {
        "location": location,
        "temperature": 22.5,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": 10,
        "precipitation": 0
    }


def get_forecast(location: str, days: int = 7) -> Dict[str, Any]:
    """Mock function to get weather forecast for a location."""
    return {
        "location": location,
        "forecast": [
            {
                "date": "2024-01-15",
                "temperature_min": 18,
                "temperature_max": 25,
                "condition": "sunny",
                "precipitation_chance": 10
            },
            {
                "date": "2024-01-16",
                "temperature_min": 20,
                "temperature_max": 27,
                "condition": "partly cloudy",
                "precipitation_chance": 30
            }
        ]
    }


def get_weather_mock(location: str, start_date: str, end_date: str) -> WeatherAnalysis:
    """Mock weather analysis for fallback scenarios."""
    return WeatherAnalysis(
        summary=f"Weather forecast for {location} from {start_date} to {end_date}: Expected to have mild, pleasant conditions with comfortable temperatures throughout your visit. Perfect weather for outdoor activities and sightseeing. (Note: This is demo data - add your Google AI API key for real weather forecasts)",
        temperature_range=[18.0, 27.0],
        precipitation_chance=20.0,
        recommended_clothing=[
            "Comfortable walking shoes",
            "Layered clothing for temperature changes", 
            "Light jacket or sweater for evenings",
            "Sun protection (hat, sunglasses)",
            "Comfortable day pack"
        ],
        weather_warnings=None
    )


class WeatherAgentGCP:
    """Google Cloud Platform Weather Agent using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the weather agent with Gemini model."""
        self.model_name = model_name
        
        # Define weather tools for function calling
        self.weather_tools = [
            Tool(function_declarations=[
                FunctionDeclaration(
                    name="get_current_weather",
                    description="Get current weather conditions for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for"
                            }
                        },
                        "required": ["location"]
                    }
                ),
                FunctionDeclaration(
                    name="get_forecast",
                    description="Get weather forecast for a location (up to 7 days)",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get forecast for"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days to forecast (max 7)"
                            }
                        },
                        "required": ["location"]
                    }
                )
            ])
        ]
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=self.weather_tools
        )
    
    def _execute_function_call(self, function_call) -> Dict[str, Any]:
        """Execute a function call and return the result."""
        function_name = function_call.name
        function_args = {key: value for key, value in function_call.args.items()}
        
        if function_name == "get_current_weather":
            return get_current_weather(**function_args)
        elif function_name == "get_forecast":
            # Set default for days if not provided
            if "days" not in function_args:
                function_args["days"] = 7
            return get_forecast(**function_args)
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def analyze_weather(self, context: TripContext, use_real_weather: bool = False) -> WeatherAnalysis:
        """Analyze weather for the given trip context."""
        current_date = datetime.now().date()
        trip_start = datetime.strptime(context.query.start_date, "%Y-%m-%d").date()
        days_until_trip = (trip_start - current_date).days
        
        # Note: Real weather is handled by MCP server in manager_gcp.py
        # This method only handles enhanced location-based mock weather
        print(f"📍 Using enhanced location-based mock weather for {context.query.location}...")
            
        # Use enhanced location-based mock weather
        weather_data = _get_location_based_mock_weather(
            context.query.location, 
            context.query.start_date, 
            context.query.end_date
        )
        return self._parse_real_weather_data(weather_data, context)
    
    def _parse_weather_response(self, response_text: str, context: TripContext) -> WeatherAnalysis:
        """Parse the model's response into a WeatherAnalysis object."""
        try:
            # Clean up the response text
            clean_response = response_text.strip()
            
            # Create a user-friendly weather summary
            if clean_response and len(clean_response) > 50:
                # Extract useful information and create a clean summary
                summary_parts = []
                
                # Basic forecast intro
                summary_parts.append(f"Weather forecast for your trip to {context.query.location}:")
                
                # Try to extract key weather info from AI response
                lines = clean_response.split('\n')
                weather_info = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('*') and not line.startswith('#'):
                        # Skip function call results and metadata
                        if 'function' not in line.lower() and 'result:' not in line.lower():
                            weather_info.append(line)
                
                # Take the most relevant weather information
                if weather_info:
                    # Join the first few meaningful lines
                    relevant_info = ' '.join(weather_info[:3])
                    summary_parts.append(relevant_info[:300])
                else:
                    summary_parts.append("Pleasant conditions expected during your visit.")
                
                summary = ' '.join(summary_parts)
            else:
                summary = f"Weather forecast for {context.query.location} from {context.query.start_date} to {context.query.end_date}: Pleasant conditions expected during your visit."
            
            # Extract temperature info from response
            temp_range = [18.0, 26.0]  # Default
            if 'temperature' in clean_response.lower():
                # Try to extract temperature numbers
                import re
                temp_matches = re.findall(r'(\d+).*?(?:°|degree)', clean_response)
                if len(temp_matches) >= 2:
                    try:
                        temps = [float(t) for t in temp_matches[:2]]
                        temp_range = [min(temps), max(temps)]
                    except:
                        pass
            
            # Extract precipitation info
            precip_chance = 25.0  # Default
            if any(word in clean_response.lower() for word in ['rain', 'shower', 'precipitation']):
                precip_chance = 60.0
                if 'low' in clean_response.lower() or 'little' in clean_response.lower():
                    precip_chance = 20.0
                elif 'high' in clean_response.lower() or 'likely' in clean_response.lower():
                    precip_chance = 80.0
            
            # Smart clothing recommendations
            clothing = [
                "Comfortable walking shoes",
                "Layered clothing for temperature changes"
            ]
            
            # Add temperature-based clothing
            avg_temp = sum(temp_range) / 2
            if avg_temp < 15:
                clothing.extend(["Warm jacket", "Long pants", "Warm layers"])
            elif avg_temp < 25:
                clothing.extend(["Light jacket or sweater", "Long pants or jeans"])
            else:
                clothing.extend(["Light clothing", "Shorts or light pants", "Sun hat"])
            
            # Add weather-specific items
            if precip_chance > 40:
                clothing.extend(["Rain jacket or umbrella", "Waterproof shoes"])
            
            clothing.extend(["Sun protection (sunglasses, sunscreen)", "Comfortable day pack"])
            
            # Weather warnings
            warnings = None
            if precip_chance > 70:
                warnings = ["High chance of rain - pack rain protection"]
            elif any(word in clean_response.lower() for word in ['storm', 'severe', 'extreme']):
                warnings = ["Check weather conditions before outdoor activities"]
            
            return WeatherAnalysis(
                summary=summary,
                temperature_range=temp_range,
                precipitation_chance=precip_chance,
                recommended_clothing=clothing,
                weather_warnings=warnings
            )
            
        except Exception as e:
            print(f"Error parsing weather response: {e}")
            return get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)
    
    def _parse_real_weather_data(self, weather_data: Dict[str, Any], context: TripContext) -> WeatherAnalysis:
        """Parse weather data from API or enhanced mock into WeatherAnalysis."""
        forecast = weather_data.get("forecast", [])
        if not forecast:
            # Fallback to basic mock
            return get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)
        
        # Calculate temperature range
        temps_min = [day["temperature_min"] for day in forecast]
        temps_max = [day["temperature_max"] for day in forecast]
        temp_range = [min(temps_min), max(temps_max)]
        
        # Calculate average precipitation chance
        precip_chances = [day["precipitation_chance"] for day in forecast]
        avg_precipitation = sum(precip_chances) / len(precip_chances)
        
        # Generate clothing recommendations based on temperature and weather
        clothing = self._generate_clothing_recommendations(temp_range, avg_precipitation, forecast)
        
        # Generate weather summary
        location_name = weather_data.get("location", context.query.location)
        conditions = [day["condition"] for day in forecast]
        most_common_condition = max(set(conditions), key=conditions.count)
        
        summary = f"Weather forecast for {location_name} from {context.query.start_date} to {context.query.end_date}: "
        if avg_precipitation < 20:
            summary += f"Expect mostly {most_common_condition} conditions with temperatures ranging from {temp_range[0]:.0f}°C to {temp_range[1]:.0f}°C. Great weather for outdoor activities!"
        elif avg_precipitation < 50:
            summary += f"Variable conditions with {most_common_condition} weather and temperatures from {temp_range[0]:.0f}°C to {temp_range[1]:.0f}°C. Pack for mixed weather."
        else:
            summary += f"Expect frequent {most_common_condition} with temperatures from {temp_range[0]:.0f}°C to {temp_range[1]:.0f}°C. Indoor activities recommended."
        
        return WeatherAnalysis(
            summary=summary,
            temperature_range=temp_range,
            precipitation_chance=avg_precipitation,
            recommended_clothing=clothing
        )
    
    def _generate_clothing_recommendations(self, temp_range: List[float], precipitation: float, forecast: List[Dict]) -> List[str]:
        """Generate clothing recommendations based on weather data."""
        clothing = []
        min_temp, max_temp = temp_range
        
        # Temperature-based recommendations
        if max_temp >= 25:
            clothing.extend(["light t-shirts", "shorts", "sandals", "sun hat"])
        elif max_temp >= 20:
            clothing.extend(["t-shirts", "light pants", "comfortable shoes"])
        elif max_temp >= 15:
            clothing.extend(["long sleeves", "pants", "light jacket"])
        elif max_temp >= 10:
            clothing.extend(["sweater", "pants", "jacket", "closed shoes"])
        else:
            clothing.extend(["warm coat", "sweater", "long pants", "boots"])
        
        if min_temp < 10:
            clothing.append("warm layers")
        
        # Precipitation-based recommendations
        if precipitation > 30:
            clothing.extend(["rain jacket", "umbrella", "waterproof shoes"])
        elif precipitation > 15:
            clothing.append("light rain jacket")
        
        # Seasonal recommendations
        has_sun = any("clear" in day.get("condition", "").lower() or "sunny" in day.get("condition", "").lower() for day in forecast)
        if has_sun:
            clothing.extend(["sunglasses", "sunscreen"])
        
        return clothing[:8]  # Limit to 8 items


def create_weather_agent_gcp() -> WeatherAgentGCP:
    """Create a GCP-powered weather agent."""
    return WeatherAgentGCP()
