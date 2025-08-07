"""
Google Cloud Platform Weather Agent using Gemini and function calling.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from models_gcp import TripContext, WeatherAnalysis


# Mock weather functions (replace with real API calls)
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
    
    async def analyze_weather(self, context: TripContext) -> WeatherAnalysis:
        """Analyze weather for the given trip context."""
        current_date = datetime.now().date()
        trip_start = datetime.strptime(context.query.start_date, "%Y-%m-%d").date()
        days_until_trip = (trip_start - current_date).days
        
        prompt = f"""
You are a weather analyst helping travelers prepare for their trip.
Current date: {current_date}
Trip destination: {context.query.location}
Trip dates: {context.query.start_date} to {context.query.end_date}
Days until trip: {days_until_trip}

Please analyze the weather for this trip:

1. If the trip is within 7 days, use get_forecast to get detailed forecast data
2. If the trip is more than 7 days away, use get_current_weather for general conditions
3. Provide temperature range, precipitation chance, clothing recommendations, and any warnings
4. Return your analysis in a structured format with:
   - summary: Brief weather overview
   - temperature_range: [min_temp, max_temp] in Celsius
   - precipitation_chance: Percentage (0-100)
   - recommended_clothing: List of clothing items
   - weather_warnings: List of warnings (if any)

Please use the weather tools to get actual data before providing your analysis.
"""

        try:
            # Start conversation with the model
            chat = self.model.start_chat()
            response = chat.send_message(prompt)
            
            # Handle function calls
            while response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                if hasattr(part, 'function_call') and part.function_call:
                    # Execute the function call
                    function_result = self._execute_function_call(part.function_call)
                    
                    # Send the function result back to the model
                    response = chat.send_message(
                        f"Function {part.function_call.name} result: {function_result}"
                    )
                else:
                    # Extract structured response
                    response_text = part.text
                    return self._parse_weather_response(response_text, context)
            
            # Fallback if no proper response
            return get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)
            
        except Exception as e:
            print(f"Weather analysis error: {e}")
            return get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)
    
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


def create_weather_agent_gcp() -> WeatherAgentGCP:
    """Create a GCP-powered weather agent."""
    return WeatherAgentGCP()
