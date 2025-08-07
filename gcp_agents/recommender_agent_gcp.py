"""
Google Cloud Platform Recommendation Agent using Gemini for trip planning.
"""

from typing import List
import google.generativeai as genai
from models_gcp import TripContext, SearchResult, TripPlan, ActivityRecommendation, WeatherAnalysis


class RecommendationAgentGCP:
    """Google Cloud Platform Recommendation Agent using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the recommendation agent with Gemini model."""
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name=model_name)
    
    async def create_trip_plan(
        self,
        context: TripContext,
        search_results: SearchResult,
        weather_info: WeatherAnalysis
    ) -> TripPlan:
        """Create a comprehensive trip plan based on search results and weather."""
        
        prompt = f"""
You are an expert travel planner creating a comprehensive trip plan.

Trip Details:
- Destination: {context.query.location}
- Dates: {context.query.start_date} to {context.query.end_date}
- Participants: {context.query.participant_number} people, ages {context.query.participant_ages}

Weather Information:
- Summary: {weather_info.summary}
- Temperature Range: {weather_info.temperature_range[0]}°C to {weather_info.temperature_range[1]}°C
- Precipitation Chance: {weather_info.precipitation_chance}%
- Recommended Clothing: {', '.join(weather_info.recommended_clothing)}
{f"- Weather Warnings: {', '.join(weather_info.weather_warnings)}" if weather_info.weather_warnings else ""}

Available Activities:
{search_results.search_summary}

Activities Found:
{chr(10).join([f"- {activity.name}: {activity.description} (Duration: {activity.duration}, Weather dependent: {activity.weather_dependent})" for activity in search_results.activities])}

Create exactly 5 specific activity recommendations for this trip. 

For EACH activity, provide the following in this exact format:

ACTIVITY 1: [Activity Name]
DESCRIPTION: [2-3 sentence description]
REASONING: [Why this fits the trip and group]
BEST_TIME: [Morning/Afternoon/Evening]
WEATHER: [Weather considerations]
TIPS: [Preparation tips]

ACTIVITY 2: [Activity Name]
DESCRIPTION: [2-3 sentence description]
REASONING: [Why this fits the trip and group]
BEST_TIME: [Morning/Afternoon/Evening]
WEATHER: [Weather considerations]
TIPS: [Preparation tips]

ACTIVITY 3: [Activity Name]
DESCRIPTION: [2-3 sentence description]
REASONING: [Why this fits the trip and group]
BEST_TIME: [Morning/Afternoon/Evening]
WEATHER: [Weather considerations]
TIPS: [Preparation tips]

ACTIVITY 4: [Activity Name]
DESCRIPTION: [2-3 sentence description]
REASONING: [Why this fits the trip and group]
BEST_TIME: [Morning/Afternoon/Evening]
WEATHER: [Weather considerations]
TIPS: [Preparation tips]

ACTIVITY 5: [Activity Name]
DESCRIPTION: [2-3 sentence description]
REASONING: [Why this fits the trip and group]
BEST_TIME: [Morning/Afternoon/Evening]
WEATHER: [Weather considerations]
TIPS: [Preparation tips]

Requirements:
- Mix of indoor/outdoor activities
- Variety in experience types (cultural, recreational, dining, sightseeing)
- Age-appropriate for adults in their 30s
- Consider the weather summary provided
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            return self._parse_trip_plan_response(response_text, context, search_results, weather_info)
            
        except Exception as e:
            print(f"Trip planning error: {e}")
            return self._create_fallback_trip_plan(context, search_results, weather_info)
    
    def _parse_trip_plan_response(
        self,
        response_text: str,
        context: TripContext,
        search_results: SearchResult,
        weather_info: WeatherAnalysis
    ) -> TripPlan:
        """Parse the model's response into a structured TripPlan."""
        try:
            # Try to parse AI-generated activities from the response
            recommended_activities = []
            
            # Parse the structured AI response
            if response_text and len(response_text) > 100:
                import re
                
                # Extract activities using the structured format
                activity_pattern = r'ACTIVITY \d+: (.+?)(?=ACTIVITY \d+:|$)'
                activity_matches = re.findall(activity_pattern, response_text, re.DOTALL | re.IGNORECASE)
                
                for match in activity_matches[:5]:  # Ensure max 5 activities
                    activity_text = match.strip()
                    lines = activity_text.split('\n')
                    
                    activity_data = {
                        'name': '',
                        'description': '',
                        'reasoning': '',
                        'best_time': 'Flexible',
                        'weather_considerations': [],
                        'preparation_tips': []
                    }
                    
                    # Parse the first line as the activity name
                    if lines:
                        activity_data['name'] = lines[0].strip()
                    
                    # Parse structured fields
                    for line in lines[1:]:
                        line = line.strip()
                        if line.upper().startswith('DESCRIPTION:'):
                            activity_data['description'] = line[12:].strip()
                        elif line.upper().startswith('REASONING:'):
                            activity_data['reasoning'] = line[10:].strip()
                        elif line.upper().startswith('BEST_TIME:'):
                            activity_data['best_time'] = line[10:].strip()
                        elif line.upper().startswith('WEATHER:'):
                            activity_data['weather_considerations'] = [line[8:].strip()]
                        elif line.upper().startswith('TIPS:'):
                            activity_data['preparation_tips'] = [line[5:].strip()]
                    
                    # Create activity recommendation
                    if activity_data['name']:
                        recommended_activities.append(self._create_activity_from_ai(activity_data, weather_info))
            
            # If we didn't get 5 activities from AI parsing, supplement with smart recommendations
            while len(recommended_activities) < 5 and len(search_results.activities) > len(recommended_activities):
                i = len(recommended_activities)
                activity = search_results.activities[i]
                times = ["Morning", "Late Morning", "Afternoon", "Late Afternoon", "Evening"]
                
                recommendation = ActivityRecommendation(
                    name=activity.name,
                    description=activity.description,
                    reasoning=f"Recommended as activity #{i+1} - complements your itinerary with {'indoor' if not activity.weather_dependent else 'outdoor'} options.",
                    best_time=times[i % len(times)],
                    weather_considerations=[
                        f"Temperature: {weather_info.temperature_range[0]}-{weather_info.temperature_range[1]}°C",
                        f"Rain chance: {weather_info.precipitation_chance}%"
                    ] if activity.weather_dependent else ["Indoor activity - weather independent"],
                    preparation_tips=[
                        "Check opening hours in advance",
                        "Bring essentials and comfortable shoes",
                        "Consider booking ahead if popular"
                    ],
                    source_url=activity.source_url
                )
                recommended_activities.append(recommendation)
            
            # Generate packing list based on weather and activities
            packing_list = weather_info.recommended_clothing.copy()
            packing_list.extend([
                "Comfortable walking shoes",
                "Water bottle",
                "Snacks",
                "Phone charger",
                "Camera",
                "Cash/cards",
                "Sunscreen",
                "First aid kit"
            ])
            
            # Add weather-specific items
            if weather_info.precipitation_chance > 40:
                packing_list.extend(["Umbrella", "Rain jacket"])
            
            # Add items for children if present
            if any(age < 12 for age in context.query.participant_ages):
                packing_list.extend([
                    "Child-friendly snacks",
                    "Entertainment for kids",
                    "Wet wipes",
                    "Extra clothes for children"
                ])
            
            # Generate general tips
            general_tips = [
                f"Check weather forecast before heading out - {weather_info.precipitation_chance}% chance of rain",
                "Book activities in advance during peak season",
                "Keep emergency contacts handy",
                "Stay hydrated and take breaks as needed"
            ]
            
            # Add age-specific tips
            if any(age < 12 for age in context.query.participant_ages):
                general_tips.append("Plan for shorter activity durations with children")
                general_tips.append("Locate nearby restrooms and family facilities")
            
            if any(age > 65 for age in context.query.participant_ages):
                general_tips.append("Consider accessibility of venues")
                general_tips.append("Plan rest periods between activities")
            
            # Add weather warnings as tips
            if weather_info.weather_warnings:
                general_tips.extend(weather_info.weather_warnings)
            
            return TripPlan(
                location=context.query.location,
                dates=f"{context.query.start_date} to {context.query.end_date}",
                participants_summary=f"{context.query.participant_number} participants (ages: {', '.join(map(str, context.query.participant_ages))})",
                weather_summary=weather_info.summary,
                recommended_activities=recommended_activities,
                packing_list=list(set(packing_list)),  # Remove duplicates
                general_tips=general_tips
            )
            
        except Exception as e:
            print(f"Error parsing trip plan response: {e}")
            return self._create_fallback_trip_plan(context, search_results, weather_info)
    
    def _create_activity_from_ai(self, activity_data: dict, weather_info: WeatherAnalysis) -> ActivityRecommendation:
        """Create an ActivityRecommendation from parsed AI response data."""
        return ActivityRecommendation(
            name=activity_data.get('name', 'Activity').strip(),
            description=activity_data.get('description', 'AI-recommended activity').strip(),
            reasoning=activity_data.get('reasoning', 'Recommended by AI for your trip').strip(),
            best_time=activity_data.get('best_time', 'Flexible').strip(),
            weather_considerations=activity_data.get('weather_considerations') or [
                f"Temperature: {weather_info.temperature_range[0]}-{weather_info.temperature_range[1]}°C"
            ],
            preparation_tips=activity_data.get('preparation_tips') or [
                "Check opening hours",
                "Bring essentials"
            ],
            source_url=None
        )
    
    def _create_fallback_trip_plan(
        self,
        context: TripContext,
        search_results: SearchResult,
        weather_info: WeatherAnalysis
    ) -> TripPlan:
        """Create a basic fallback trip plan."""
        
        # Basic recommended activities - select up to 5
        recommended_activities = []
        activities_to_recommend = min(5, len(search_results.activities))
        
        for i in range(activities_to_recommend):
            activity = search_results.activities[i]
            times = ["Morning", "Afternoon", "Evening", "Anytime", "Mid-day"]
            recommendation = ActivityRecommendation(
                name=activity.name,
                description=activity.description,
                reasoning=f"Selected as activity #{i+1} for the trip - suitable for your group",
                best_time=times[i % len(times)],
                weather_considerations=["Check weather before departure"] if activity.weather_dependent else ["Indoor activity - weather independent"],
                preparation_tips=["Bring essentials", "Arrive early", "Check operating hours"],
                source_url=activity.source_url
            )
            recommended_activities.append(recommendation)
        
        return TripPlan(
            location=context.query.location,
            dates=f"{context.query.start_date} to {context.query.end_date}",
            participants_summary=f"{context.query.participant_number} participants",
            weather_summary=weather_info.summary,
            recommended_activities=recommended_activities,
            packing_list=["Comfortable clothing", "Walking shoes", "Water", "Snacks"],
            general_tips=["Check weather forecast", "Bring essentials", "Have a great trip!"]
        )


def create_recommendation_agent_gcp() -> RecommendationAgentGCP:
    """Create a GCP-powered recommendation agent."""
    return RecommendationAgentGCP()
