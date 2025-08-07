"""
Google Cloud Platform Adventure Manager - orchestrates the multi-agent workflow using Gemini.
"""

import os
import asyncio
from typing import Optional
import google.generativeai as genai
from google.auth.exceptions import DefaultCredentialsError

from models_gcp import TripQuery, TripContext, CHILD_AGE_THRESHOLD
from gcp_agents import (
    create_weather_agent_gcp,
    create_activity_search_agent_gcp,
    create_recommendation_agent_gcp,
    create_kid_friendly_activity_agent_gcp,
    get_weather_mock,
    WeatherAnalysis,
    SearchResult,
    TripPlan
)
from tracing_adk import get_enhanced_tracer, setup_adk_tracing_environment

# MCP imports for weather server
try:
    import subprocess
    import json
    from datetime import datetime, timedelta
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class AdventureManagerGCP:
    """Manages the adventure planning workflow using Google Cloud Platform agents."""

    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None):
        """Initialize the GCP Adventure Manager.
        
        Args:
            api_key: Google AI API key for Gemini (optional if using environment variable)
            project_id: GCP project ID (optional, needed for some advanced features)
        """
        self.project_id = project_id
        
        # Configure Google AI
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get from environment
            api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
            else:
                print("Warning: No Google AI API key found. Using mock responses.")
                print("Set GOOGLE_AI_API_KEY or GEMINI_API_KEY environment variable.")
        
        # Set up enhanced ADK tracing environment
        setup_adk_tracing_environment()
        
        # Initialize agents
        try:
            self.weather_agent = create_weather_agent_gcp()
            self.activity_search_agent = create_activity_search_agent_gcp()
            self.recommendation_agent = create_recommendation_agent_gcp()
            self.kid_friendly_agent = create_kid_friendly_activity_agent_gcp()
            self.agents_initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize GCP agents: {e}")
            self.agents_initialized = False

    async def run(self, query: TripQuery, use_real_weather: bool = False) -> TripPlan:
        """Run the complete adventure planning workflow.
        
        Args:
            query: Trip query with destination, dates, and participant info
            use_real_weather: Whether to attempt real weather data (not implemented in this version)
            
        Returns:
            Complete trip plan with recommendations
        """
        tracer = get_enhanced_tracer()
        
        # Start workflow tracing
        tracer.start_workflow("Adventure Planning", {
            "destination": query.location,
            "dates": f"{query.start_date} to {query.end_date}",
            "participants": query.participant_number,
            "ages": query.participant_ages,
            "use_real_weather": use_real_weather
        })
        
        print(f"🚀 Starting GCP-powered adventure planning for {query.location}...")
        print(f"📅 Dates: {query.start_date} to {query.end_date}")
        print(f"👥 Participants: {query.participant_number} people (ages: {query.participant_ages})")
        
        # Create the context object
        trip_context = TripContext(query=query)
        
        try:
            # 1. Get Weather Information
            print("\n🌤️  Getting weather information...")
            weather_info = await self._get_weather_info(trip_context, use_real_weather)
            print(f"Weather summary: {weather_info.summary}")
            
            # 2. Search for Activities (with child-friendly routing)
            print("\n🎯 Searching for activities...")
            search_results, agent_used = await self._search_for_activities(trip_context, weather_info)
            print(f"Found {len(search_results.activities)} activities using {agent_used}")
            
            # 3. Generate Trip Plan
            print("\n📋 Creating comprehensive trip plan...")
            trip_plan = await self._generate_trip_plan(search_results, weather_info, trip_context)
            
            print("\n✅ Adventure planning complete!")
            tracer.complete_workflow("Adventure Planning", success=True)
            return trip_plan
            
        except Exception as e:
            print(f"❌ Error during planning: {e}")
            tracer.log_error("Workflow Manager", str(e))
            tracer.complete_workflow("Adventure Planning", success=False)
            return self._create_fallback_plan(trip_context)

    async def _get_weather_info(self, context: TripContext, use_real_weather: bool = False) -> WeatherAnalysis:
        """Get weather information for the trip."""
        tracer = get_enhanced_tracer()
        agent_key = tracer.start_agent("Weather Agent", f"Analyze weather for {context.query.location}", {
            "location": context.query.location,
            "dates": f"{context.query.start_date} to {context.query.end_date}",
            "use_real_weather": use_real_weather
        })
        
        if not self.agents_initialized:
            print("Weather agent not initialized, using basic mock data...")
            weather_info = get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)
            tracer.complete_agent(agent_key, "Weather Agent", "Basic mock weather data generated", {
                "temperature_range": weather_info.temperature_range,
                "precipitation_chance": weather_info.precipitation_chance
            })
            return weather_info
        
        try:
            if use_real_weather:
                # Use MCP weather server for real data
                print("🌐 Connecting to MCP Weather Server for real weather data...")
                tracer.log_tool_call("Weather Agent", "mcp_weather_server", {"context": "Trip context", "use_real_weather": True})
                weather_info = await self._get_mcp_weather_info(context)
                tracer.complete_agent(agent_key, "Weather Agent", "MCP weather data retrieved", {
                    "temperature_range": weather_info.temperature_range,
                    "precipitation_chance": weather_info.precipitation_chance
                })
                return weather_info
            else:
                # Use enhanced location-based mock data
                print(f"📍 Using enhanced location-based weather for {context.query.location}...")
                tracer.log_tool_call("Weather Agent", "analyze_weather", {"context": "Trip context", "use_real_weather": False})
                weather_info = await self.weather_agent.analyze_weather(context, use_real_weather=False)
                tracer.complete_agent(agent_key, "Weather Agent", "Enhanced mock weather analysis completed", {
                    "temperature_range": weather_info.temperature_range,
                    "precipitation_chance": weather_info.precipitation_chance
                })
                return weather_info
        except Exception as e:
            print(f"Weather analysis failed: {e}. Using mock data.")
            tracer.log_error("Weather Agent", str(e))
            weather_info = get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)
            tracer.complete_agent(agent_key, "Weather Agent", "Fallback to mock data", {"error": str(e)})
            return weather_info

    async def _get_mcp_weather_info(self, context: TripContext) -> WeatherAnalysis:
        """Get weather information using the MCP weather server."""
        if not MCP_AVAILABLE:
            print("⚠️  MCP not available, using enhanced mock data...")
            return await self.weather_agent.analyze_weather(context, use_real_weather=False)
        
        try:
            # Use Docker to run the MCP weather server
            print(f"🐳 Starting MCP weather server for {context.query.location}...")
            
            # Try with sudo first, then without
            commands_to_try = [
                ["sudo", "docker", "run", "--rm", 
                 "-e", f"LOCATION={context.query.location}",
                 "-e", f"START_DATE={context.query.start_date}",
                 "-e", f"END_DATE={context.query.end_date}",
                 "mcp-weather", "get_forecast", context.query.location],
                ["docker", "run", "--rm",
                 "-e", f"LOCATION={context.query.location}",
                 "-e", f"START_DATE={context.query.start_date}",
                 "-e", f"END_DATE={context.query.end_date}",
                 "mcp-weather", "get_forecast", context.query.location]
            ]
            
            result = None
            for cmd in commands_to_try:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        break
                except subprocess.TimeoutExpired:
                    print("⚠️  MCP weather server timeout")
                    continue
                except FileNotFoundError:
                    if "sudo" in cmd:
                        continue  # Try without sudo
                    break
            
            if result and result.returncode == 0:
                # Parse the MCP response
                try:
                    weather_data = json.loads(result.stdout)
                    return self._parse_mcp_weather_response(weather_data, context)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Failed to parse MCP weather response: {e}")
            else:
                print(f"⚠️  MCP weather server failed: {result.stderr if result else 'Unknown error'}")
            
        except Exception as e:
            print(f"⚠️  MCP weather server error: {e}")
        
        # Fallback to enhanced mock data
        print("📍 Falling back to enhanced location-based weather...")
        return await self.weather_agent.analyze_weather(context, use_real_weather=False)
    
    def _parse_mcp_weather_response(self, weather_data: dict, context: TripContext) -> WeatherAnalysis:
        """Parse MCP weather server response into WeatherAnalysis."""
        try:
            # Extract forecast data
            daily_forecasts = weather_data.get("daily_forecasts", [])
            location_info = weather_data.get("location", {})
            
            if not daily_forecasts:
                raise ValueError("No forecast data in MCP response")
            
            # Calculate temperature range
            temps_max = [day.get("temperature_max", 20) for day in daily_forecasts]
            temps_min = [day.get("temperature_min", 15) for day in daily_forecasts]
            temp_range = [min(temps_min), max(temps_max)]
            
            # Calculate precipitation chance
            precip_chances = [day.get("precipitation_probability", 30) for day in daily_forecasts]
            avg_precipitation = sum(precip_chances) / len(precip_chances)
            
            # Generate weather summary
            location_name = location_info.get("name", context.query.location)
            summary = f"Real weather forecast for {location_name} from {context.query.start_date} to {context.query.end_date}: "
            
            if avg_precipitation < 20:
                summary += f"Clear conditions expected with temperatures from {temp_range[0]:.0f}°C to {temp_range[1]:.0f}°C. Perfect for outdoor activities!"
            elif avg_precipitation < 50:
                summary += f"Mixed weather conditions with temperatures from {temp_range[0]:.0f}°C to {temp_range[1]:.0f}°C. Pack for variable weather."
            else:
                summary += f"Rainy weather likely with temperatures from {temp_range[0]:.0f}°C to {temp_range[1]:.0f}°C. Indoor activities recommended."
            
            # Generate clothing recommendations
            clothing = []
            min_temp, max_temp = temp_range
            
            if max_temp >= 25:
                clothing.extend(["light t-shirts", "shorts", "sandals", "sun hat"])
            elif max_temp >= 20:
                clothing.extend(["t-shirts", "light pants", "comfortable shoes"])
            elif max_temp >= 15:
                clothing.extend(["long sleeves", "pants", "light jacket"])
            else:
                clothing.extend(["warm clothes", "jacket", "layers"])
            
            if avg_precipitation > 30:
                clothing.extend(["rain jacket", "umbrella"])
            
            return WeatherAnalysis(
                summary=summary,
                temperature_range=temp_range,
                precipitation_chance=avg_precipitation,
                recommended_clothing=clothing
            )
            
        except Exception as e:
            print(f"⚠️  Error parsing MCP weather data: {e}")
            # Fallback to enhanced mock
            return get_weather_mock(context.query.location, context.query.start_date, context.query.end_date)

    async def _search_for_activities(self, context: TripContext, weather_info: WeatherAnalysis) -> tuple[SearchResult, str]:
        """Search for activities, handling kid-friendly routing."""
        tracer = get_enhanced_tracer()
        
        if not self.agents_initialized:
            fallback_results = self._create_fallback_search_results(context)
            return fallback_results, "Fallback Agent"
        
        # Check if we have children in the group
        has_children = any(age < CHILD_AGE_THRESHOLD for age in context.query.participant_ages)
        
        try:
            if has_children:
                # Log handoff decision
                tracer.log_handoff("Activity Router", "Kid-Friendly Agent", 
                                 f"Children detected (ages: {[age for age in context.query.participant_ages if age < CHILD_AGE_THRESHOLD]})")
                
                print(f"👶 Children detected (under {CHILD_AGE_THRESHOLD}). Using kid-friendly search...")
                agent_key = tracer.start_agent("Kid-Friendly Agent", "Find family-friendly activities", {
                    "children_ages": [age for age in context.query.participant_ages if age < CHILD_AGE_THRESHOLD],
                    "location": context.query.location
                })
                
                search_results = await self.kid_friendly_agent.find_kid_friendly_activities(
                    context, weather_info.summary
                )
                
                tracer.complete_agent(agent_key, "Kid-Friendly Agent", f"Found {len(search_results.activities)} kid-friendly activities", {
                    "activity_count": len(search_results.activities),
                    "search_summary": search_results.search_summary
                })
                return search_results, "Kid-Friendly Agent"
            else:
                print("👨‍👩‍👧‍👦 Adult group detected. Using general activity search...")
                agent_key = tracer.start_agent("Activity Search Agent", "Find general activities", {
                    "adult_ages": context.query.participant_ages,
                    "location": context.query.location
                })
                
                tracer.log_tool_call("Activity Search Agent", "search_activities", {
                    "context": "Trip context",
                    "weather_summary": weather_info.summary[:100] + "..."
                })
                
                search_results = await self.activity_search_agent.search_activities(
                    context, weather_info.summary
                )
                
                tracer.complete_agent(agent_key, "Activity Search Agent", f"Found {len(search_results.activities)} activities", {
                    "activity_count": len(search_results.activities),
                    "search_summary": search_results.search_summary
                })
                return search_results, "Activity Search Agent"
                
        except Exception as e:
            print(f"Activity search failed: {e}. Using fallback.")
            tracer.log_error("Activity Search Router", str(e))
            return self._create_fallback_search_results(context), "Fallback Agent"

    async def _generate_trip_plan(
        self, 
        search_results: SearchResult, 
        weather_info: WeatherAnalysis, 
        context: TripContext
    ) -> TripPlan:
        """Generate the final trip plan."""
        tracer = get_enhanced_tracer()
        
        if not self.agents_initialized:
            return self._create_fallback_plan(context)
        
        agent_key = tracer.start_agent("Recommendation Agent", "Create comprehensive trip plan", {
            "activities_to_process": len(search_results.activities),
            "location": context.query.location,
            "participants": context.query.participant_number
        })
        
        try:
            tracer.log_tool_call("Recommendation Agent", "create_trip_plan", {
                "search_results_count": len(search_results.activities),
                "weather_temp_range": weather_info.temperature_range,
                "context": "Trip context"
            })
            
            trip_plan = await self.recommendation_agent.create_trip_plan(
                context, search_results, weather_info
            )
            
            tracer.log_response("Recommendation Agent", 
                              f"Generated trip plan with {len(trip_plan.recommended_activities)} activities", 
                              len(str(trip_plan)))
            
            tracer.complete_agent(agent_key, "Recommendation Agent", 
                                f"Created trip plan with {len(trip_plan.recommended_activities)} recommendations", {
                                    "recommended_activities": len(trip_plan.recommended_activities),
                                    "packing_items": len(trip_plan.packing_list),
                                    "general_tips": len(trip_plan.general_tips)
                                })
            return trip_plan
        except Exception as e:
            print(f"Trip plan generation failed: {e}. Creating fallback plan.")
            tracer.log_error("Recommendation Agent", str(e))
            tracer.complete_agent(agent_key, "Recommendation Agent", "Failed - using fallback plan", {"error": str(e)})
            return self._create_fallback_plan(context)

    def _create_fallback_search_results(self, context: TripContext) -> SearchResult:
        """Create basic fallback search results."""
        from models_gcp import ActivityResult
        
        activities = [
            ActivityResult(
                name=f"Explore Downtown {context.query.location}",
                description="Walk around the city center and discover local attractions",
                location=context.query.location,
                age_range=[5, 99],
                price_range="Free",
                duration="2-4 hours",
                weather_dependent=True,
                source_url=None
            ),
            ActivityResult(
                name=f"Visit Local Museum",
                description="Learn about local history and culture",
                location=context.query.location,
                age_range=[8, 99],
                price_range="$10-20",
                duration="1-3 hours",
                weather_dependent=False,
                source_url=None
            ),
            ActivityResult(
                name=f"Local Park and Recreation",
                description="Enjoy outdoor activities and green spaces",
                location=context.query.location,
                age_range=[3, 99],
                price_range="Free",
                duration="2-4 hours",
                weather_dependent=True,
                source_url=None
            ),
            ActivityResult(
                name=f"Shopping and Local Markets",
                description="Browse local shops, markets, and unique boutiques",
                location=context.query.location,
                age_range=[10, 99],
                price_range="$20-100",
                duration="2-3 hours",
                weather_dependent=False,
                source_url=None
            ),
            ActivityResult(
                name=f"Cultural Center or Gallery",
                description="Experience local art, culture, and community events",
                location=context.query.location,
                age_range=[12, 99],
                price_range="$5-25",
                duration="1-2 hours",
                weather_dependent=False,
                source_url=None
            )
        ]
        
        return SearchResult(
            activities=activities,
            search_summary=f"Basic activity suggestions for {context.query.location}"
        )

    def _create_fallback_plan(self, context: TripContext) -> TripPlan:
        """Create a basic fallback trip plan."""
        from models_gcp import ActivityRecommendation
        
        # Basic weather info
        weather_summary = f"Weather information for {context.query.location} during your visit"
        
        # Basic activity recommendation
        activities = [
            ActivityRecommendation(
                name=f"Explore {context.query.location}",
                description="Discover the local area and main attractions",
                reasoning="A versatile activity suitable for most travelers",
                best_time="Morning or afternoon",
                weather_considerations=["Check local weather forecast"],
                preparation_tips=["Wear comfortable shoes", "Bring water"],
                source_url=None
            )
        ]
        
        # Basic packing list
        packing_list = [
            "Comfortable walking shoes",
            "Weather-appropriate clothing",
            "Water bottle",
            "Snacks",
            "Phone and charger",
            "Camera",
            "Cash and cards"
        ]
        
        # Basic tips
        general_tips = [
            "Check local weather before heading out",
            "Research local customs and etiquette",
            "Keep emergency contacts handy",
            "Stay hydrated and take breaks as needed",
            "Have a wonderful trip!"
        ]
        
        return TripPlan(
            location=context.query.location,
            dates=f"{context.query.start_date} to {context.query.end_date}",
            participants_summary=f"{context.query.participant_number} participants",
            weather_summary=weather_summary,
            recommended_activities=activities,
            packing_list=packing_list,
            general_tips=general_tips
        )

    def print_trip_plan(self, plan: TripPlan) -> None:
        """Print the final trip plan in a beautiful format."""
        print("\n" + "="*60)
        print("🗺️  YOUR PERSONALIZED ADVENTURE PLAN")
        print("="*60)
        
        print(f"\n📍 Destination: {plan.location}")
        print(f"📅 Dates: {plan.dates}")
        print(f"👥 Participants: {plan.participants_summary}")
        
        print(f"\n🌤️  Weather Overview:")
        print(f"   {plan.weather_summary}")
        
        print(f"\n🎯 Recommended Activities:")
        if not plan.recommended_activities:
            print("   • No specific activities recommended")
        else:
            for i, activity in enumerate(plan.recommended_activities, 1):
                print(f"\n   {i}. {activity.name}")
                print(f"      📝 {activity.description}")
                print(f"      💡 Why: {activity.reasoning}")
                if activity.best_time:
                    print(f"      ⏰ Best time: {activity.best_time}")
                if activity.weather_considerations:
                    print(f"      🌦️  Weather notes:")
                    for consideration in activity.weather_considerations:
                        print(f"         • {consideration}")
                if activity.preparation_tips:
                    print(f"      🎒 Prep tips:")
                    for tip in activity.preparation_tips:
                        print(f"         • {tip}")
        
        print(f"\n🎒 Packing Checklist:")
        for item in plan.packing_list:
            print(f"   ☐ {item}")
        
        print(f"\n💡 Pro Tips:")
        for tip in plan.general_tips:
            print(f"   • {tip}")
        
        print("\n" + "="*60)
        print("✨ Have an amazing adventure!")
        print("="*60)
