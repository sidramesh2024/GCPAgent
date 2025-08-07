#!/usr/bin/env python3
"""
Simplified multi-agent application that works without MCP weather server
"""
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from models import TripQuery, TripContext
from my_agents import create_activity_search_agent, create_recommendation_agent
import agents as openai_agents

# Create simple classes for the workflow
Runner = openai_agents.Runner
trace = openai_agents.trace
gen_trace_id = openai_agents.gen_trace_id


class SimpleWeatherAnalysis:
    """Mock weather analysis for demo purposes"""
    def __init__(self, location: str):
        self.summary = f"Expected mild weather for {location} during your trip dates. Pack layers and check forecast closer to departure."
        self.temperature_range = [15.0, 25.0]  # Celsius
        self.precipitation_chance = 0.3  # 30%
        self.recommended_clothing = [
            "Light jacket or sweater",
            "Comfortable walking shoes", 
            "Light rain jacket or umbrella",
            "Layered clothing"
        ]
        self.weather_warnings = ["Check local forecast before outdoor activities"]


class SimpleAdventureManager:
    """Simplified adventure manager without MCP weather dependency"""

    def __init__(self):
        self.activity_search_agent = create_activity_search_agent()
        self.recommendation_agent = create_recommendation_agent()

    async def run(self, query: TripQuery) -> None:
        """Run the simplified adventure planning workflow"""
        trace_id = gen_trace_id()
        print(f"🚀 Starting simplified adventure planning... (Trace ID: {trace_id})")
        print(f"🔗 View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")

        # Create the context object
        trip_context = TripContext(query=query)

        with trace("Adventure Planning (Simplified)", trace_id=trace_id):
            # 1. Get Mock Weather Information
            weather_info = self._get_simple_weather_info(trip_context)
            print("✅ Weather analysis complete (using mock data)")

            # 2. Search for Activities
            print("🔍 Searching for activities...")
            search_results, search_agent_used = await self._search_for_activities(trip_context, weather_info)
            print(f"✅ Activity search complete using {search_agent_used.name}")

            # 3. Generate Trip Plan
            print("📋 Generating trip recommendations...")
            trip_plan = await self._generate_trip_plan(search_results, weather_info, trip_context)
            print("✅ Trip plan generated")

            # Display the final trip plan
            self._print_trip_plan(trip_plan)

    def _get_simple_weather_info(self, context: TripContext):
        """Get simple weather information without MCP server"""
        print(f"🌤️  Getting weather info for {context.query.location}...")
        return SimpleWeatherAnalysis(context.query.location)

    async def _search_for_activities(self, context: TripContext, weather_info):
        """Run the ActivitySearchAgent"""
        participants_str = f"{context.query.participant_number} participants (ages: {context.query.participant_ages})"
        input_str = (
            f"Find activities for a trip to {context.query.location} "
            f"from {context.query.start_date} to {context.query.end_date} "
            f"for {participants_str}.\n\n"
            f"Consider the following weather summary:\n{weather_info.summary}"
        )

        # Run the search agent
        result = await Runner.run(
            self.activity_search_agent,
            input_str,
            context=context
        )

        search_results = result.final_output
        final_agent = result.last_agent

        return search_results, final_agent

    async def _generate_trip_plan(self, search_results, weather_info, context: TripContext):
        """Run the RecommendationAgent to create the final plan"""
        participants_str = f"{context.query.participant_number} participants (ages: {context.query.participant_ages})"
        dates_str = f"{context.query.start_date} to {context.query.end_date}"
        
        # Create weather info dict for the prompt
        weather_dict = {
            "summary": weather_info.summary,
            "temperature_range": weather_info.temperature_range,
            "precipitation_chance": weather_info.precipitation_chance,
            "recommended_clothing": weather_info.recommended_clothing,
            "weather_warnings": weather_info.weather_warnings
        }
        
        input_str = (
            f"Create a trip plan for {context.query.location} from {dates_str} "
            f"for {participants_str}.\n\n"
            f"Weather Information:\n{weather_dict}\n\n"
            f"Potential Activities Found:\n{search_results}"
        )

        result = await Runner.run(
            self.recommendation_agent,
            input_str,
            context=context
        )

        return result.final_output

    def _print_trip_plan(self, plan) -> None:
        """Print the final trip plan in a structured format"""
        print("\n" + "="*50)
        print("🎒 YOUR ADVENTURE PLAN")
        print("="*50)
        
        if hasattr(plan, 'location'):
            print(f"📍 Location: {plan.location}")
            print(f"📅 Dates: {plan.dates}")
            print(f"👥 Participants: {plan.participants_summary}")
            print(f"\n🌤️  Weather Summary:\n{plan.weather_summary}")
            
            if hasattr(plan, 'recommended_activities') and plan.recommended_activities:
                print(f"\n🎯 Recommended Activities ({len(plan.recommended_activities)}):")
                for i, activity in enumerate(plan.recommended_activities, 1):
                    print(f"\n{i}. 🏃 {activity.name}")
                    print(f"   📝 {activity.description}")
                    print(f"   💭 Why: {activity.reasoning}")
                    if hasattr(activity, 'best_time') and activity.best_time:
                        print(f"   ⏰ Best Time: {activity.best_time}")
                    if hasattr(activity, 'source_url') and activity.source_url:
                        print(f"   🔗 More Info: {activity.source_url}")
            
            if hasattr(plan, 'packing_list') and plan.packing_list:
                print(f"\n🎒 Packing List:")
                for item in plan.packing_list:
                    print(f"   • {item}")
            
            if hasattr(plan, 'general_tips') and plan.general_tips:
                print(f"\n💡 General Tips:")
                for tip in plan.general_tips:
                    print(f"   • {tip}")
        else:
            # Fallback for different response formats
            print(f"📋 Trip Plan Generated:\n{plan}")
        
        print("\n" + "="*50)
        print("✨ Happy travels! ✨")
        print("="*50)


async def main() -> None:
    """
    Main entry point for the simplified AdventureBot application.
    """
    print("🎉 Welcome to the Adventure Planning Multi-Agent System!")
    print("📍 Planning a sample trip...")
    
    # Sample trip query data
    query = TripQuery(
        start_date="2025-12-01",
        end_date="2025-12-14",
        location="Toronto",
        participant_number=2,
        participant_ages=[32, 35]
    )
    
    print(f"\n🗺️  Trip Details:")
    print(f"   📍 Destination: {query.location}")
    print(f"   📅 Dates: {query.start_date} to {query.end_date}")
    print(f"   👥 Travelers: {query.participant_number} people (ages: {query.participant_ages})")
    
    # Initialize and run the adventure manager
    manager = SimpleAdventureManager()
    await manager.run(query)


if __name__ == "__main__":
    asyncio.run(main())
