"""
Main entry point for the GCP-powered multi-agent travel planner.
"""

import asyncio
import os
from dotenv import load_dotenv
from manager_gcp import AdventureManagerGCP
from models_gcp import TripQuery

# Load environment variables
load_dotenv()


async def main():
    """Run the GCP-powered adventure planning system."""
    print("🌟 Welcome to the GCP-Powered Adventure Planner!")
    print("   Powered by Google Gemini and Cloud Platform")
    print("-" * 50)
    
    # Get API key from environment
    api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    if not api_key:
        print("⚠️  No Google AI API key found!")
        print("   Please set GOOGLE_AI_API_KEY or GEMINI_API_KEY in your .env file")
        print("   The system will run with mock data for demonstration purposes.")
        print()
    
    # Initialize the manager
    manager = AdventureManagerGCP(api_key=api_key, project_id=project_id)
    
    # Example trip query
    query = TripQuery(
        start_date="2024-02-15",
        end_date="2024-02-18",
        location="San Francisco, CA",
        participant_number=4,
        participant_ages=[8, 10, 35, 37]  # Family with children
    )
    
    print("📋 Example Trip Planning:")
    print(f"   📍 Destination: {query.location}")
    print(f"   📅 Dates: {query.start_date} to {query.end_date}")
    print(f"   👥 Participants: {query.participant_number} people (ages: {query.participant_ages})")
    print()
    
    try:
        # Run the planning workflow
        trip_plan = await manager.run(query, use_real_weather=False)
        
        # Display the results
        manager.print_trip_plan(trip_plan)
        
    except KeyboardInterrupt:
        print("\n👋 Planning interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("   Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())
