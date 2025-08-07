#!/usr/bin/env python3
"""
Simple test script to verify the multi-agent system works
without the weather MCP server dependency
"""
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from models import TripQuery, TripContext
from my_agents import create_activity_search_agent, create_recommendation_agent

async def test_agents():
    """Test the search and recommendation agents without weather dependency"""
    print("🚀 Testing Multi-Agent System...")
    
    # Create a sample trip query
    query = TripQuery(
        start_date="2025-12-01",
        end_date="2025-12-14", 
        location="Toronto",
        participant_number=2,
        participant_ages=[32, 35]
    )
    
    context = TripContext(query=query)
    
    print(f"📍 Planning trip to {query.location}")
    print(f"📅 Dates: {query.start_date} to {query.end_date}")
    print(f"👥 Participants: {query.participant_number} people (ages: {query.participant_ages})")
    
    # Create agents
    search_agent = create_activity_search_agent()
    recommendation_agent = create_recommendation_agent()
    
    print(f"\n✅ Search Agent: {search_agent.name}")
    print(f"✅ Recommendation Agent: {recommendation_agent.name}")
    
    print("\n🎉 Multi-agent system initialized successfully!")
    print("\n💡 To run the full system with weather data:")
    print("   1. Ensure Docker is running")
    print("   2. Build weather MCP server: cd mcp_server_weather && docker build -t mcp_server_weather .")
    print("   3. Run: python main.py")

if __name__ == "__main__":
    asyncio.run(test_agents())
