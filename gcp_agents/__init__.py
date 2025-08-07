"""
Google Cloud Platform agent definitions for the AdventureBot application.
Uses Google's Agent Development Kit and Gemini models.
"""

# Import the GCP agent creation functions and their output types
from .weather_agent_gcp import create_weather_agent_gcp, get_weather_mock
from .search_agent_gcp import create_activity_search_agent_gcp
from .recommender_agent_gcp import create_recommendation_agent_gcp
from .kid_friendly_agent_gcp import create_kid_friendly_activity_agent_gcp

# Import models that are referenced in manager
from models_gcp import (
    WeatherAnalysis,
    SearchResult,
    TripPlan,
    ActivityRecommendation,
    ActivityResult,
    TripContext,
    TripQuery
)

__all__ = [
    # Agent creation functions
    'create_weather_agent_gcp',
    'create_activity_search_agent_gcp',
    'create_recommendation_agent_gcp',
    'create_kid_friendly_activity_agent_gcp',
    
    # Utility functions
    'get_weather_mock',

    # Model types
    'WeatherAnalysis',
    'SearchResult',
    'TripPlan',
    'ActivityRecommendation',
    'ActivityResult',
    'TripContext',
    'TripQuery',
]
