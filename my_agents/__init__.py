"""Agent definitions for the AdventureBot application"""

# Import the consolidated agent creation functions and their output types
from .weather_agent import create_weather_agent, WeatherAnalysis
from .search_agent import create_activity_search_agent
from .recommender_agent import create_recommendation_agent, TripPlan, ActivityRecommendation
from .kid_friendly_agent import create_kid_friendly_activity_agent

# Import models that are referenced in manager.py
from models import SearchResult

__all__ = [
    # Agent creation functions
    'create_weather_agent',
    'create_activity_search_agent',
    'create_recommendation_agent',
    'create_kid_friendly_activity_agent',

    # Model types (consolidated)
    'WeatherAnalysis',
    'SearchResult',
    'TripPlan',
    'ActivityRecommendation',
]