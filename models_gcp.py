"""
Google Cloud Platform compatible models for the multi-agent travel planner.
These models work with Google's Agent Development Kit and Gemini models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# Define shared constants
CHILD_AGE_THRESHOLD = 12


class TripQuery(BaseModel):
    """Input data structure for adventure planning"""
    start_date: str  # YYYY-MM-DD format
    end_date: str    # YYYY-MM-DD format
    location: str
    participant_number: int
    participant_ages: List[int]


class TripContext(BaseModel):
    """Context object holding trip details and derived information."""
    query: TripQuery
    meets_child_threshold: bool = Field(default=False, description="Flag indicating if any participant meets the child age threshold.")


class ActivityResult(BaseModel):
    """Structured result of an activity search"""
    name: str
    description: str
    location: str
    age_range: Optional[List[int]] = None  # [min_age, max_age] if applicable
    price_range: Optional[str] = None
    duration: Optional[str] = None
    weather_dependent: bool
    source_url: Optional[str] = None


class SearchResult(BaseModel):
    """Collection of activities found from search"""
    activities: List[ActivityResult]
    search_summary: str


class WeatherAnalysis(BaseModel):
    """Weather analysis with recommendations"""
    summary: str
    temperature_range: List[float]  # [min_temp, max_temp]
    precipitation_chance: float
    recommended_clothing: List[str]
    weather_warnings: Optional[List[str]] = None


class ActivityRecommendation(BaseModel):
    """Recommended activity with evaluation and context"""
    name: str
    description: str
    reasoning: str
    best_time: Optional[str] = None
    weather_considerations: Optional[List[str]] = None
    preparation_tips: Optional[List[str]] = None
    source_url: Optional[str] = None


class TripPlan(BaseModel):
    """Complete trip plan with recommendations"""
    location: str
    dates: str
    participants_summary: str
    weather_summary: str
    recommended_activities: List[ActivityRecommendation]
    packing_list: List[str]
    general_tips: List[str]
