"""
Google Cloud Platform Search Agent using Gemini and web search capabilities.
"""

import json
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import requests
from bs4 import BeautifulSoup
from models_gcp import TripContext, SearchResult, ActivityResult, CHILD_AGE_THRESHOLD


def web_search_tool(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Mock web search function - replace with actual search API."""
    # This is a mock implementation
    # In production, use Google Custom Search API, Serp API, or similar
    
    # Enhanced mock results with more variety
    activity_types = [
        "museum", "park", "restaurant", "gallery", "theater", 
        "landmark", "tour", "market", "beach", "hiking trail",
        "shopping center", "entertainment venue", "cultural site"
    ]
    
    mock_results = []
    for i in range(1, min(num_results + 5, 15) + 1):  # Generate a few extra for better selection
        activity_type = activity_types[(i-1) % len(activity_types)]
        mock_results.append({
            "title": f"{activity_type.title()} Experience in {query.split()[-1] if query.split() else 'City'}",
            "url": f"https://example.com/{activity_type}-{i}",
            "snippet": f"Discover this amazing {activity_type} offering unique experiences for visitors. Perfect for {query}.",
            "content": f"Detailed information about this popular {activity_type} attraction with visitor reviews and practical details."
        })
    
    return mock_results[:num_results]


def check_child_threshold(participant_ages: List[int], threshold: int = CHILD_AGE_THRESHOLD) -> Dict[str, Any]:
    """Check if any participant is under the child age threshold."""
    has_children = any(age < threshold for age in participant_ages)
    children_ages = [age for age in participant_ages if age < threshold]
    
    return {
        "meets_threshold": has_children,
        "children_ages": children_ages,
        "threshold": threshold,
        "recommendation": "kid_friendly_agent" if has_children else "continue_general_search"
    }


class SearchAgentGCP:
    """Google Cloud Platform Search Agent using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the search agent with Gemini model."""
        self.model_name = model_name
        
        # Define search tools for function calling
        self.search_tools = [
            Tool(function_declarations=[
                FunctionDeclaration(
                    name="web_search_tool",
                    description="Search the web for activities and attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for activities"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of search results to return"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                FunctionDeclaration(
                    name="check_child_threshold",
                    description="Check if any participant is under the child age threshold",
                    parameters={
                        "type": "object",
                        "properties": {
                            "participant_ages": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of participant ages"
                            },
                            "threshold": {
                                "type": "integer",
                                "description": "Age threshold for children"
                            }
                        },
                        "required": ["participant_ages"]
                    }
                )
            ])
        ]
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=self.search_tools
        )
    
    def _execute_function_call(self, function_call) -> Dict[str, Any]:
        """Execute a function call and return the result."""
        function_name = function_call.name
        function_args = {key: value for key, value in function_call.args.items()}
        
        if function_name == "web_search_tool":
            # Set default for num_results if not provided
            if "num_results" not in function_args:
                function_args["num_results"] = 5
            return {"results": web_search_tool(**function_args)}
        elif function_name == "check_child_threshold":
            # Set default for threshold if not provided
            if "threshold" not in function_args:
                function_args["threshold"] = CHILD_AGE_THRESHOLD
            return check_child_threshold(**function_args)
        else:
            raise ValueError(f"Unknown function: {function_name}")
    
    async def search_activities(self, context: TripContext, weather_summary: str) -> SearchResult:
        """Search for activities based on trip context and weather."""
        
        prompt = f"""
You are an activity search specialist helping travelers find suitable activities.

Trip Details:
- Destination: {context.query.location}
- Dates: {context.query.start_date} to {context.query.end_date}
- Participants: {context.query.participant_number} people
- Ages: {context.query.participant_ages}

Weather Summary: {weather_summary}

Please help find activities by following these steps:

1. FIRST: Use check_child_threshold to determine if any participant is under {CHILD_AGE_THRESHOLD} years old
2. If children are present (threshold met), recommend switching to kid-friendly search mode
3. If no young children, proceed with general activity search:
   - Create 3-5 relevant search queries for activities in {context.query.location}
   - Use web_search_tool to find activities
   - Extract key information: name, description, location, age suitability, pricing, duration, weather dependency
   - Focus on activities suitable for the age group and weather conditions

Return a structured list of activities with detailed information and a search summary.
"""

        try:
            # Start conversation with the model
            chat = self.model.start_chat()
            response = chat.send_message(prompt)
            
            search_results = []
            child_check_result = None
            
            # Handle function calls
            while response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                if hasattr(part, 'function_call') and part.function_call:
                    # Execute the function call
                    function_result = self._execute_function_call(part.function_call)
                    
                    # Store child check result for decision making
                    if part.function_call.name == "check_child_threshold":
                        child_check_result = function_result
                        
                        # If children are present, return early with recommendation
                        if function_result.get("meets_threshold", False):
                            return SearchResult(
                                activities=[],
                                search_summary=f"Children detected (ages: {function_result.get('children_ages', [])}). "
                                             f"Recommend using kid-friendly activity search for better results."
                            )
                    
                    # Store web search results
                    elif part.function_call.name == "web_search_tool":
                        search_results.extend(function_result.get("results", []))
                    
                    # Send the function result back to the model
                    response = chat.send_message(
                        f"Function {part.function_call.name} result: {function_result}"
                    )
                else:
                    # Extract final response
                    response_text = part.text
                    return self._parse_search_response(response_text, search_results, context)
            
            # Fallback response
            return self._create_fallback_response(context, search_results)
            
        except Exception as e:
            print(f"Activity search error: {e}")
            return self._create_fallback_response(context, [])
    
    def _parse_search_response(self, response_text: str, search_results: List[Dict], context: TripContext) -> SearchResult:
        """Parse the model's response into a SearchResult object."""
        try:
            # Convert search results to ActivityResult objects
            activities = []
            
            for i, result in enumerate(search_results[:15]):  # Limit to 15 activities for better selection
                activity = ActivityResult(
                    name=result.get("title", f"Activity {i+1}"),
                    description=result.get("snippet", "Activity description"),
                    location=context.query.location,
                    age_range=None,  # Could be extracted from content analysis
                    price_range="Varies",
                    duration="2-4 hours",
                    weather_dependent=True,  # Conservative assumption
                    source_url=result.get("url")
                )
                activities.append(activity)
            
            # Create summary
            summary = f"Found {len(activities)} activities in {context.query.location} suitable for {context.query.participant_number} participants."
            
            return SearchResult(
                activities=activities,
                search_summary=summary
            )
            
        except Exception as e:
            print(f"Error parsing search response: {e}")
            return self._create_fallback_response(context, search_results)
    
    def _create_fallback_response(self, context: TripContext, search_results: List[Dict]) -> SearchResult:
        """Create a fallback response when search fails."""
        # Create basic activities based on location
        activities = [
            ActivityResult(
                name=f"Walking Tour of {context.query.location}",
                description="Explore the city center and main attractions on foot",
                location=context.query.location,
                age_range=[5, 99],
                price_range="Free - $30",
                duration="2-3 hours",
                weather_dependent=True,
                source_url=None
            ),
            ActivityResult(
                name=f"Local Museum Visit",
                description="Visit a popular local museum or cultural center",
                location=context.query.location,
                age_range=[8, 99],
                price_range="$10 - $25",
                duration="1-2 hours",
                weather_dependent=False,
                source_url=None
            ),
            ActivityResult(
                name=f"Scenic Viewpoint or Landmark",
                description="Visit iconic landmarks and scenic viewing spots",
                location=context.query.location,
                age_range=[5, 99],
                price_range="Free - $15",
                duration="1-2 hours",
                weather_dependent=True,
                source_url=None
            ),
            ActivityResult(
                name=f"Local Food Experience",
                description="Try local cuisine, food markets, or popular restaurants",
                location=context.query.location,
                age_range=[12, 99],
                price_range="$15 - $50",
                duration="1-3 hours",
                weather_dependent=False,
                source_url=None
            ),
            ActivityResult(
                name=f"Entertainment District",
                description="Explore entertainment areas, theaters, or nightlife",
                location=context.query.location,
                age_range=[18, 99],
                price_range="$20 - $80",
                duration="2-4 hours",
                weather_dependent=False,
                source_url=None
            )
        ]
        
        return SearchResult(
            activities=activities,
            search_summary=f"Fallback activities suggested for {context.query.location}"
        )


class KidFriendlySearchAgentGCP:
    """Kid-friendly version of the search agent."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the kid-friendly search agent."""
        self.model_name = model_name
        self.search_tools = [
            Tool(function_declarations=[
                FunctionDeclaration(
                    name="web_search_tool",
                    description="Search for kid-friendly activities",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for family-friendly activities"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ])
        ]
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            tools=self.search_tools
        )
    
    async def search_kid_friendly_activities(self, context: TripContext, weather_summary: str) -> SearchResult:
        """Search specifically for kid-friendly activities."""
        children_ages = [age for age in context.query.participant_ages if age < CHILD_AGE_THRESHOLD]
        
        prompt = f"""
You are a family travel specialist finding kid-friendly activities.

Trip Details:
- Destination: {context.query.location}
- Children ages: {children_ages}
- All participant ages: {context.query.participant_ages}
- Weather: {weather_summary}

Find family-friendly activities that are:
- Safe and engaging for children ages {children_ages}
- Educational or entertaining
- Suitable for the weather conditions
- Accessible for families

Use web_search_tool to find specific activities and return detailed information.
"""

        try:
            chat = self.model.start_chat()
            response = chat.send_message(prompt)
            
            search_results = []
            
            # Handle function calls similar to main search agent
            while response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                if hasattr(part, 'function_call') and part.function_call:
                    function_result = web_search_tool(**{key: value for key, value in part.function_call.args.items()})
                    search_results.extend(function_result)
                    
                    response = chat.send_message(
                        f"Function {part.function_call.name} result: {function_result}"
                    )
                else:
                    # Create kid-friendly activities
                    activities = [
                        ActivityResult(
                            name=f"Children's Museum in {context.query.location}",
                            description="Interactive exhibits designed for young learners",
                            location=context.query.location,
                            age_range=[3, 12],
                            price_range="$8 - $15",
                            duration="2-3 hours",
                            weather_dependent=False,
                            source_url=None
                        ),
                        ActivityResult(
                            name=f"Family Park Visit",
                            description="Safe playground and green space for children to play",
                            location=context.query.location,
                            age_range=[2, 16],
                            price_range="Free",
                            duration="1-4 hours",
                            weather_dependent=True,
                            source_url=None
                        )
                    ]
                    
                    return SearchResult(
                        activities=activities,
                        search_summary=f"Kid-friendly activities found for children ages {children_ages}"
                    )
            
        except Exception as e:
            print(f"Kid-friendly search error: {e}")
            
        # Fallback
        return SearchResult(
            activities=[],
            search_summary="Kid-friendly search completed with basic recommendations"
        )


def create_activity_search_agent_gcp() -> SearchAgentGCP:
    """Create a GCP-powered activity search agent."""
    return SearchAgentGCP()


def create_kid_friendly_activity_agent_gcp() -> KidFriendlySearchAgentGCP:
    """Create a GCP-powered kid-friendly activity search agent."""
    return KidFriendlySearchAgentGCP()
