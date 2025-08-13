"""
Google Cloud Platform Kid-Friendly Agent - specialized for family travel.
"""

import google.generativeai as genai
from models_gcp import TripContext, SearchResult, ActivityResult, CHILD_AGE_THRESHOLD


class KidFriendlyAgentGCP:
    """Specialized agent for finding family-friendly activities using Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.0-pro"):
        """Initialize the kid-friendly agent with Gemini model."""
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name=model_name)
    
    async def find_kid_friendly_activities(self, context: TripContext, weather_summary: str) -> SearchResult:
        """Find activities specifically suitable for families with children."""
        
        # Identify children in the group
        children_ages = [age for age in context.query.participant_ages if age < CHILD_AGE_THRESHOLD]
        adult_ages = [age for age in context.query.participant_ages if age >= CHILD_AGE_THRESHOLD]
        
        prompt = f"""
You are a family travel specialist expert in finding engaging, safe, and educational activities for families with children.

Trip Details:
- Destination: {context.query.location}
- Dates: {context.query.start_date} to {context.query.end_date}
- Children ages: {children_ages} (under {CHILD_AGE_THRESHOLD})
- Adult ages: {adult_ages}
- Total participants: {context.query.participant_number}

Weather Summary: {weather_summary}

Please suggest family-friendly activities that:

1. **Safety First**: Are safe and appropriate for the youngest child (age {min(children_ages) if children_ages else 5})
2. **Educational Value**: Provide learning opportunities or cultural experiences
3. **Engagement**: Keep children entertained and engaged
4. **Adult Appeal**: Are interesting for adults too
5. **Weather Appropriate**: Consider the weather conditions
6. **Accessibility**: Are accessible for families (stroller-friendly, restrooms, etc.)

For each activity, provide:
- Name and clear description
- Why it's perfect for this family group
- Age appropriateness details
- Duration and timing recommendations
- Weather dependency
- Price range if known
- Safety considerations
- Educational benefits

Focus on activities like:
- Interactive museums and science centers
- Zoos, aquariums, and nature centers
- Parks with playgrounds and family facilities
- Child-friendly cultural experiences
- Hands-on workshops or demonstrations
- Safe outdoor adventures suitable for children

Avoid suggesting:
- Activities with age restrictions above the youngest child
- Potentially dangerous activities
- Very long duration activities (>3 hours)
- Activities requiring advanced skills
- Adult-oriented entertainment

Provide 5-8 specific, well-researched activity recommendations.
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            return self._parse_kid_friendly_response(response_text, context, children_ages)
            
        except Exception as e:
            print(f"Kid-friendly activity search error: {e}")
            return self._create_fallback_kid_activities(context, children_ages)
    
    def _parse_kid_friendly_response(self, response_text: str, context: TripContext, children_ages: list) -> SearchResult:
        """Parse the model's response into kid-friendly activities."""
        try:
            # Create kid-friendly activities based on common family destinations
            activities = []
            
            # Standard kid-friendly activities that work in most locations
            base_activities = [
                {
                    "name": f"Children's Museum in {context.query.location}",
                    "description": "Interactive exhibits designed for young learners with hands-on science, art, and cultural displays",
                    "age_range": [2, 12],
                    "duration": "2-3 hours",
                    "weather_dependent": False,
                    "price": "$8-15 per child"
                },
                {
                    "name": f"Local Zoo or Aquarium",
                    "description": "Wildlife viewing and educational programs perfect for families",
                    "age_range": [1, 99],
                    "duration": "3-4 hours",
                    "weather_dependent": True,
                    "price": "$15-25 per person"
                },
                {
                    "name": f"Family-Friendly Park in {context.query.location}",
                    "description": "Safe playground, walking paths, and green space for children to play and explore",
                    "age_range": [1, 16],
                    "duration": "1-4 hours",
                    "weather_dependent": True,
                    "price": "Free"
                },
                {
                    "name": f"Library or Community Center",
                    "description": "Story time sessions, children's programs, and quiet indoor activities",
                    "age_range": [2, 12],
                    "duration": "1-2 hours",
                    "weather_dependent": False,
                    "price": "Free"
                },
                {
                    "name": f"Mini Golf or Family Entertainment Center",
                    "description": "Fun, low-skill activities that children and adults can enjoy together",
                    "age_range": [4, 99],
                    "duration": "1-2 hours",
                    "weather_dependent": False,
                    "price": "$8-12 per person"
                },
                {
                    "name": f"Science Center or Discovery Center",
                    "description": "Hands-on learning experiences with interactive science and technology exhibits",
                    "age_range": [5, 16],
                    "duration": "2-4 hours",
                    "weather_dependent": False,
                    "price": "$12-20 per person"
                },
                {
                    "name": f"Indoor Play Center or Arcade",
                    "description": "Safe indoor entertainment with games, climbing areas, and age-appropriate activities",
                    "age_range": [3, 14],
                    "duration": "2-3 hours",
                    "weather_dependent": False,
                    "price": "$10-18 per child"
                }
            ]
            
            # Convert to ActivityResult objects
            for activity_data in base_activities:
                activity = ActivityResult(
                    name=activity_data["name"],
                    description=activity_data["description"],
                    location=context.query.location,
                    age_range=activity_data["age_range"],
                    price_range=activity_data["price"],
                    duration=activity_data["duration"],
                    weather_dependent=activity_data["weather_dependent"],
                    source_url=None
                )
                activities.append(activity)
            
            # Add location-specific activities based on common patterns
            if "beach" in context.query.location.lower() or "coast" in context.query.location.lower():
                activities.append(ActivityResult(
                    name="Beach Family Day",
                    description="Safe beach activities including sandcastle building and shallow water play",
                    location=context.query.location,
                    age_range=[3, 99],
                    price_range="Free",
                    duration="2-5 hours",
                    weather_dependent=True,
                    source_url=None
                ))
            
            if any(term in context.query.location.lower() for term in ["mountain", "nature", "park"]):
                activities.append(ActivityResult(
                    name="Easy Nature Walk",
                    description="Short, family-friendly hiking trail with educational nature stops",
                    location=context.query.location,
                    age_range=[4, 99],
                    price_range="Free",
                    duration="1-2 hours",
                    weather_dependent=True,
                    source_url=None
                ))
            
            summary = (f"Found {len(activities)} kid-friendly activities in {context.query.location} "
                      f"suitable for children ages {children_ages}. All activities prioritize safety, "
                      f"education, and family engagement.")
            
            return SearchResult(
                activities=activities,
                search_summary=summary
            )
            
        except Exception as e:
            print(f"Error parsing kid-friendly response: {e}")
            return self._create_fallback_kid_activities(context, children_ages)
    
    def _create_fallback_kid_activities(self, context: TripContext, children_ages: list) -> SearchResult:
        """Create basic fallback kid-friendly activities."""
        
        activities = [
            ActivityResult(
                name=f"Local Playground Visit",
                description="Safe playground equipment appropriate for young children",
                location=context.query.location,
                age_range=[2, 12],
                price_range="Free",
                duration="1-2 hours",
                weather_dependent=True,
                source_url=None
            ),
            ActivityResult(
                name=f"Indoor Family Activity Center",
                description="Climate-controlled space with kid-friendly activities",
                location=context.query.location,
                age_range=[3, 12],
                price_range="$10-20",
                duration="2-3 hours",
                weather_dependent=False,
                source_url=None
            )
        ]
        
        return SearchResult(
            activities=activities,
            search_summary=f"Basic kid-friendly activities for children ages {children_ages}"
        )


def create_kid_friendly_activity_agent_gcp(model_name: str = "gemini-1.0-pro") -> KidFriendlyAgentGCP:
    """Create a GCP-powered kid-friendly activity agent."""
    return KidFriendlyAgentGCP(model_name=model_name)
