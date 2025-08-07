import streamlit as st
import asyncio
from datetime import date, datetime
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Import your multi-agent system
import agents as openai_agents
Runner = openai_agents.Runner
trace = openai_agents.trace
gen_trace_id = openai_agents.gen_trace_id
Agent = openai_agents.Agent
RunResult = openai_agents.RunResult
MCPServerStdio = openai_agents.mcp.MCPServerStdio
from models import TripQuery, TripContext
from my_agents import (
    create_weather_agent, WeatherAnalysis,
    create_activity_search_agent, SearchResult,
    create_recommendation_agent, TripPlan,
)

# Configure the Streamlit page
st.set_page_config(
    page_title="🌍 AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.trip-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}
.activity-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.weather-info {
    background: linear-gradient(135deg, #74b9ff, #0984e3);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class StreamlitAdventureManager:
    """Adventure manager optimized for Streamlit with better error handling"""

    def __init__(self):
        if 'activity_search_agent' not in st.session_state:
            st.session_state.activity_search_agent = create_activity_search_agent()
        if 'recommendation_agent' not in st.session_state:
            st.session_state.recommendation_agent = create_recommendation_agent()

    async def run(self, query: TripQuery, progress_bar=None, status_text=None, use_real_weather=False, sudo_password="") -> dict:
        """Run the adventure planning workflow with Streamlit progress updates"""
        
        results = {
            'weather_info': None,
            'search_results': None,
            'trip_plan': None,
            'trace_id': None,
            'error': None
        }
        
        try:
            trace_id = gen_trace_id()
            results['trace_id'] = trace_id
            
            if status_text:
                status_text.text("🌍 Creating trip context...")
            
            trip_context = TripContext(query=query)
            
            with trace("Adventure Planning (Streamlit)", trace_id=trace_id):
                # 1. Get Weather Information
                if status_text:
                    status_text.text("🌤️ Getting weather information...")
                if progress_bar:
                    progress_bar.progress(0.2)
                
                weather_info = await self._get_weather_info(trip_context, use_real_weather, sudo_password)
                results['weather_info'] = weather_info
                
                # 2. Search for Activities
                if status_text:
                    status_text.text("🔍 Searching for activities...")
                if progress_bar:
                    progress_bar.progress(0.5)
                
                search_results, search_agent_used = await self._search_for_activities(trip_context, weather_info)
                results['search_results'] = search_results
                
                # 3. Generate Trip Plan
                if status_text:
                    status_text.text("📋 Creating your personalized trip plan...")
                if progress_bar:
                    progress_bar.progress(0.8)
                
                trip_plan = await self._generate_trip_plan(search_results, weather_info, trip_context)
                results['trip_plan'] = trip_plan
                
                if progress_bar:
                    progress_bar.progress(1.0)
                if status_text:
                    status_text.text("✅ Trip plan complete!")
                    
        except Exception as e:
            results['error'] = str(e)
            if status_text:
                status_text.text(f"❌ Error: {str(e)}")
        
        return results

    async def _get_weather_info(self, context: TripContext, use_real_weather=False, sudo_password="") -> WeatherAnalysis:
        """Get weather information with fallback to mock data"""
        if not use_real_weather:
            # Use mock weather data by default (faster and more reliable)
            return WeatherAnalysis(
                summary=f"Expected pleasant weather for {context.query.location} during your trip dates. Weather forecast will be updated closer to your departure date.",
                temperature_range=[18.0, 28.0],
                precipitation_chance=0.25,
                recommended_clothing=[
                    "Comfortable walking shoes",
                    "Light jacket for evenings", 
                    "Umbrella or light rain jacket",
                    "Layered clothing for temperature changes"
                ],
                weather_warnings=["Check local weather forecast 24-48 hours before departure"]
            )
        
        try:
            # Try MCP weather server first
            if sudo_password:
                # Use sudo with password
                weather_mcp_server = MCPServerStdio(
                    params={
                        "command": "sudo",
                        "args": ["docker", "run", "--rm", "-i", "mcp_server_weather"],
                    }
                )
            else:
                # Try without sudo first
                weather_mcp_server = MCPServerStdio(
                    params={
                        "command": "docker",
                        "args": ["run", "--rm", "-i", "mcp_server_weather"],
                    }
                )
            
            async with weather_mcp_server as server:
                weather_agent = create_weather_agent(mcp_servers=[server])
                input_str = (
                    f"Get weather analysis for a trip to {context.query.location} "
                    f"from {context.query.start_date} to {context.query.end_date}."
                )
                result = await Runner.run(weather_agent, input_str, context=context)
                return result.final_output_as(WeatherAnalysis)
                
        except Exception:
            # Fallback to mock weather data
            return WeatherAnalysis(
                summary=f"Expected pleasant weather for {context.query.location} during your trip dates. Weather forecast will be updated closer to your departure date.",
                temperature_range=[18.0, 28.0],
                precipitation_chance=0.25,
                recommended_clothing=[
                    "Comfortable walking shoes",
                    "Light jacket for evenings", 
                    "Umbrella or light rain jacket",
                    "Layered clothing for temperature changes"
                ],
                weather_warnings=["Check local weather forecast 24-48 hours before departure"]
            )

    async def _search_for_activities(self, context: TripContext, weather_info: WeatherAnalysis):
        """Search for activities"""
        participants_str = f"{context.query.participant_number} participants (ages: {context.query.participant_ages})"
        input_str = (
            f"Find activities for a trip to {context.query.location} "
            f"from {context.query.start_date} to {context.query.end_date} "
            f"for {participants_str}.\n\n"
            f"Consider the following weather summary:\n{weather_info.summary}"
        )

        result = await Runner.run(st.session_state.activity_search_agent, input_str, context=context)
        return result.final_output_as(SearchResult), result.last_agent

    async def _generate_trip_plan(self, search_results: SearchResult, weather_info: WeatherAnalysis, context: TripContext):
        """Generate the final trip plan"""
        participants_str = f"{context.query.participant_number} participants (ages: {context.query.participant_ages})"
        dates_str = f"{context.query.start_date} to {context.query.end_date}"
        input_str = (
            f"Create a trip plan for {context.query.location} from {dates_str} "
            f"for {participants_str}.\n\n"
            f"Weather Information:\n{weather_info.model_dump()}\n\n"
            f"Potential Activities Found:\n{search_results.model_dump()}"
        )

        result = await Runner.run(st.session_state.recommendation_agent, input_str, context=context)
        return result.final_output_as(TripPlan)


def display_weather_info(weather_info):
    """Display weather information in a nice format"""
    st.markdown('<div class="weather-info">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌡️ Temperature Range", 
                 f"{weather_info.temperature_range[0]}°C - {weather_info.temperature_range[1]}°C")
    
    with col2:
        st.metric("🌧️ Precipitation Chance", 
                 f"{int(weather_info.precipitation_chance * 100)}%")
    
    with col3:
        st.metric("👔 Clothing Items", 
                 f"{len(weather_info.recommended_clothing)}")
    
    st.markdown("**Weather Summary:**")
    st.write(weather_info.summary)
    
    if weather_info.weather_warnings:
        st.warning("⚠️ " + " | ".join(weather_info.weather_warnings))
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_activities(trip_plan):
    """Display recommended activities"""
    if hasattr(trip_plan, 'recommended_activities') and trip_plan.recommended_activities:
        for i, activity in enumerate(trip_plan.recommended_activities, 1):
            st.markdown(f'<div class="activity-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{i}. {activity.name}")
                st.write(activity.description)
                st.write(f"**Why we recommend this:** {activity.reasoning}")
                
                if hasattr(activity, 'best_time') and activity.best_time:
                    st.info(f"⏰ **Best Time:** {activity.best_time}")
            
            with col2:
                if hasattr(activity, 'source_url') and activity.source_url:
                    st.link_button("🔗 More Info", activity.source_url)
            
            if hasattr(activity, 'weather_considerations') and activity.weather_considerations:
                with st.expander("🌤️ Weather Considerations"):
                    for consideration in activity.weather_considerations:
                        st.write(f"• {consideration}")
            
            if hasattr(activity, 'preparation_tips') and activity.preparation_tips:
                with st.expander("💡 Preparation Tips"):
                    for tip in activity.preparation_tips:
                        st.write(f"• {tip}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No specific activities were recommended. Try adjusting your search criteria.")


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 AI Travel Planner</h1>', unsafe_allow_html=True)
    st.markdown("### Plan your perfect trip with AI-powered recommendations!")
    
    # Sidebar for trip inputs
    with st.sidebar:
        st.header("✈️ Trip Details")
        
        # Location input
        location = st.text_input(
            "🗺️ Destination", 
            value="New York City",
            help="Enter your travel destination"
        )
        
        # Date inputs
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "📅 Start Date",
                value=date(2025, 8, 10),
                min_value=date.today()
            )
        
        with col2:
            end_date = st.date_input(
                "📅 End Date",
                value=date(2025, 8, 14),
                min_value=start_date
            )
        
        # Traveler information
        st.subheader("👥 Travelers")
        participant_number = st.number_input(
            "Number of Travelers",
            min_value=1,
            max_value=10,
            value=2
        )
        
        # Ages input
        ages_input = st.text_input(
            "Ages (comma-separated)",
            value="5, 35",
            help="Enter ages separated by commas (e.g., 25, 30, 8)"
        )
        
        # Parse ages
        try:
            participant_ages = [int(age.strip()) for age in ages_input.split(',') if age.strip()]
        except ValueError:
            st.error("Please enter valid ages separated by commas")
            participant_ages = [25, 30]
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            use_real_weather = st.checkbox(
                "🌤️ Use Real Weather Data",
                value=False,
                help="Enable real weather data (requires Docker/sudo access)"
            )
            
            sudo_password = ""
            if use_real_weather:
                sudo_password = st.text_input(
                    "🔐 Sudo Password",
                    type="password",
                    help="Required for real weather data (Docker access)"
                )
                st.warning("⚠️ Password is only used for Docker access and not stored")
        
        # Plan Trip Button
        plan_button = st.button("🚀 Plan My Trip!", type="primary", use_container_width=True)
    
    # Main content area
    if plan_button:
        if not location.strip():
            st.error("Please enter a destination!")
            return
        
        if len(participant_ages) != participant_number:
            st.warning(f"Number of ages ({len(participant_ages)}) doesn't match number of travelers ({participant_number}). Using provided ages.")
        
        # Create trip query
        query = TripQuery(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            location=location,
            participant_number=participant_number,
            participant_ages=participant_ages
        )
        
        # Show trip summary
        st.markdown('<div class="trip-card">', unsafe_allow_html=True)
        st.subheader("🎯 Trip Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📍 Destination", location)
        with col2:
            st.metric("📅 Duration", f"{(end_date - start_date).days + 1} days")
        with col3:
            st.metric("👥 Travelers", participant_number)
        with col4:
            st.metric("👶 Ages", f"{min(participant_ages)}-{max(participant_ages)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run the AI planning process
        manager = StreamlitAdventureManager()
        
        try:
            # Run the async planning process
            results = asyncio.run(manager.run(query, progress_bar, status_text, use_real_weather, sudo_password))
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if results['error']:
                st.error(f"An error occurred: {results['error']}")
                return
            
            # Display results
            st.success("🎉 Your trip plan is ready!")
            
            # Weather Information
            if results['weather_info']:
                st.header("🌤️ Weather Information")
                display_weather_info(results['weather_info'])
            
            # Trip Plan
            if results['trip_plan']:
                trip_plan = results['trip_plan']
                
                # Activities
                st.header("🎯 Recommended Activities")
                display_activities(trip_plan)
                
                # Packing List
                if hasattr(trip_plan, 'packing_list') and trip_plan.packing_list:
                    st.header("🎒 Packing List")
                    packing_df = pd.DataFrame({'Items to Pack': trip_plan.packing_list})
                    st.dataframe(packing_df, use_container_width=True)
                
                # General Tips
                if hasattr(trip_plan, 'general_tips') and trip_plan.general_tips:
                    st.header("💡 Travel Tips")
                    for tip in trip_plan.general_tips:
                        st.info(f"💡 {tip}")
                
                # Trace information
                if results['trace_id']:
                    with st.expander("🔍 AI Processing Details"):
                        st.code(f"Trace ID: {results['trace_id']}")
                        st.markdown(f"[View detailed trace](https://platform.openai.com/traces/trace?trace_id={results['trace_id']})")
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"An error occurred while planning your trip: {str(e)}")
    
    else:
        # Welcome message when no trip is planned yet
        st.markdown("""
        ## 🎯 How it works:
        
        1. **📍 Enter your destination** in the sidebar
        2. **📅 Choose your travel dates**
        3. **👥 Add traveler information** (ages help us find age-appropriate activities)
        4. **🚀 Click "Plan My Trip!"** to get AI-powered recommendations
        
        Our AI agents will:
        - 🌤️ **Analyze weather** for your travel dates
        - 🔍 **Search for activities** tailored to your group
        - 📋 **Create personalized recommendations** with detailed planning
        - 🎒 **Generate packing lists** and travel tips
        
        ### ✨ Features:
        - **Smart activity matching** based on traveler ages
        - **Weather-aware recommendations**
        - **Family-friendly options** for trips with children
        - **Detailed activity information** with links and tips
        """)


if __name__ == "__main__":
    main()
