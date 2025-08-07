"""
Streamlit web interface for the GCP-powered multi-agent travel planner.
Uses Google Gemini models instead of OpenAI.
"""

import streamlit as st
import asyncio
import os
import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv

from manager_gcp import AdventureManagerGCP
from models_gcp import TripQuery, TripPlan
from tracing_adk import get_enhanced_tracer

# Load environment variables
load_dotenv()


class StreamlitAdventureManagerGCP:
    """Streamlit wrapper for the GCP Adventure Manager."""
    
    def __init__(self):
        """Initialize the Streamlit GCP Adventure Manager."""
        self.manager = None
        self._initialize_manager()
    
    def _initialize_manager(self):
        """Initialize the GCP manager with API credentials."""
        try:
            api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            
            self.manager = AdventureManagerGCP(api_key=api_key, project_id=project_id)
            
            if not api_key:
                st.warning("⚠️ No Google AI API key found. Using mock data for demonstration.")
                st.info("To use real Gemini AI, set GOOGLE_AI_API_KEY in your .env file")
            else:
                st.success("✅ Connected to Google Gemini AI")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize GCP manager: {e}")
            self.manager = None
    
    async def run_planning(self, query: TripQuery, use_real_weather: bool = False) -> TripPlan:
        """Run the planning workflow asynchronously."""
        if not self.manager:
            raise Exception("GCP Manager not initialized")
        
        return await self.manager.run(query, use_real_weather=use_real_weather)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="GCP Adventure Planner",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .activity-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .weather-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .packing-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🗺️ GCP Adventure Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by Google Gemini AI & Cloud Platform</div>', unsafe_allow_html=True)
    
    # Initialize the app manager
    app_manager = StreamlitAdventureManagerGCP()
    
    # Sidebar for input
    with st.sidebar:
        st.header("✈️ Plan Your Adventure")
        
        # Trip details form
        with st.form("trip_form"):
            st.subheader("Trip Details")
            
            # Destination
            destination = st.text_input(
                "📍 Destination",
                placeholder="e.g., San Francisco, CA",
                help="Enter the city or location you want to visit"
            )
            
            # Dates
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "📅 Start Date",
                    value=date.today(),
                    help="Trip start date"
                )
            with col2:
                end_date = st.date_input(
                    "📅 End Date",
                    value=date.today(),
                    help="Trip end date"
                )
            
            # Participants
            participant_number = st.number_input(
                "👥 Number of Participants",
                min_value=1,
                max_value=20,
                value=2,
                help="Total number of people going on the trip"
            )
            
            # Ages
            ages_input = st.text_input(
                "🎂 Participant Ages",
                placeholder="e.g., 8, 10, 35, 37",
                help="Enter ages separated by commas"
            )
            
            # Options
            st.subheader("Options")
            use_real_weather = st.checkbox(
                "🌤️ Use Real Weather Data",
                value=False,
                help="Attempt to get real weather data (requires additional setup)"
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Plan My Adventure", use_container_width=True)
        
        # API Key status
        st.markdown("---")
        st.subheader("🔧 Configuration")
        
        api_key = os.getenv('GOOGLE_AI_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key:
            st.success("✅ Google AI API key configured")
        else:
            st.warning("⚠️ No API key found")
            st.info("Set GOOGLE_AI_API_KEY in your .env file for full functionality")
    
    # Main content area
    if submitted:
        # Validate inputs
        if not destination:
            st.error("❌ Please enter a destination")
            return
        
        if start_date >= end_date:
            st.error("❌ End date must be after start date")
            return
        
        if not ages_input:
            st.error("❌ Please enter participant ages")
            return
        
        try:
            # Parse ages
            ages = [int(age.strip()) for age in ages_input.split(",")]
            
            if len(ages) != participant_number:
                st.error(f"❌ Number of ages ({len(ages)}) doesn't match number of participants ({participant_number})")
                return
            
            # Create trip query
            query = TripQuery(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                location=destination,
                participant_number=participant_number,
                participant_ages=ages
            )
            
            # Show planning in progress
            with st.spinner("🤖 AI agents are planning your adventure..."):
                # Run the planning workflow
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    trip_plan = loop.run_until_complete(
                        app_manager.run_planning(query, use_real_weather)
                    )
                    
                    # Display results
                    display_trip_plan(trip_plan)
                    
                    # Display agent interaction trace
                    display_agent_trace()
                    
                    st.success("🎉 Your adventure plan is ready!")
                    
                except Exception as e:
                    st.error(f"❌ Planning failed: {e}")
                    st.info("This might be due to API configuration issues. Check your Google AI API key.")
                    
                    # Show a basic fallback
                    st.warning("📋 **Fallback Plan Generated**")
                    st.write(f"**Destination:** {query.location}")
                    st.write(f"**Dates:** {query.start_date} to {query.end_date}")
                    st.write(f"**Participants:** {query.participant_number} people")
                    st.info("*Add your Google AI API key for detailed planning with real AI-powered recommendations*")
                
                finally:
                    loop.close()
        
        except ValueError:
            st.error("❌ Please enter valid ages (numbers separated by commas)")
    
    else:
        # Show welcome message and features
        display_welcome_page()


def display_welcome_page():
    """Display the welcome page with features and instructions."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 Welcome to Your AI-Powered Travel Assistant!
        
        """)


def display_trip_plan(plan: TripPlan):
    """Display the generated trip plan in a beautiful format."""
    
    st.success("✅ Your adventure plan is ready!")
    
    # Trip overview
    st.markdown("## 🗺️ Trip Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📍 Destination", plan.location)
    with col2:
        st.metric("📅 Duration", plan.dates)
    with col3:
        st.metric("👥 Group", plan.participants_summary)
    
    # Weather information
    st.markdown("## 🌤️ Weather Overview")
    
    # Display weather in a nice format
    with st.container():
        # Clean up the weather summary for display
        weather_text = plan.weather_summary
        
        # Remove any code-like content
        if "Function" in weather_text or "result:" in weather_text:
            # Extract only the meaningful weather info
            lines = weather_text.split('\n')
            clean_lines = []
            for line in lines:
                if line.strip() and not any(skip in line.lower() for skip in ['function', 'result:', 'get_current', 'get_forecast']):
                    clean_lines.append(line.strip())
            
            if clean_lines:
                weather_text = ' '.join(clean_lines[:3])  # Take first 3 meaningful lines
            else:
                weather_text = f"Weather information for {plan.location} during your visit."
        
        # Display clean weather summary
        st.info(weather_text)
        
        # Try to extract weather metrics from the plan's weather data
        # Note: This would work better if we passed the WeatherAnalysis object directly
        # For now, we'll show smart defaults
        
        temp_display = "18°C - 26°C"
        rain_display = "25%"
        clothing_display = "Light layers"
        
        # Try to extract temperature info from weather text  
        import re
        if weather_text:
            # More specific regex to match actual temperatures (number followed directly by ° or degree)
            temp_matches = re.findall(r'(\d+)\s*(?:°C|°|degree)', weather_text)
            if len(temp_matches) >= 2:
                try:
                    temps = [int(t) for t in temp_matches]
                    # Filter out unrealistic temperatures (dates, years, etc.)
                    realistic_temps = [t for t in temps if -30 <= t <= 60]
                    if len(realistic_temps) >= 2:
                        temp_display = f"{min(realistic_temps)}°C - {max(realistic_temps)}°C"
                except:
                    pass
            
            # Check for rain mentions
            if any(word in weather_text.lower() for word in ['rain', 'shower', 'precipitation']):
                if 'high' in weather_text.lower() or 'likely' in weather_text.lower():
                    rain_display = "60-80%"
                elif 'low' in weather_text.lower():
                    rain_display = "10-20%"
                else:
                    rain_display = "40-60%"
            
            # Determine clothing style
            avg_temp_from_text = 22  # default
            try:
                if temp_matches:
                    temps = [int(t) for t in temp_matches[:2]]
                    avg_temp_from_text = sum(temps) / len(temps)
            except:
                pass
            
            if avg_temp_from_text < 15:
                clothing_display = "Warm layers"
            elif avg_temp_from_text > 25:
                clothing_display = "Light clothing"
            else:
                clothing_display = "Light layers"
        
        # Display weather metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperature", temp_display, help="Expected temperature range")
        with col2:
            st.metric("☔ Rain Chance", rain_display, help="Probability of precipitation")
        with col3:
            st.metric("👕 Clothing", clothing_display, help="Recommended clothing style")
    
    # Recommended activities
    st.markdown("## 🎯 Recommended Activities")
    
    if not plan.recommended_activities:
        st.info("No specific activities recommended based on search criteria.")
    else:
        for i, activity in enumerate(plan.recommended_activities, 1):
            with st.container():
                st.subheader(f"{i}. {activity.name}")
                st.write(f"**Description:** {activity.description}")
                st.write(f"**Why recommended:** {activity.reasoning}")
                if activity.best_time:
                    st.write(f"**Best time:** {activity.best_time}")
                
                if activity.weather_considerations:
                    st.write("**Weather Considerations:**")
                    for consideration in activity.weather_considerations:
                        st.write(f"• {consideration}")
                
                if activity.preparation_tips:
                    st.write("**Preparation Tips:**")
                    for tip in activity.preparation_tips:
                        st.write(f"• {tip}")
                
                st.markdown("---")
    
    # Packing list and tips
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🎒 Packing Checklist")
        for item in plan.packing_list:
            st.write(f"☐ {item}")
    
    with col2:
        st.markdown("## 💡 Pro Tips")
        for tip in plan.general_tips:
            st.markdown(f"• {tip}")
    
    # Download/export options
    st.markdown("---")
    st.markdown("### 📥 Export Your Plan")
    
    # Create a downloadable text version
    plan_text = f"""
ADVENTURE PLAN: {plan.location}
{plan.dates}
{plan.participants_summary}

WEATHER:
{plan.weather_summary}

RECOMMENDED ACTIVITIES:
"""
    
    for i, activity in enumerate(plan.recommended_activities, 1):
        plan_text += f"""
{i}. {activity.name}
   Description: {activity.description}
   Reasoning: {activity.reasoning}
   Best time: {activity.best_time or 'Flexible'}
"""
    
    plan_text += f"""

PACKING LIST:
{chr(10).join([f"- {item}" for item in plan.packing_list])}

GENERAL TIPS:
{chr(10).join([f"- {tip}" for tip in plan.general_tips])}
"""
    
    st.download_button(
        label="📄 Download as Text File",
        data=plan_text,
        file_name=f"adventure_plan_{plan.location.replace(' ', '_').replace(',', '')}.txt",
        mime="text/plain"
    )


def display_agent_trace():
    """Display the agent interaction trace in Streamlit."""
    tracer = get_enhanced_tracer()
    events = tracer.get_events_for_streamlit()
    
    if not events:
        return
    
    st.markdown("---")
    st.markdown("## 🔍 Agent Interaction Trace")
    
    # Display ADK tracing status
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Tracing Status")
    with col2:
        # Check if ADK tracing is available
        adk_status = "✅ Active" if hasattr(tracer, 'adk_instrumented') and tracer.adk_instrumented else "❌ Disabled"
        st.metric("Google ADK Tracing", adk_status)
    
    if hasattr(tracer, 'adk_instrumented') and tracer.adk_instrumented:
        st.info("🔍 **Enhanced Tracing Active**: Both custom event tracking and Google ADK OpenTelemetry spans are being captured!")
    else:
        st.warning("⚠️ **Basic Tracing Only**: Google ADK tracing is disabled. Install dependencies for full tracing.")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Event Timeline", "📊 Agent Summary", "🔧 Raw Details", "🔍 ADK Info"])
    
    with tab1:
        st.markdown("### Agent Workflow Timeline")
        
        # Create a timeline visualization
        for i, event in enumerate(events):
            # Color coding by event type
            if "start" in event["event_type"].lower():
                icon = "🚀"
                color = "blue"
            elif "complete" in event["event_type"].lower():
                icon = "✅"
                color = "green"
            elif "error" in event["event_type"].lower():
                icon = "❌"
                color = "red"
            elif "tool" in event["event_type"].lower():
                icon = "🔧"
                color = "orange"
            elif "handoff" in event["event_type"].lower():
                icon = "🔄"
                color = "purple"
            else:
                icon = "📝"
                color = "gray"
            
            duration_text = f" ({event['duration_ms']:.1f}ms)" if event['duration_ms'] else ""
            
            # Display event
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 8])
                with col1:
                    st.write(f"`{event['timestamp']}`")
                with col2:
                    st.write(f"**{event['agent']}**")
                with col3:
                    st.write(f"{icon} {event['message']}{duration_text}")
    
    with tab2:
        st.markdown("### Agent Activity Summary")
        
        # Aggregate agent statistics
        agent_stats = {}
        for event in events:
            agent = event['agent']
            if agent not in agent_stats:
                agent_stats[agent] = {
                    'total_events': 0,
                    'tool_calls': 0,
                    'errors': 0,
                    'total_duration': 0,
                    'start_time': None,
                    'end_time': None
                }
            
            agent_stats[agent]['total_events'] += 1
            
            if 'tool' in event['event_type'].lower():
                agent_stats[agent]['tool_calls'] += 1
            elif 'error' in event['event_type'].lower():
                agent_stats[agent]['errors'] += 1
            
            if event['duration_ms']:
                agent_stats[agent]['total_duration'] += event['duration_ms']
            
            # Track start/end times
            if agent_stats[agent]['start_time'] is None:
                agent_stats[agent]['start_time'] = event['timestamp']
            agent_stats[agent]['end_time'] = event['timestamp']
        
        # Display as a table
        if agent_stats:
            df_data = []
            for agent, stats in agent_stats.items():
                df_data.append({
                    'Agent': agent,
                    'Events': stats['total_events'],
                    'Tool Calls': stats['tool_calls'],
                    'Errors': stats['errors'],
                    'Total Duration (ms)': f"{stats['total_duration']:.1f}" if stats['total_duration'] > 0 else "N/A"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.markdown("### Raw Event Details")
        
        # Show detailed event information
        for i, event in enumerate(events):
            with st.expander(f"Event {i+1}: {event['agent']} - {event['event_type']}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Basic Info:**")
                    st.write(f"- Timestamp: `{event['timestamp']}`")
                    st.write(f"- Agent: `{event['agent']}`")
                    st.write(f"- Type: `{event['event_type']}`")
                    if event['duration_ms']:
                        st.write(f"- Duration: `{event['duration_ms']:.1f}ms`")
                
                with col2:
                    st.write("**Message:**")
                    st.write(event['message'])
                    
                    if event['details']:
                        st.write("**Details:**")
                        st.json(event['details'])
    
    with tab4:
        st.markdown("### Google ADK Tracing Information")
        
        # Display ADK tracing capabilities and status
        if hasattr(tracer, 'adk_instrumented') and tracer.adk_instrumented:
            st.success("🎉 **Google ADK Tracing is Active!**")
            
            st.markdown("""
            **What's being traced:**
            - 🤖 Gemini model interactions via OpenTelemetry spans  
            - 🔧 Function/tool calls with parameters and results
            - ⏱️ Response timing and token usage (when available)
            - 🔄 Agent handoffs and workflow orchestration
            - 📝 Custom event tracking (shown in other tabs)
            
            **Benefits:**
            - **Performance Monitoring**: See exactly how long each model call takes
            - **Debugging**: Trace errors back to specific agent interactions  
            - **Optimization**: Identify bottlenecks in your agent workflow
            - **Compliance**: Full audit trail of AI model usage
            """)
            
            st.info("💡 **Pro Tip**: Check your terminal for detailed OpenTelemetry span output!")
            
            # Show environment info
            with st.expander("🔧 Tracing Configuration"):
                st.code(f"""
Project Name: {os.getenv('TRACEAI_PROJECT_NAME', 'GCP-Travel-Agent')}
OpenTelemetry Log Level: {os.getenv('OTEL_LOG_LEVEL', 'INFO')}
Tracer Provider: {type(tracer.tracer_provider).__name__ if hasattr(tracer, 'tracer_provider') and tracer.tracer_provider else 'Not configured'}
                """, language="yaml")
            
        else:
            st.error("❌ **Google ADK Tracing is Disabled**")
            
            st.markdown("""
            **To enable Google ADK tracing:**
            
            1. **Install dependencies** (if not already done):
            ```bash
            pip install traceai-google-adk opentelemetry-sdk
            ```
            
            2. **Restart the application**:
            ```bash
            streamlit run streamlit_app_gcp.py
            ```
            
            **What you'll get with ADK tracing:**
            - 🔍 Native OpenTelemetry spans for all Gemini interactions
            - 📊 Detailed performance metrics and timing
            - 🐛 Enhanced debugging capabilities  
            - 📈 Production-ready observability
            """)
            
            if st.button("🔄 Retry ADK Tracing Setup"):
                st.rerun()
    
    # Add a summary at the bottom
    st.markdown("### 📈 Workflow Summary")
    
    total_events = len(events)
    total_agents = len(set(event['agent'] for event in events))
    total_errors = sum(1 for event in events if 'error' in event['event_type'].lower())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", total_events)
    with col2:
        st.metric("Agents Used", total_agents)
    with col3:
        st.metric("Errors", total_errors, delta=None if total_errors == 0 else f"{total_errors} errors")
    with col4:
        workflow_duration = None
        if events:
            try:
                start_time = datetime.strptime(events[0]['timestamp'], "%H:%M:%S.%f")
                end_time = datetime.strptime(events[-1]['timestamp'], "%H:%M:%S.%f")
                workflow_duration = (end_time - start_time).total_seconds() * 1000
                st.metric("Total Duration", f"{workflow_duration:.1f}ms")
            except:
                st.metric("Total Duration", "N/A")


if __name__ == "__main__":
    main()