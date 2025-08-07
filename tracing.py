"""
Tracing and logging system for GCP multi-agent interactions.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AgentEventType(Enum):
    """Types of agent events to track."""
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete" 
    AGENT_ERROR = "agent_error"
    TOOL_CALL = "tool_call"
    HANDOFF = "handoff"
    RESPONSE_GENERATED = "response_generated"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"

@dataclass
class AgentEvent:
    """Represents a single event in the agent workflow."""
    timestamp: datetime
    event_type: AgentEventType
    agent_name: str
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None

class AgentTracer:
    """Tracks and logs agent interactions."""
    
    def __init__(self):
        self.events: List[AgentEvent] = []
        self.start_times: Dict[str, float] = {}
        self.enabled = True
    
    def start_workflow(self, workflow_name: str, details: Dict[str, Any] = None):
        """Start tracing a workflow."""
        if not self.enabled:
            return
            
        self.events.clear()  # Clear previous events
        self.start_times[workflow_name] = time.time()
        
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.WORKFLOW_START,
            agent_name="Workflow Manager",
            message=f"🚀 Starting workflow: {workflow_name}",
            details=details or {}
        )
        self.events.append(event)
        self._log_event(event)
    
    def start_agent(self, agent_name: str, task: str, context: Dict[str, Any] = None):
        """Start tracing an agent's work."""
        if not self.enabled:
            return
            
        agent_key = f"{agent_name}_{len(self.events)}"
        self.start_times[agent_key] = time.time()
        
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.AGENT_START,
            agent_name=agent_name,
            message=f"🤖 {agent_name} starting: {task}",
            details=context or {}
        )
        self.events.append(event)
        self._log_event(event)
        
        return agent_key
    
    def complete_agent(self, agent_key: str, agent_name: str, result: str, details: Dict[str, Any] = None):
        """Complete tracing an agent's work."""
        if not self.enabled:
            return
            
        duration = None
        if agent_key in self.start_times:
            duration = (time.time() - self.start_times[agent_key]) * 1000  # Convert to ms
            del self.start_times[agent_key]
        
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.AGENT_COMPLETE,
            agent_name=agent_name,
            message=f"✅ {agent_name} completed: {result}",
            details=details or {},
            duration_ms=duration
        )
        self.events.append(event)
        self._log_event(event)
    
    def log_tool_call(self, agent_name: str, tool_name: str, parameters: Dict[str, Any], result: Any = None):
        """Log a tool call by an agent."""
        if not self.enabled:
            return
            
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.TOOL_CALL,
            agent_name=agent_name,
            message=f"🔧 {agent_name} called tool: {tool_name}",
            details={
                "tool_name": tool_name,
                "parameters": parameters,
                "result_preview": str(result)[:100] + "..." if result and len(str(result)) > 100 else str(result)
            }
        )
        self.events.append(event)
        self._log_event(event)
    
    def log_handoff(self, from_agent: str, to_agent: str, reason: str, context: Dict[str, Any] = None):
        """Log a handoff between agents."""
        if not self.enabled:
            return
            
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.HANDOFF,
            agent_name=from_agent,
            message=f"🔄 {from_agent} → {to_agent}: {reason}",
            details={"to_agent": to_agent, "reason": reason, "context": context or {}}
        )
        self.events.append(event)
        self._log_event(event)
    
    def log_response(self, agent_name: str, response_preview: str, full_response_length: int):
        """Log an agent's response generation."""
        if not self.enabled:
            return
            
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.RESPONSE_GENERATED,
            agent_name=agent_name,
            message=f"💬 {agent_name} generated response ({full_response_length} chars)",
            details={
                "response_preview": response_preview[:200] + "..." if len(response_preview) > 200 else response_preview,
                "full_length": full_response_length
            }
        )
        self.events.append(event)
        self._log_event(event)
    
    def log_error(self, agent_name: str, error: str, details: Dict[str, Any] = None):
        """Log an agent error."""
        if not self.enabled:
            return
            
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.AGENT_ERROR,
            agent_name=agent_name,
            message=f"❌ {agent_name} error: {error}",
            details=details or {}
        )
        self.events.append(event)
        self._log_event(event)
    
    def complete_workflow(self, workflow_name: str, success: bool = True):
        """Complete workflow tracing."""
        if not self.enabled:
            return
            
        duration = None
        if workflow_name in self.start_times:
            duration = (time.time() - self.start_times[workflow_name]) * 1000
            del self.start_times[workflow_name]
        
        status = "✅ successfully" if success else "❌ with errors"
        event = AgentEvent(
            timestamp=datetime.now(),
            event_type=AgentEventType.WORKFLOW_COMPLETE,
            agent_name="Workflow Manager",
            message=f"🏁 Workflow {workflow_name} completed {status}",
            details={"success": success, "total_events": len(self.events)},
            duration_ms=duration
        )
        self.events.append(event)
        self._log_event(event)
    
    def _log_event(self, event: AgentEvent):
        """Log event to console."""
        timestamp_str = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        duration_str = f" ({event.duration_ms:.1f}ms)" if event.duration_ms else ""
        
        print(f"[{timestamp_str}] {event.message}{duration_str}")
        
        # Log additional details if present
        if event.details and any(event.details.values()):
            relevant_details = {k: v for k, v in event.details.items() if v is not None}
            if relevant_details:
                print(f"           Details: {relevant_details}")
    
    def get_summary(self) -> str:
        """Get a summary of the agent interactions."""
        if not self.events:
            return "No agent interactions recorded."
        
        summary_lines = ["🔍 Agent Interaction Summary", "=" * 50]
        
        # Count events by type
        event_counts = {}
        for event in self.events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
        
        summary_lines.append(f"📊 Total Events: {len(self.events)}")
        for event_type, count in event_counts.items():
            summary_lines.append(f"   • {event_type.replace('_', ' ').title()}: {count}")
        
        # Agent activity summary
        agent_activity = {}
        for event in self.events:
            if event.agent_name not in agent_activity:
                agent_activity[event.agent_name] = {"events": 0, "tools": 0, "errors": 0}
            agent_activity[event.agent_name]["events"] += 1
            if event.event_type == AgentEventType.TOOL_CALL:
                agent_activity[event.agent_name]["tools"] += 1
            elif event.event_type == AgentEventType.AGENT_ERROR:
                agent_activity[event.agent_name]["errors"] += 1
        
        summary_lines.append(f"\n🤖 Agent Activity:")
        for agent, stats in agent_activity.items():
            summary_lines.append(f"   • {agent}: {stats['events']} events, {stats['tools']} tools, {stats['errors']} errors")
        
        # Timeline of key events
        summary_lines.append(f"\n⏱️  Key Events Timeline:")
        for event in self.events:
            if event.event_type in [AgentEventType.AGENT_START, AgentEventType.AGENT_COMPLETE, AgentEventType.HANDOFF]:
                time_str = event.timestamp.strftime("%H:%M:%S")
                summary_lines.append(f"   {time_str} - {event.message}")
        
        return "\n".join(summary_lines)
    
    def get_events_for_streamlit(self) -> List[Dict[str, Any]]:
        """Get events formatted for Streamlit display."""
        return [
            {
                "timestamp": event.timestamp.strftime("%H:%M:%S.%f")[:-3],
                "agent": event.agent_name,
                "event_type": event.event_type.value.replace("_", " ").title(),
                "message": event.message.replace("🚀", "").replace("🤖", "").replace("✅", "").replace("❌", "").replace("🔧", "").replace("🔄", "").replace("💬", "").replace("🏁", "").strip(),
                "duration_ms": event.duration_ms,
                "details": event.details or {}
            }
            for event in self.events
        ]

# Global tracer instance
tracer = AgentTracer()

def enable_tracing():
    """Enable agent tracing."""
    tracer.enabled = True

def disable_tracing():
    """Disable agent tracing."""
    tracer.enabled = False

def get_tracer() -> AgentTracer:
    """Get the global tracer instance."""
    return tracer
