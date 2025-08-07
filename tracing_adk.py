"""
Enhanced ADK Tracing System
Combines custom tracing with Google's official ADK instrumentation
"""

import datetime
import os
from typing import List, Dict, Any, Optional
from tracing import AgentTracer as CustomTracer, get_tracer as get_custom_tracer

try:
    from traceai_google_adk import GoogleADKInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    ADK_TRACING_AVAILABLE = True
except ImportError:
    print("⚠️  ADK tracing dependencies not available. Using custom tracing only.")
    ADK_TRACING_AVAILABLE = False


class EnhancedADKTracer:
    """Enhanced tracer that combines custom tracing with Google ADK instrumentation."""
    
    def __init__(self):
        self.custom_tracer = get_custom_tracer()
        self.adk_instrumented = False
        self.tracer_provider = None
        
        if ADK_TRACING_AVAILABLE:
            self._setup_adk_tracing()
    
    def _setup_adk_tracing(self):
        """Set up Google ADK tracing with OpenTelemetry."""
        try:
            # Set up tracer provider with console exporter for local development
            tracer_provider = trace_sdk.TracerProvider()
            
            # Add console exporter to see traces in terminal
            console_exporter = ConsoleSpanExporter()
            span_processor = SimpleSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Optional: Add OTLP exporter if you have a trace collector
            # otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
            # otlp_processor = SimpleSpanProcessor(otlp_exporter)
            # tracer_provider.add_span_processor(otlp_processor)
            
            # Instrument Google ADK
            GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
            
            self.tracer_provider = tracer_provider
            self.adk_instrumented = True
            
            print("✅ Google ADK tracing instrumented successfully!")
            
        except Exception as e:
            print(f"⚠️  Failed to set up ADK tracing: {e}")
            print("   Continuing with custom tracing only.")
    
    # Delegate all custom tracing methods to the underlying custom tracer
    def start_workflow(self, workflow_name: str, details: Optional[Dict[str, Any]] = None):
        """Start workflow with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"🔍 [ADK Trace] Starting workflow: {workflow_name}")
        return self.custom_tracer.start_workflow(workflow_name, details)
    
    def complete_workflow(self, workflow_name: str, success: bool = True):
        """Complete workflow with both custom and ADK tracing."""
        if self.adk_instrumented:
            status = "successfully" if success else "with errors"
            print(f"🔍 [ADK Trace] Workflow {workflow_name} completed {status}")
        return self.custom_tracer.complete_workflow(workflow_name, success)
    
    def start_agent(self, agent_name: str, task: str, context: Dict[str, Any] = None):
        """Start agent with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"🤖 [ADK Trace] Agent {agent_name} starting: {task}")
        return self.custom_tracer.start_agent(agent_name, task, context)
    
    def complete_agent(self, agent_key: str, agent_name: str, result: str, details: Dict[str, Any] = None):
        """Complete agent with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"🤖 [ADK Trace] Agent {agent_name} completed: {result}")
        return self.custom_tracer.complete_agent(agent_key, agent_name, result, details)
    
    def log_tool_call(self, agent_name: str, tool_name: str, parameters: Dict[str, Any], result: Any = None):
        """Log tool call with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"🔧 [ADK Trace] {agent_name} called tool: {tool_name} with params: {parameters}")
        return self.custom_tracer.log_tool_call(agent_name, tool_name, parameters, result)
    
    def log_response(self, agent_name: str, response_preview: str, full_response_length: int):
        """Log response with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"💬 [ADK Trace] {agent_name} generated response ({full_response_length} chars)")
        return self.custom_tracer.log_response(agent_name, response_preview, full_response_length)
    
    def log_handoff(self, from_agent: str, to_agent: str, reason: str, context: Dict[str, Any] = None):
        """Log handoff with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"🔄 [ADK Trace] Handoff from {from_agent} to {to_agent}: {reason}")
        return self.custom_tracer.log_handoff(from_agent, to_agent, reason, context)
    
    def log_error(self, agent_name: str, error: str, details: Dict[str, Any] = None):
        """Log error with both custom and ADK tracing."""
        if self.adk_instrumented:
            print(f"❌ [ADK Trace] Error in {agent_name}: {error}")
        return self.custom_tracer.log_error(agent_name, error, details)
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get custom tracing events."""
        return self.custom_tracer.get_events()
    
    def get_events_for_streamlit(self) -> List[Dict[str, Any]]:
        """Get events formatted for Streamlit display."""
        return self.custom_tracer.get_events_for_streamlit()
    
    def get_summary(self) -> str:
        """Get tracing summary including ADK status."""
        base_summary = self.custom_tracer.get_summary() if hasattr(self.custom_tracer, 'get_summary') else "Custom tracing active"
        
        if self.adk_instrumented:
            adk_status = "\n\n🔍 Google ADK Tracing: ✅ ACTIVE\n   • OpenTelemetry spans exported to console\n   • Native Gemini model interactions traced"
        else:
            adk_status = "\n\n🔍 Google ADK Tracing: ❌ DISABLED\n   • Install dependencies: pip install traceai-google-adk opentelemetry-sdk"
        
        return base_summary + adk_status


# Global enhanced tracer instance
_enhanced_tracer = None

def get_enhanced_tracer() -> EnhancedADKTracer:
    """Get the global enhanced ADK tracer instance."""
    global _enhanced_tracer
    if _enhanced_tracer is None:
        _enhanced_tracer = EnhancedADKTracer()
    return _enhanced_tracer


def setup_adk_tracing_environment():
    """Set up environment variables for optimal ADK tracing."""
    # Set project name for ADK tracing
    if not os.getenv('TRACEAI_PROJECT_NAME'):
        os.environ['TRACEAI_PROJECT_NAME'] = 'GCP-Travel-Agent'
    
    # Enable detailed logging for development
    if not os.getenv('OTEL_LOG_LEVEL'):
        os.environ['OTEL_LOG_LEVEL'] = 'INFO'
    
    print("🔧 ADK tracing environment configured")


# Convenience function to maintain backward compatibility
def get_tracer() -> EnhancedADKTracer:
    """Backward compatible function that returns enhanced tracer."""
    return get_enhanced_tracer()
