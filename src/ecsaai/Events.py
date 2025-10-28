# ===========================================
# Event Definitions
# ===========================================

from .main import BaseEvent

# Basic event classes for the framework
class InputEvent(BaseEvent):
    """Input event for user messages"""
    type: str = "input"
    content: str

class OutputEvent(BaseEvent):
    """Output event for responses"""
    type: str = "output"
    content: str
    format: str = "text"

class LLMRequest(BaseEvent):
    """Request to LLM component"""
    type: str = "llm_request"
    prompt: str
    model: str = "gpt-4"

class LLMResponse(BaseEvent):
    """Response from LLM component"""
    type: str = "llm_response"
    response: str
    model: str

# Export all events
__all__ = [
    'InputEvent',
    'OutputEvent',
    'LLMRequest',
    'LLMResponse'
]
