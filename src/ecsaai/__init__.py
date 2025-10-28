"""
ECSA AI Framework - Modular AI Agents
"""

__version__ = "0.1.0"

# Main classes
from .main import (
    Agent,
    Component,
    AsyncComponent,
    ComponentConfig,
    ComponentMetrics,
    BaseEvent,
    BaseRequest,
    BaseResponse,
    ErrorEvent,
    ComponentMiddleware,
    ComponentTestHarness
)

# Events
try:
    from .Events import *
except ImportError:
    pass

# Components
try:
    from .Components import (
        LLMComponent, LLMConfig,
        ContextComponent, ContextConfig,
        OutputComponent, OutputConfig
    )
except ImportError:
    pass

# Libs
try:
    from .libs import *
except ImportError:
    pass

__all__ = [
    # Main classes
    "Agent", "Component", "AsyncComponent",
    "ComponentConfig", "ComponentMetrics",
    "BaseEvent", "BaseRequest", "BaseResponse", "ErrorEvent",
    "ComponentMiddleware", "ComponentTestHarness",
]
