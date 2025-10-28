# ===========================================
# ECSA Component Architecture
# ===========================================
"""
New component architecture with critical improvements:
- Lifecycle hooks (on_init, on_shutdown, on_reload)
- Robust error handling
- Type-safe event contracts
- Inter-component communication
- Configuration schemas
- Basic observability
- Middleware support
"""

import sys
import os
from abc import ABC, abstractmethod
from typing import List, ClassVar, Dict, Type, Optional, Any, Annotated
from pydantic import BaseModel, Field
import asyncio
from enum import Enum
import uuid
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================
# Base Schemas
# ===========================================

class ComponentConfig(BaseModel):
    """Base class for component configurations"""
    pass

class ComponentMetrics(BaseModel):
    """Automatic metrics per component"""
    events_received: int = 0
    events_emitted: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    last_error: Optional[str] = None

class BaseEvent(BaseModel):
    """Base event with metadata"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    target: str
    type: str
    timestamp: float = Field(default_factory=time.time)
    meta: Dict[str, Any] = Field(default_factory=dict)

class ErrorEvent(BaseEvent):
    """Error event"""
    type: str = "error"
    error: str
    original_event: Optional[BaseEvent] = None

# ===========================================
# Component Base Class
# ===========================================

class Component(ABC):
    """
    Component base class with lifecycle management
    """

    # Identification
    name: ClassVar[str]
    agent_id: Optional[str] = None

    # Contracts type-safe
    receives: ClassVar[List[Type[BaseEvent]]] = []
    emits: ClassVar[List[Type[BaseEvent]]] = []

    # Configuration
    config_class: ClassVar[Type[ComponentConfig]] = ComponentConfig

    # State
    metrics: ComponentMetrics = ComponentMetrics()
    _agent_ref: Optional[Any] = None
    _initialized: bool = False

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize component with configuration"""
        self.config = config or self.config_class()
        self._setup_initial_metrics()

    def _setup_initial_metrics(self):
        """Setup initial metrics"""
        self.metrics = ComponentMetrics()
        self.last_latency = 0.0

    async def on_init(self):
        """Called when agent initializes"""
        self._initialized = True
        logger.info(f"✅ Component '{self.name}' initialized")

    async def on_shutdown(self):
        """Cleanup resources"""
        logger.info(f"🛑 Component '{self.name}' shutting down")

    async def on_error(self, event: BaseEvent, error: Exception) -> List[BaseEvent]:
        """Handle errors"""
        error_msg = f"{type(error).__name__}: {str(error)}"
        self.metrics.errors += 1
        self.metrics.last_error = error_msg
        logger.error(f"❌ Error in component '{self.name}': {error_msg}")
        return [ErrorEvent(
            sender=self.name,
            target=self.agent_id or "unknown",
            error=error_msg,
            original_event=event
        )]

    async def _safe_handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Wrapper that handles errors automatically"""
        try:
            start_time = time.time()

            # Validate event
            if not self._validate_event(event):
                logger.warning(f"⚠️ Component '{self.name}' received invalid event: {type(event).__name__}")
                return []

            # Update metrics
            self.metrics.events_received += 1

            # Handle event
            result = await self.handle_event(event)

            # Calculate latency
            latency = (time.time() - start_time) * 1000
            self._update_latency(latency)

            # Update metrics
            self.metrics.events_emitted += len(result)

            return result

        except Exception as e:
            return await self.on_error(event, e)

    def _validate_event(self, event: BaseEvent) -> bool:
        """Validate if component can receive this event"""
        return type(event) in self.receives

    def _update_latency(self, latency: float):
        """Update average latency (EWMA)"""
        alpha = 0.1  # Smoothing factor
        self.metrics.avg_latency_ms = (
            alpha * latency +
            (1 - alpha) * self.metrics.avg_latency_ms
        )

    @abstractmethod
    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Handle events - must be implemented by subclasses"""
        pass

# ===========================================
# Agent Class
# ===========================================

class Agent:
    """
    Agent that manages components
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.components: Dict[str, Component] = {}
        self._running = False

    async def add_component(self, component_class: Type[Component], config: Optional[ComponentConfig] = None, **kwargs):
        """Add component to agent"""
        try:
            component = component_class(config=config, **kwargs)
            component.set_agent_id(self.agent_id)
            component.set_agent_ref(self)

            if component.name in self.components:
                raise ValueError(f"Component '{component.name}' already exists")

            self.components[component.name] = component
            logger.info(f"✅ Component '{component.name}' added to agent '{self.agent_id}'")

        except Exception as e:
            logger.error(f"❌ Error adding component: {e}")
            raise

    async def init_all(self):
        """Initialize all components"""
        logger.info(f"🚀 Initializing agent '{self.agent_id}' with {len(self.components)} components")

        for name, component in self.components.items():
            try:
                await component.on_init()
            except Exception as e:
                logger.error(f"❌ Error initializing '{name}': {e}")
                raise

        self._running = True
        logger.info(f"✅ Agent '{self.agent_id}' initialized")

    async def shutdown_all(self):
        """Shutdown all components"""
        logger.info(f"🛑 Shutting down agent '{self.agent_id}'")

        self._running = False

        for name, component in self.components.items():
            try:
                await component.on_shutdown()
            except Exception as e:
                logger.error(f"❌ Error shutting down '{name}': {e}")

        logger.info(f"✅ Agent '{self.agent_id}' shut down")

    def set_agent_id(self, agent_id: str):
        """Set agent ID for a component"""
        pass

    def set_agent_ref(self, agent_ref: Any):
        """Set agent reference for a component"""
        pass

# ===========================================
# Additional Classes for Export
# ===========================================

class BaseRequest(BaseModel):
    """Base request class"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timeout_ms: int = 5000
    meta: Dict[str, Any] = Field(default_factory=dict)

class BaseResponse(BaseModel):
    """Base response class"""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

class ComponentMiddleware(ABC):
    """Base middleware for components"""

    async def before_handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        """Process event before handle"""
        return event

    async def after_handle(self, event: BaseEvent, result: List[BaseEvent]) -> List[BaseEvent]:
        """Process result after handle"""
        return result

class AsyncComponent(Component):
    """Async component that can receive requests"""

    @abstractmethod
    async def request(self, req: BaseRequest) -> BaseResponse:
        """Handle requests"""
        pass

class ComponentTestHarness:
    """Test harness for components"""

    def __init__(self, component: Component):
        self.component = component
        self.sent_events: List[BaseEvent] = []

    async def send(self, event: BaseEvent) -> List[BaseEvent]:
        result = await self.component._safe_handle_event(event)
        self.sent_events.extend(result)
        return result

    def get_metrics(self) -> ComponentMetrics:
        return self.component.metrics

    def reset(self):
        self.sent_events.clear()
        self.component._setup_initial_metrics()
