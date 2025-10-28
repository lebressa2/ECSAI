# ===========================================
# ECSA Component Architecture Refactored
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

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class BaseEvent(BaseModel):
    """Base event with metadata"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    target: str
    type: str
    timestamp: float = Field(default_factory=time.time)
    meta: Dict[str, Any] = Field(default_factory=dict)

class BaseRequest(BaseModel):
    """Base for typed requests"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timeout_ms: int = 5000
    meta: Dict[str, Any] = Field(default_factory=dict)

class BaseResponse(BaseModel):
    """Base for typed responses"""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

class ErrorEvent(BaseEvent):
    """Error event"""
    type: str = "error"
    error: str
    original_event: Optional[BaseEvent] = None

# ===========================================
# Middleware System
# ===========================================

class ComponentMiddleware(ABC):
    """Base middleware to intercept events"""

    async def before_handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        """Process event before handle. Return None to cancel."""
        return event

    async def after_handle(
        self,
        event: BaseEvent,
        result: List[BaseEvent]
    ) -> List[BaseEvent]:
        """Process result after handle"""
        return result

# ===========================================
# Component Base Class
# ===========================================

class Component(ABC):
    """
    Componente base com nova API poderosa e robusta
    """

    # Identificação
    name: ClassVar[str]
    agent_id: Optional[str] = None

    # Contracts type-safe
    receives: ClassVar[List[Type[BaseEvent]]] = []
    emits: ClassVar[List[Type[BaseEvent]]] = []

    # Configuração
    config_class: ClassVar[Type[ComponentConfig]] = ComponentConfig

    # Middleware e observability
    middlewares: List[ComponentMiddleware] = []
    metrics: ComponentMetrics = ComponentMetrics()

    # Estado interno
    _agent_ref: Optional[Any] = None  # Referência ao agent pai
    _initialized: bool = False

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Inicializa componente com configuração"""
        self.config = config or self.config_class()
        self._setup_initial_metrics()

    def _setup_initial_metrics(self):
        """Configura métricas iniciais"""
        self.metrics = ComponentMetrics()
        self.last_latency = 0.0

    def set_agent_id(self, agent_id: str):
        """Define ID do agente pai"""
        self.agent_id = agent_id

    def set_agent_ref(self, agent_ref: Any):
        """Define referência ao agente pai para comunicação inter-componente"""
        self._agent_ref = agent_ref

    # ===========================================
    # Lifecycle Hooks
    # ===========================================

    async def on_init(self):
        """Chamado quando o agente inicializa"""
        self._initialized = True
        logger.info(f"✅ Component '{self.name}' inicializado")

    async def on_shutdown(self):
        """Cleanup de recursos"""
        logger.info(f"🛑 Component '{self.name}' finalizando")

    async def on_reload(self, new_config: ComponentConfig):
        """Hot reload de configuração"""
        old_config = self.config
        self.config = new_config
        logger.info(f"🔄 Component '{self.name}' recarregado com nova configuração")

    # ===========================================
    # Validation & Safety
    # ===========================================

    def _validate_event(self, event: BaseEvent) -> bool:
        """Valida se o componente pode receber este evento"""
        return type(event) in self.receives

    async def _validate_contracts(self):
        """Valida contratos declarados vs implementados"""
        # Pode implementar validação mais sofisticada
        pass

    # ===========================================
    # Error Handling
    # ===========================================

    async def on_error(self, event: BaseEvent, error: Exception) -> List[BaseEvent]:
        """
        Hook para tratamento personalizado de erros.
        Override para customizar comportamento.
        """
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
        """Wrapper que trata erros automaticamente"""
        try:
            start_time = time.time()

            # Validação prévia
            if not self._validate_event(event):
                logger.warning(f"⚠️ Component '{self.name}' recebeu evento inválido: {type(event).__name__}")
                return []

            # Atualiza métricas
            self.metrics.events_received += 1

            # Aplica middlewares (before)
            processed_event = event
            for mw in self.middlewares:
                processed_event = await mw.before_handle(processed_event)
                if processed_event is None:
                    return []  # Middleware cancelou o evento

            # Handle principal com timer
            result = await self.handle_event(processed_event)

            # Calcula latência
            latency = (time.time() - start_time) * 1000
            self._update_latency(latency)

            # Atualiza métricas
            self.metrics.events_emitted += len(result)

            # Aplica middlewares (after)
            for mw in self.middlewares:
                result = await mw.after_handle(processed_event, result)

            return result

        except Exception as e:
            return await self.on_error(event, e)

    def _update_latency(self, latency: float):
        """Atualiza latência média (EWMA)"""
        alpha = 0.1  # Fator de suavização
        self.metrics.avg_latency_ms = (
            alpha * latency +
            (1 - alpha) * self.metrics.avg_latency_ms
        )

    # ===========================================
    # Core API
    # ===========================================

    @abstractmethod
    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Handle de eventos - deve ser implementado pelos componentes"""
        pass

    # ===========================================
    # Inter-Component Communication
    # ===========================================

    async def call_component(
        self,
        component_name: str,
        request: BaseRequest
    ) -> BaseResponse:
        """
        Chama AsyncComponent do mesmo agent diretamente
        """
        if self._agent_ref is None:
            raise RuntimeError(f"Component '{self.name}' não inicializado com agent_ref")

        if not hasattr(self._agent_ref, 'components') or not isinstance(self._agent_ref.components, dict):
            raise RuntimeError("Agent não tem registro de componentes válido")

        target_component = self._agent_ref.components.get(component_name)
        if target_component is None:
            raise ValueError(f"Component '{component_name}' não encontrado no agent")

        if not isinstance(target_component, AsyncComponent):
            raise TypeError(f"Component '{component_name}' não é AsyncComponent")

        try:
            response = await target_component.request(request)

            if not isinstance(response, BaseResponse):
                return BaseResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Response inválida de {component_name}"
                )

            return response

        except Exception as e:
            logger.error(f"Erro chamando componente '{component_name}': {e}")
            return BaseResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )

    # ===========================================
    # Info & Diagnostics
    # ===========================================

    def get_info(self) -> Dict[str, Any]:
        """Retorna informações diagnósticas do componente"""
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "config": self.config.model_dump(),
            "metrics": self.metrics.model_dump(),
            "middlewares": [type(mw).__name__ for mw in self.middlewares],
            "initialized": self._initialized,
            "receives": [t.__name__ for t in self.receives],
            "emits": [t.__name__ for t in self.emits]
        }

# ===========================================
# Async Component
# ===========================================

class AsyncComponent(Component):
    """
    Componente que pode receber requests diretos (tipo RPC)
    """

    @abstractmethod
    async def request(self, req: BaseRequest) -> BaseResponse:
        """Handle de requests tipados"""
        pass

# ===========================================
# Agent Registry & Factory
# ===========================================

class Agent:
    """
    Agent que gerencia componentes com nova arquitetura
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.components: Dict[str, Component] = {}
        self._running = False
        self._external_bus: Optional[Any] = None  # External bus

    def set_external_bus(self, bus):
        """Define o bus externo para comunicação inter-agente"""
        self._external_bus = bus

    async def add_component(self, component_class: Type[Component], config: Optional[ComponentConfig] = None, **kwargs):
        """Adiciona componente ao agent"""
        try:
            # Instancia
            component = component_class(config=config, **kwargs)

            # Seta propriedades
            component.set_agent_id(self.agent_id)
            component.set_agent_ref(self)

            # Adiciona ao registro
            if component.name in self.components:
                raise ValueError(f"Component '{component.name}' já existe no agent")

            self.components[component.name] = component
            logger.info(f"✅ Component '{component.name}' adicionado ao agent '{self.agent_id}'")

        except Exception as e:
            logger.error(f"❌ Erro adicionando componente '{component_class.__name__}': {e}")
            raise

    async def dispatch_intra(self, events: List[BaseEvent]) -> List[BaseEvent]:
        """
        Dispara eventos intra-agente: targets são nomes de componentes diretamente.
        Processa recursivamente os eventos gerados.
        """
        pending_events = events.copy()
        all_new_events = []

        while pending_events:
            event = pending_events.pop(0)
            new_events = await self._handle_intra_event(event)
            pending_events.extend(new_events)
            all_new_events.extend(new_events)

        return all_new_events

    async def _handle_intra_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Handle um evento intra-agente"""
        target = event.target
        if not target:
            logger.warning("Evento sem target em dispatch_intra")
            return []

        # Assume target é nome do componente
        if target not in self.components:
            logger.error(f"Component '{target}' não encontrado no agent '{self.agent_id}'")
            return []

        component = self.components[target]
        return await component._safe_handle_event(event)

    async def init_all(self):
        """Inicializa todos os componentes"""
        logger.info(f"🚀 Inicializando agent '{self.agent_id}' com {len(self.components)} componentes")

        for name, component in self.components.items():
            try:
                await component.on_init()
            except Exception as e:
                logger.error(f"❌ Erro inicializando '{name}': {e}")
                raise

        self._running = True
        logger.info(f"✅ Agent '{self.agent_id}' inicializado")

    async def shutdown_all(self):
        """Finaliza todos os componentes"""
        logger.info(f"🛑 Finalizando agent '{self.agent_id}'")

        self._running = False

        for name, component in self.components.items():
            try:
                await component.on_shutdown()
            except Exception as e:
                logger.error(f"❌ Erro finalizando '{name}': {e}")

        logger.info(f"✅ Agent '{self.agent_id}' finalizado")

    async def send_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Envia evento para componente apropriado (usado pelo bus externo)"""
        if not self._running:
            raise RuntimeError("Agent não está inicializado")

        if ":" in event.target:
            # Target format: "agent_id:component_name"
            target_agent, component_name = event.target.split(":", 1)
            if target_agent != self.agent_id:
                raise ValueError(f"Target agent '{target_agent}' doesn't match agent '{self.agent_id}'")
        else:
            # No agent prefix, assume component name directly
            component_name = event.target

        if component_name not in self.components:
            raise ValueError(f"Component '{component_name}' não encontrado")

        component = self.components[component_name]
        return await component._safe_handle_event(event)

# ===========================================
# Testing Harness (Nice-to-have)
# ===========================================

class ComponentTestHarness:
    """Helper para testar componentes isoladamente"""

    def __init__(self, component: Component):
        self.component = component
        self.sent_events: List[BaseEvent] = []
        # Mock agent ref minimal
        component._agent_ref = self

    async def send(self, event: BaseEvent) -> List[BaseEvent]:
        result = await self.component._safe_handle_event(event)
        self.sent_events.extend(result)
        return result

    def assert_emitted(self, event_type: str, count: int = 1):
        actual = len([e for e in self.sent_events if e.type == event_type])
        assert actual == count, f"Expected {count} {event_type}, got {actual}"

    def get_metrics(self) -> ComponentMetrics:
        return self.component.metrics

    def reset(self):
        self.sent_events.clear()
        self.component._setup_initial_metrics()

# ===========================================
# Demo / Example Usage
# ===========================================

if __name__ == "__main__":
    print("🚀 ECSA Component Architecture Refactored Demo")
    print("=" * 50)

    async def demo():
        # Cria agent
        agent = Agent("demo_agent")

        # Adiciona componentes (exemplo simples)
        # await agent.add_component(MyComponent, config=my_config)
        # await agent.add_component(MyAsyncComponent, config=async_config)

        print("✅ Arquitetura pronta para uso!")
        print("\n📋 Implementado:")
        print("  • Lifecycle hooks (on_init, on_shutdown, on_reload)")
        print("  • Error handling com on_error hook")
        print("  • Type-safe event contracts")
        print("  • call_component() para comunicação inter-componente")
        print("  • Configuration schemas com Pydantic")
        print("  • Middleware support")
        print("  • Basic observability (métricas)")
        print("  • ComponentTestHarness para testes")
        print("\n🎯 Pronto para criação de componentes customizados!")

    asyncio.run(demo())
