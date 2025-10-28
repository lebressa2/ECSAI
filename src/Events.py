# ===========================================
# Event Definitions - ECSA Refactored
# ===========================================
"""
Eventos tipados para a nova arquitetura Component.
Usa Pydantic BaseModel para type safety e validação.
"""

try:
    # Quando executado como módulo
    from .main import BaseEvent, ErrorEvent
except ImportError:
    # Quando executado standalone/script (fallback)
    from main import BaseEvent, ErrorEvent
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage
import uuid
import time

# ===========================================
# Core Events
# ===========================================

class InputEvent(BaseEvent):
    """Evento de entrada do usuário"""
    type: str = "input"
    content: str = Field(..., description="Conteúdo da entrada do usuário")
    session_id: str = Field(default="default_session", description="ID da sessão")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados extras")

class ContextEvent(BaseEvent):
    """Evento com contexto formatado para LLM"""
    type: str = "context"
    formatted_prompt: Optional[str] = Field(default=None, description="Prompt formatado")
    original_input: Optional[HumanMessage] = Field(default=None, description="Entrada original")
    session_id: str = Field(default="default_session", description="ID da sessão")

class LLMResponseEvent(BaseEvent):
    """Evento com resposta do LLM"""
    type: str = "llm_response"
    response: str = Field(..., description="Resposta gerada pelo LLM")
    session_id: str = Field(default="default_session", description="ID da sessão")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumidos")

class OutputEvent(BaseEvent):
    """Evento de saída final para o usuário"""
    type: str = "output"
    content: str = Field(..., description="Conteúdo da saída")
    session_id: str = Field(default="default_session", description="ID da sessão")

class UpdateSystemPromptEvent(BaseEvent):
    """Evento para atualizar prompt do sistema"""
    type: str = "update_system_prompt"
    new_prompt: str = Field(..., description="Novo prompt do sistema")
    session_id: Optional[str] = Field(default=None, description="Sessão específica (None = global)")

# ===========================================
# Request/Response Types for AsyncComponents
# ===========================================

def generate_uuid():
    return str(uuid.uuid4())

class GetContextRequest(BaseModel):
    """Request para obter contexto de uma sessão"""
    request_id: str = Field(default_factory=generate_uuid)
    session_id: str = Field(..., description="ID da sessão")
    max_messages: int = Field(default=10, description="Número máximo de mensagens")
    timeout_ms: int = Field(default=5000, description="Timeout em ms")

class GetContextResponse(BaseModel):
    """Response com mensagens de contexto"""
    request_id: str
    success: bool
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Mensagens de histórico")
    error: Optional[str] = None

class StoreMessageRequest(BaseModel):
    """Request para armazenar mensagem"""
    request_id: str = Field(default_factory=generate_uuid)
    session_id: str = Field(..., description="ID da sessão")
    message_type: str = Field(..., description="Tipo da mensagem (human/ai)")  # "human" or "ai"
    content: str = Field(..., description="Conteúdo da mensagem")
    timeout_ms: int = Field(default=5000, description="Timeout em ms")

class StoreMessageResponse(BaseModel):
    """Response de confirmação de armazenamento"""
    request_id: str
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None

class LLMRequest(BaseModel):
    """Request para geração de LLM"""
    request_id: str = Field(default_factory=generate_uuid)
    prompt: str = Field(...)
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = Field(default=30000)

class LLMResponse(BaseModel):
    """Response da geração de LLM"""
    request_id: str
    success: bool
    response: Optional[str] = None
    tokens_used: Optional[int] = None
    error: Optional[str] = None

# ===========================================
# System Events
# ===========================================

class ComponentStatusEvent(BaseEvent):
    """Evento de status do componente"""
    type: str = "component_status"
    component_name: str = Field(..., description="Nome do componente")
    status: str = Field(..., description="Status atual")  # "initialized", "error", "shutdown"
    message: Optional[str] = Field(default=None, description="Mensagem adicional")

class AgentStatusEvent(BaseEvent):
    """Evento de status do agent"""
    type: str = "agent_status"
    agent_id: str = Field(..., description="ID do agent")
    status: str = Field(..., description="Status")  # "starting", "running", "stopping", "stopped"
    components_count: int = Field(default=0, description="Número de componentes")

class MetricsEvent(BaseEvent):
    """Evento com métricas de componente"""
    type: str = "metrics"
    component_name: str = Field(..., description="Nome do componente")
    metrics: Dict[str, Any] = Field(..., description="Dados das métricas")
    timestamp: float = Field(default_factory=time.time)

# ===========================================
# Error Event (já definido no main.py)
# ===========================================
# ErrorEvent é importado do main.py

__all__ = [
    # Core Events
    'InputEvent',
    'ContextEvent',
    'LLMResponseEvent',
    'OutputEvent',
    'UpdateSystemPromptEvent',

    # Request/Response
    'GetContextRequest',
    'GetContextResponse',
    'StoreMessageRequest',
    'StoreMessageResponse',
    'LLMRequest',
    'LLMResponse',

    # System Events
    'ComponentStatusEvent',
    'AgentStatusEvent',
    'MetricsEvent',

    # Error
    'ErrorEvent',
]
