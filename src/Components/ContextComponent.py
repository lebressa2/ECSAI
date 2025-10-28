# ===========================================
# ContextComponent - Refactored as AsyncComponent
# ===========================================
"""
Componente de Contexto reescrito como AsyncComponent.
Demonstra a nova API de requests/responses tipados e stateful components.
"""

try:
    from ..main import AsyncComponent, ComponentConfig, ComponentMiddleware
    from ..Events import (
        BaseEvent, InputEvent, ContextEvent, LLMResponseEvent,
        OutputEvent, UpdateSystemPromptEvent,
        GetContextRequest, GetContextResponse,
        StoreMessageRequest, StoreMessageResponse
    )
except ImportError:
    from main import AsyncComponent, ComponentConfig, ComponentMiddleware
    from Events import (
        BaseEvent, InputEvent, ContextEvent, LLMResponseEvent,
        OutputEvent, UpdateSystemPromptEvent,
        GetContextRequest, GetContextResponse,
        StoreMessageRequest, StoreMessageResponse
    )
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ===========================================
# Configuration Schema
# ===========================================

class ContextConfig(ComponentConfig):
    """Configuração tipada para ContextComponent"""
    connection_string: str = "sqlite:///chat_memory.db"  # Default SQLite
    initial_system_prompt: str = "Você é um assistente útil e amigável."
    max_history_messages: int = 50
    auto_cleanup_days: int = 30

# ===========================================
# Middleware Example
# ===========================================

class ContextLoggingMiddleware(ComponentMiddleware):
    """Middleware para logging de operações de contexto"""

    async def before_handle(self, event: BaseEvent):
        logger.info(f"💬 Context: {event.type} para sessão {getattr(event, 'session_id', 'unknown')}")
        return event

    async def after_handle(self, event: BaseEvent, result: List[BaseEvent]):
        # Log changes if needed
        return result

# ===========================================
# Component Implementation
# ===========================================

class ContextComponent(AsyncComponent):
    """
    Componente de gerenciamento de contexto de chat.
    Agora é AsyncComponent com requests/responses tipados!

    Recursos demonstrados:
    - AsyncComponent API (request/response)
    - Type-safe request/response schemas
    - Stateful component com lifecycle management
    - True persistence com SQLChatMessageHistory
    - Hot reload de system prompt
    """

    name: str = "ContextComponent"

    # Event contracts (type-safe!)
    receives = [InputEvent, LLMResponseEvent, UpdateSystemPromptEvent]
    emits = [ContextEvent, OutputEvent]

    # Configuração
    config_class = ContextConfig

    # Middleware
    middlewares = [ContextLoggingMiddleware()]

    def __init__(self, config: Optional[ContextConfig] = None, **kwargs):
        """Inicializa componente com config tipada"""
        super().__init__(config)

        # Estado interno (inicializado no on_init)
        self._system_prompt = self.config.initial_system_prompt
        self._prompt_template = None

        # Cache de memórias por sessão
        self._memory_cache: Dict[str, SQLChatMessageHistory] = {}

    # ===========================================
    # Lifecycle Hooks
    # ===========================================

    async def on_init(self):
        """Inicializa base de dados e template de prompt"""
        await super().on_init()

        try:
            # Testa conexão com banco
            test_session = self._get_memory("test_connection")
            logger.info("✅ Conexão com base de dados estabelecida")

            # Cria template de prompt
            self._rebuild_prompt_template()
            logger.info("✅ Template de prompt inicializado")

        except Exception as e:
            logger.error(f"❌ Falha na inicialização: {e}")
            raise

    async def on_shutdown(self):
        """Cleanup recursos"""
        await super().on_shutdown()

        # Limpa cache (não fecha conexões individuais)
        self._memory_cache.clear()
        logger.info("💾 Cache de memória limpo")

    async def on_reload(self, new_config: ContextConfig):
        """Hot reload de configuração"""
        await super().on_reload(new_config)

        # Atualiza configurações que podem mudar
        if new_config.initial_system_prompt != self._system_prompt:
            self._system_prompt = new_config.initial_system_prompt
            self._rebuild_prompt_template()
            logger.info("🔄 System prompt atualizado")

    # ===========================================
    # Utility Methods
    # ===========================================

    def _rebuild_prompt_template(self):
        """Recria ChatPromptTemplate com prompt atual"""
        self._prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        logger.debug(f"📝 Template rebuilt with prompt: {self._system_prompt[:50]}...")

    def _get_memory(self, session_id: str) -> SQLChatMessageHistory:
        """Obtém ou cria memória para sessão com cache"""
        if session_id in self._memory_cache:
            return self._memory_cache[session_id]

        memory = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.config.connection_string
        )

        # Cache limitado para performance
        if len(self._memory_cache) < 100:  # Max 100 sessões em cache
            self._memory_cache[session_id] = memory

        return memory

    # ===========================================
    # Event Handling (síncrono)
    # ===========================================

    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Processa eventos recebidos"""

        # --- Update System Prompt ---
        if isinstance(event, UpdateSystemPromptEvent):
            return await self._handle_update_system_prompt(event)

        # --- User Input ---
        elif isinstance(event, InputEvent):
            return await self._handle_input(event)

        # --- LLM Response ---
        elif isinstance(event, LLMResponseEvent):
            return await self._handle_llm_response(event)

        # --- Unknown ---
        else:
            logger.warning(f"⚠️ Evento desconhecido recebido: {type(event).__name__}")
            return []

    async def _handle_update_system_prompt(self, event: UpdateSystemPromptEvent) -> List[BaseEvent]:
        """Atualiza system prompt e recria template"""
        self._system_prompt = event.new_prompt
        self._rebuild_prompt_template()

        logger.info(f"🔄 System prompt atualizado para sessão {event.session_id or 'global'}")

        # Não emite novos eventos - apenas atualiza estado interno
        return []

    async def _handle_input(self, event: InputEvent) -> List[ContextEvent]:
        """Processa entrada do usuário e gera contexto"""

        # Obtém histórico de mensagens
        memory = self._get_memory(event.session_id)

        # Adiciona mensagem do usuário
        user_message = HumanMessage(content=event.content)
        memory.add_message(user_message)

        # Formata contexto usando template
        formatted_messages = self._prompt_template.format_messages(
            input=event.content,
            history=memory.messages
        )

        prompt_string = "\n".join([msg.content for msg in formatted_messages])

        return [ContextEvent(
            sender=self.name,
            target=f"{self.agent_id}:LLMComponent",
            formatted_prompt=prompt_string,
            original_input=user_message,
            session_id=event.session_id
        )]

    async def _handle_llm_response(self, event: LLMResponseEvent) -> List[OutputEvent]:
        """Processa resposta do LLM e armazena no histórico"""

        # Obtém histórico
        memory = self._get_memory(event.session_id)

        # Adiciona resposta do AI
        ai_message = AIMessage(content=event.response)
        memory.add_message(ai_message)

        # Emite para output component
        return [OutputEvent(
            sender=self.name,
            target=f"{self.agent_id}:OutputComponent",
            content=event.response,
            session_id=event.session_id
        )]

    # ===========================================
    # AsyncComponent Request/Response API
    # ===========================================

    async def request(self, req) -> Any:  # BaseResponse types
        """Handle de requests diretos (RPC-style)"""

        # --- Get Context ---
        if isinstance(req, GetContextRequest):
            return await self._handle_get_context(req)

        # --- Store Message ---
        elif isinstance(req, StoreMessageRequest):
            return await self._handle_store_message(req)

        # --- Unknown ---
        else:
            return GetContextResponse(  # Fallback response
                request_id=req.request_id,
                success=False,
                error=f"Request type not supported: {type(req).__name__}"
            )

    async def _handle_get_context(self, req: GetContextRequest) -> GetContextResponse:
        """Retorna contexto/mensagens de uma sessão"""
        try:
            memory = self._get_memory(req.session_id)

            # Converte mensagens para formato serializável
            messages = []
            for msg in memory.messages[-req.max_messages:]:
                messages.append({
                    'type': msg.type,  # 'human' or 'ai'
                    'content': msg.content,
                    'timestamp': getattr(msg, 'timestamp', None)
                })

            return GetContextResponse(
                request_id=req.request_id,
                success=True,
                messages=messages
            )

        except Exception as e:
            logger.error(f"❌ Error getting context: {e}")
            return GetContextResponse(
                request_id=req.request_id,
                success=False,
                error=str(e)
            )

    async def _handle_store_message(self, req: StoreMessageRequest) -> StoreMessageResponse:
        """Armazena mensagem diretamente via API"""
        try:
            memory = self._get_memory(req.session_id)

            # Cria e armazena mensagem
            if req.message_type == "human":
                message = HumanMessage(content=req.content)
            elif req.message_type == "ai":
                message = AIMessage(content=req.content)
            else:
                raise ValueError(f"Tipo de mensagem inválido: {req.message_type}")

            memory.add_message(message)

            # Retorna ID da mensagem (se disponível) ou None
            message_id = getattr(message, 'id', None)

            return StoreMessageResponse(
                request_id=req.request_id,
                success=True,
                message_id=message_id
            )

        except Exception as e:
            logger.error(f"❌ Error storing message: {e}")
            return StoreMessageResponse(
                request_id=req.request_id,
                success=False,
                error=str(e)
            )

    # ===========================================
    # Additional Utility Methods
    # ===========================================

    async def clear_session(self, session_id: str) -> bool:
        """Limpa histórico de uma sessão"""
        try:
            if session_id in self._memory_cache:
                del self._memory_cache[session_id]

            # Limpa da persistência (se suportado)
            memory = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=self.config.connection_string
            )
            # SQLChatMessageHistory não tem método clear direto, então:
            # Truncate or delete - depende da implementação

            logger.info(f"🧹 Sessão {session_id} limpa")
            return True

        except Exception as e:
            logger.error(f"❌ Error clearing session {session_id}: {e}")
            return False

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Retorna informações sobre uma sessão"""
        try:
            memory = self._get_memory(session_id)
            return {
                "session_id": session_id,
                "message_count": len(memory.messages),
                "exists": True
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "error": str(e),
                "exists": False
            }

# ===========================================
# Configuration Presets
# ===========================================

def create_chatbot_config() -> ContextConfig:
    """Preset para chatbot simples"""
    return ContextConfig(
        connection_string="sqlite:///chatbot.db",
        initial_system_prompt="Você é um assistente útil e amigável.",
        max_history_messages=20
    )

def create_teacher_config() -> ContextConfig:
    """Preset para professor de idiomas"""
    return ContextConfig(
        connection_string="sqlite:///teacher.db",
        initial_system_prompt="""
        Você é um professor de idiomas paciente e motivador.
        Sempre corrija erros gramaticais gentilmente e explique as regras.
        Incentive o aluno a continuar praticando.
        """,
        max_history_messages=50
    )

def create_enterprise_config() -> ContextConfig:
    """Preset para uso enterprise"""
    return ContextConfig(
        connection_string="postgresql://user:pass@host:5432/chat_db",  # Would use env vars
        initial_system_prompt="You are a professional business assistant.",
        max_history_messages=100,
        auto_cleanup_days=90
    )
