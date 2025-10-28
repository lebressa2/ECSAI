# ===========================================
# LLMComponent - Refactored for New Architecture
# ===========================================
"""
Componente LLM com nova API robusta e poderosa.
Demonstra o uso de configurações Pydantic, lifecycle hooks,
error handling personalizado, middlewares, e comunicação inter-componente.
"""

try:
    # Quando executado como módulo
    from typing import Any
    from ..main import Component, ComponentConfig, ComponentMiddleware
    from ..Events import ContextEvent, LLMResponseEvent, LLMRequest, LLMResponse
    try:
        from ..libs.LLMFactory import LLMFactory
    except ImportError:
        # Fallback to mock factory
        class LLMFactory:
            @staticmethod
            def create_llm(llm_id: str, config: dict = None):
                class MockLLM:
                    async def ainvoke(self, prompt: str):
                        class MockResponse:
                            content = f"[MOCK RESPONSE] Processado prompt de {len(prompt)} caracteres: {prompt[:50]}..."
                            usage_metadata = {'total_tokens': len(prompt.split()) * 2}
                        return MockResponse()
                return MockLLM()
except ImportError:
    # Quando executado standalone/script (fallback)
    from main import Component, ComponentConfig, ComponentMiddleware
    from Events import ContextEvent, LLMResponseEvent, LLMRequest, LLMResponse
    # Use mock factory for standalone execution
    class LLMFactory:
        @staticmethod
        def create_llm(llm_id: str, config: dict = None):
            class MockLLM:
                async def ainvoke(self, prompt: str):
                    class MockResponse:
                        content = f"[MOCK RESPONSE] Processado prompt de {len(prompt)} caracteres: {prompt[:50]}..."
                        usage_metadata = {'total_tokens': len(prompt.split()) * 2}
                    return MockResponse()
            return MockLLM()
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# ===========================================
# Configuration Schema
# ===========================================

class LLMConfig(ComponentConfig):
    """Configuração tipada para LLMComponent"""
    llm_id: str = "google:gemini-2.5-flash"  # Modelo padrão
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    tools: Optional[List[Any]] = None  # Lista de tools

# ===========================================
# Middleware Examples
# ===========================================

class LoggingMiddleware(ComponentMiddleware):
    """Middleware que faz logging de todas as requisições"""

    async def before_handle(self, event):
        logger.info(f"📥 LLMComponent recebendo: {event.type}")
        return event

    async def after_handle(self, event, result):
        logger.info(f"📤 LLMComponent emitindo: {len(result)} eventos")
        return result

class RetryMiddleware(ComponentMiddleware):
    """Middleware que implementa retry automático"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def before_handle(self, event):
        # Adiciona informações de retry no metadata
        if 'retry_count' not in event.meta:
            event.meta['retry_count'] = 0
        return event

# ===========================================
# Component Implementation
# ===========================================

class LLMComponent(Component):
    """
    Componente LLM com nova arquitetura.

    Demonstra:
    - Configurações Pydantic tipadas
    - Lifecycle hooks (on_init, on_shutdown, on_reload)
    - Error handling personalizado
    - Middlewares automáticos
    - Communicação inter-componente (call_component)
    - Type-safe event contracts
    - Observability com métricas
    """

    name: str = "LLMComponent"

    # Contracts type-safe (ao invés de strings!)
    receives = [ContextEvent]  # Recebe eventos ContextEvent
    emits = [LLMResponseEvent]  # Emite eventos LLMResponseEvent

    # Configuração
    config_class = LLMConfig

    # Middlewares ativos
    middlewares = [LoggingMiddleware(), RetryMiddleware()]

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """Inicializa componente com config tipada"""
        super().__init__(config)

        # Config valida automaticamente via Pydantic!
        self.llm = None  # Inicializado no on_init

        # Estado adicional
        self._request_count = 0

    # ===========================================
    # Lifecycle Hooks
    # ===========================================

    async def on_init(self):
        """Inicializa LLM quando o agent inicia"""
        await super().on_init()

        try:
            # Cria LLM com configurações validadas
            llm_config = {
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'timeout': self.config.timeout_seconds,
            }

            self.llm = LLMFactory.create_llm(self.config.llm_id, llm_config, tools=self.config.tools)
            logger.info(f"🧠 LLM '{self.config.llm_id}' inicializado com sucesso")

        except Exception as e:
            logger.error(f"❌ Falha ao inicializar LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")

    async def on_shutdown(self):
        """Cleanup quando o agent finaliza"""
        await super().on_shutdown()

        # Disconnect LLM ou cleanup resources
        self.llm = None
        logger.info("🧠 LLM desconectado")

    async def on_reload(self, new_config: LLMConfig):
        """Hot reload de configuração LLM"""
        await super().on_reload(new_config)

        try:
            # Recria LLM com nova config
            llm_config = {
                'temperature': new_config.temperature,
                'max_tokens': new_config.max_tokens,
                'timeout': new_config.timeout_seconds,
            }

            old_llm = self.llm
            self.llm = LLMFactory.create_llm(new_config.llm_id, llm_config, tools=new_config.tools)

            # Dispose do antigo
            if old_llm:
                del old_llm

            logger.info(f"🔄 LLM recarregado: {new_config.llm_id}")

        except Exception as e:
            logger.error(f"❌ Falha no hot reload do LLM: {e}")
            raise

    # ===========================================
    # Error Handling Personalizado
    # ===========================================

    async def on_error(self, event: ContextEvent, error: Exception) -> List[LLMResponseEvent]:
        """Tratamento personalizado de erros LLM"""
        retry_count = event.meta.get('retry_count', 0)

        # Se pode tentar novamente, retorna evento vazio para retry
        if retry_count < self.config.max_retries and self._is_retryable_error(error):
            logger.warning(f"🔄 Tentando novamente LLM (tentativa {retry_count + 1})")

            # Modifica evento para próxima tentativa
            event.meta['retry_count'] = retry_count + 1
            return []  # Retry: não emite erro ainda

        # Erro final: usa implementação pai
        return await super().on_error(event, error)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Verifica se erro permite retry"""
        error_str = str(error).lower()
        retryable = ['timeout', 'rate limit', 'connection', 'temporary']
        return any(keyword in error_str for keyword in retryable)

    # ===========================================
    # Core Logic
    # ===========================================

    async def handle_event(self, event: ContextEvent) -> List[LLMResponseEvent]:
        """
        Processa evento de contexto e gera resposta LLM.
        Demonstra uso de call_component para comunicação interna.
        """

        # Validação adicional
        if not event.formatted_prompt:
            logger.warning("⚠️ Recebeu ContextEvent sem formatted_prompt")
            return [LLMResponseEvent(
                sender=self.name,
                target=self.agent_id or "unknown",
                response="[ERRO]: Prompt não fornecido",
                session_id=event.session_id
            )]

        try:
            # Chamada para LLM (poderia ser AsyncComponent)
            response = await self._call_llm(event.formatted_prompt)
            tokens_used = getattr(response, 'usage_metadata', {}).get('total_tokens', None)

            # Poderia chamar outro componente aqui usando call_component!
            # Exemplo: await self.call_component("MetricsComponent", StoreMetricsRequest(...))

            return [LLMResponseEvent(
                sender=self.name,
                target=f"{self.agent_id}:ContextComponent",  # Ou OutputComponent
                response=response.content or str(response),
                session_id=event.session_id,
                tokens_used=tokens_used
            )]

        except Exception as e:
            # on_error será chamado automaticamente pelo _safe_handle_event
            raise  # Re-raise para error handling

    async def _call_llm(self, prompt: str) -> any:
        """Chamada isolada para LLM para facilitar testing"""
        if not self.llm:
            raise RuntimeError("LLM não inicializado")

        self._request_count += 1
        logger.info(f"🤖 Chamada LLM #{self._request_count}: prompt com {len(prompt)} chars")

        return await self.llm.ainvoke(prompt)

    # ===========================================
    # Observability
    # ===========================================

    def record_custom_metric(self, metric_name: str, value: float):
        """Registra métrica customizada"""
        if not hasattr(self, '_custom_metrics'):
            self._custom_metrics = {}

        self._custom_metrics[metric_name] = value
        logger.debug(f"📊 Métrica custom: {metric_name} = {value}")

    # ===========================================
    # Utility Methods
    # ===========================================

    async def get_health_check(self) -> dict:
        """Health check do componente"""
        return {
            "status": "healthy" if self.llm else "unhealthy",
            "llm_id": self.config.llm_id,
            "requests_count": self._request_count,
            "metrics": self.metrics.model_dump(),
            "initialization_time": getattr(self, '_init_time', None)
        }

# ===========================================
# Alternative Configuration Presets
# ===========================================

def create_fast_llm_config() -> LLMConfig:
    """Preset para LLM rápido (baixo latency)"""
    return LLMConfig(
        llm_id="google:gemini-2.5-flash",
        temperature=0.3,
        max_tokens=1000,
        timeout_seconds=10
    )

def create_creative_llm_config() -> LLMConfig:
    """Preset para geração criativa"""
    return LLMConfig(
        llm_id="openai:gpt-4",
        temperature=0.9,
        max_tokens=2000,
        timeout_seconds=60
    )

def create_analytical_llm_config() -> LLMConfig:
    """Preset para análise e raciocínio"""
    return LLMConfig(
        llm_id="anthropic:claude-3-haiku",
        temperature=0.1,
        max_tokens=1500,
        timeout_seconds=45,
        system_prompt="Você é um assistente analítico. Sempre explique seu raciocínio passo a passo."
    )
