# ===========================================
# LLMFactory - 100% Standalone
# ===========================================
"""
Fábrica standalone para criação de LLMs com suporte a ferramenta.
Abstrai as minúcias das APIs de LLMs, permitindo plug-in em qualquer script.

Uso:
    from LLMFactory import LLMFactory
    llm = LLMFactory.create_llm("google:gemini-2.5-flash", tools=[...], config={...})
    response = await llm.run("Olá")  # Aceita string, BaseMessage, ou list[BaseMessage]
"""

import os
from typing import Any, Optional, List, Dict, Callable, Union
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
import asyncio
import logging
from enum import Enum

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ===========================================
# Tool Execution Strategies
# ===========================================

class ToolExecutionMode(Enum):
    """Estratégias de execução de tools"""
    AUTOMATIC = "automatic"      # Executa tools automaticamente
    MANUAL = "manual"           # Retorna tool calls para execução manual
    HYBRID = "hybrid"           # Pergunta ao usuário antes de executar

class ToolExecutionResult(BaseModel):
    """Resultado da execução de uma tool"""
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tool_call_id: Optional[str] = None

# ===========================================
# LLMFactory - Standalone Factory
# ===========================================

class LLMFactory:
    """
    Fábrica para criar LLMs de forma unificada a partir de strings de especificação.
    Abstrai as diferenças entre provedores (Google, OpenAI, Anthropic, etc.)
    """

    @staticmethod
    def create_llm(
        model_spec: str,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[BaseTool]] = None
    ) -> "ToolAwareLLM":
        """
        Cria uma instância ToolAwareLLM com o provedor e modelo especificado.

        Args:
            model_spec: String no formato "provider:model" (ex: "google:gemini-2.5-flash")
            config: Dicionário de configurações opcionais (temperature, max_tokens, etc.)
            tools: Lista de ferramentas a serem usadas com o LLM

        Returns:
            Instância ToolAwareLLM configurada

        Raises:
            ValueError: Se provedor ou modelo inválido/não suportado
        """
        config = config or {}

        # Parse model_spec
        if ":" not in model_spec:
            raise ValueError("model_spec deve estar no formato 'provider:model'")
        provider, model = model_spec.split(":", 1)
        provider = provider.lower()

        # Instantiate the LLM based on provider
        llm_instance = LLMFactory._create_llm_instance(provider, model, config)

        # Wrap in ToolAwareLLM
        tool_aware_llm = ToolAwareLLM(
            llm_instance=llm_instance,
            tools=tools or [],
            execution_mode=config.get('execution_mode', ToolExecutionMode.AUTOMATIC),
            max_tool_iterations=config.get('max_tool_iterations', 3),
            tool_timeout_seconds=config.get('tool_timeout_seconds', 30.0)
        )

        logger.info(f"🧠 LLM criado: {model_spec} com {len(tools) if tools else 0} ferramentas")
        return tool_aware_llm

    @staticmethod
    def _create_llm_instance(provider: str, model: str, config: Dict[str, Any]) -> Any:
        """Cria instância do LLM baseado no provedor"""

        # Common config params with env fallback for api_key
        api_key = config.get('api_key')
        common_config = {
            'model': model,
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens'),
        }

        # Get API key from env if not provided (auto-detect from provider)
        if not api_key:
            # Auto-detect API key from .env using provider name
            api_key_env_var = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_env_var)
            if api_key:
                logger.info(f"🔑 API key loaded from {api_key_env_var}")

        # Set API key in common config
        common_config['api_key'] = api_key

        # Remove None values to avoid passing kwargs with None
        common_config = {k: v for k, v in common_config.items() if v is not None}

        if provider == 'google':
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Google usa GOOGLE_API_KEY
                return ChatGoogleGenerativeAI(**common_config)
            except ImportError:
                raise ImportError("langchain_google_genai não instalado. Instale com: pip install langchain-google-genai")

        elif provider == 'openai':
            try:
                from langchain_openai import ChatOpenAI
                # OpenAI usa OPENAI_API_KEY
                return ChatOpenAI(**common_config)
            except ImportError:
                raise ImportError("langchain_openai não instalado. Instale com: pip install langchain-openai")

        elif provider == 'anthropic':
            try:
                from langchain_anthropic import ChatAnthropic
                # Anthropic usa ANTHROPIC_API_KEY
                return ChatAnthropic(**common_config)
            except ImportError:
                raise ImportError("langchain_anthropic não instalado. Instale com: pip install langchain-anthropic")

        elif provider == 'openrouter':
            try:
                from langchain_openai import ChatOpenAI
                # OpenRouter usa API compatível com OpenAI
                openrouter_config = {
                    'model': model,
                    'temperature': config.get('temperature', 0.7),
                    'max_tokens': config.get('max_tokens'),
                    'api_key': api_key,  # Já definido acima com lógica automática (OPENROUTER_API_KEY)
                    'base_url': 'https://openrouter.ai/api/v1'
                }
                # Remove valores None para evitar passar kwargs com None
                openrouter_config = {k: v for k, v in openrouter_config.items() if v is not None}
                return ChatOpenAI(**openrouter_config)
            except ImportError:
                raise ImportError("langchain_openai não instalado. Instale com: pip install langchain-openai")

        else:
            supported = ['google', 'openai', 'anthropic', 'openrouter']
            raise ValueError(f"Provedor '{provider}' não suportado. Provedores disponíveis: {supported}")

# ===========================================
# ToolAwareLLM - Enhanced Version
# ===========================================

class ToolAwareLLM:
    """
    Wrapper universal que torna qualquer LLM tool-aware.

    Features:
    - Auto-executa tools quando LLM as chama
    - Retry logic com resultados de tools
    - Multiple execution modes
    - Extensível via hooks
    - Observability built-in
    """

    def __init__(
        self,
        llm_instance: Any,
        tools: Optional[List[BaseTool]] = None,
        execution_mode: ToolExecutionMode = ToolExecutionMode.AUTOMATIC,
        max_tool_iterations: int = 3,
        tool_timeout_seconds: float = 30.0,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
    ):
        self.llm = llm_instance
        self.tools = tools or []
        self.execution_mode = execution_mode
        self.max_tool_iterations = max_tool_iterations
        self.tool_timeout_seconds = tool_timeout_seconds

        # Hooks para observability
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end

        # Metrics
        self.tool_calls_count = 0
        self.tool_errors_count = 0

        # Tool registry (nome -> tool)
        self._tool_registry = {tool.name: tool for tool in self.tools}

# ===========================================
    # Core API
    # ===========================================

    def bind_tools(self, tools: List[BaseTool]) -> "ToolAwareLLM":
        """Adiciona tools dinamicamente"""
        self.tools.extend(tools)
        for tool in tools:
            self._tool_registry[tool.name] = tool
        return self

    def _convert_to_messages(self, input: Union[str, BaseMessage, List[BaseMessage]]) -> List[BaseMessage]:
        """Converte input para lista de BaseMessages"""
        if isinstance(input, str):
            # Converte string para HumanMessage
            return [HumanMessage(content=input)]
        elif isinstance(input, BaseMessage):
            # Single message, wrap in list
            return [input]
        elif isinstance(input, list):
            # Already a list of messages
            return input
        else:
            raise TypeError(f"Input must be str, BaseMessage, or list of BaseMessages, got {type(input)}")

    async def run(self, input: Union[str, BaseMessage, List[BaseMessage]], **kwargs) -> Any:
        """
        Unified run method that handles string, BaseMessage, or list of BaseMessages.

        Args:
            input: Can be a string (converted to HumanMessage), a single BaseMessage, or a list of BaseMessages
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        # Convert input to messages list
        messages = self._convert_to_messages(input)

        # If LLM supports bind_tools natively
        if hasattr(self.llm, "bind_tools") and self.tools:
            return await self._invoke_with_native_tools(messages, **kwargs)

        # Fallback: no tools or LLM doesn't support them
        return await self._invoke_llm_with_messages(messages, **kwargs)

    def __call__(self, input: Union[str, BaseMessage, List[BaseMessage]], **kwargs) -> Any:
        """Sync call wrapper"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run(input, **kwargs))

    # ===========================================
    # Tool Execution Logic
    # ===========================================

    async def _invoke_with_native_tools(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Executa LLM com suporte nativo a tools"""

        logger.info(f"🔧 Binding {len(self.tools)} tools to LLM")
        bound_llm = self.llm.bind_tools(self.tools)

        current_messages = messages
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            # Chama LLM com messages
            response = await self._invoke_llm_with_messages_bound(current_messages, bound_llm, **kwargs)

            # Verifica se tem tool calls
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # Resposta final sem tools
                return response

            # Executa tools baseado no mode
            if self.execution_mode == ToolExecutionMode.AUTOMATIC:
                tool_results = await self._execute_tools_batch(tool_calls)

                # Adiciona AIMessage com tool_calls à conversa
                current_messages.append(response)

                # Adiciona ToolMessages com resultados
                for tool_call, tool_result in zip(tool_calls, tool_results):
                    tool_call_id = tool_call.get("id")
                    from langchain_core.messages import ToolMessage
                    current_messages.append(
                        ToolMessage(
                            content=str(tool_result.result) if tool_result.success else tool_result.error,
                            tool_call_id=tool_call_id,
                            name=tool_result.tool_name
                        )
                    )

                logger.info(f"🔄 Tool iteration {iteration}/{self.max_tool_iterations}")

            elif self.execution_mode == ToolExecutionMode.MANUAL:
                # Retorna response com tool calls para execução externa
                response._tool_calls = tool_calls
                response._requires_tools = True
                return response

            elif self.execution_mode == ToolExecutionMode.HYBRID:
                # TODO: Implementar confirmação do usuário
                raise NotImplementedError("Hybrid mode not implemented yet")

        logger.warning(f"⚠️ Reached max tool iterations ({self.max_tool_iterations})")
        return response

    async def _execute_tools_batch(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolExecutionResult]:
        """Executa múltiplas tools em paralelo"""

        tasks = [
            self._execute_single_tool(tool_call)
            for tool_call in tool_calls
        ]

        return await asyncio.gather(*tasks)

    async def _execute_single_tool(
        self,
        tool_call: Dict[str, Any]
    ) -> ToolExecutionResult:
        """Executa uma tool com error handling e timeout"""

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})

        # Valida se tool existe
        if tool_name not in self._tool_registry:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.error(f"❌ {error_msg}")
            self.tool_errors_count += 1
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=error_msg
            )

        tool = self._tool_registry[tool_name]

        # Hook: on_tool_start
        if self.on_tool_start:
            await self._safe_call_hook(self.on_tool_start, tool_name, tool_args)

        import time
        start_time = time.time()

        try:
            # Executa com timeout
            result = await asyncio.wait_for(
                self._run_tool(tool, tool_args),
                timeout=self.tool_timeout_seconds
            )

            execution_time = (time.time() - start_time) * 1000

            self.tool_calls_count += 1
            logger.info(f"✅ Tool '{tool_name}' executed in {execution_time:.2f}ms")

            # Hook: on_tool_end
            if self.on_tool_end:
                await self._safe_call_hook(
                    self.on_tool_end,
                    tool_name,
                    tool_args,
                    result
                )

            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {self.tool_timeout_seconds}s"
            logger.error(f"⏱️ Tool '{tool_name}' timed out")
            self.tool_errors_count += 1

            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"❌ Tool '{tool_name}' failed: {error_msg}")
            self.tool_errors_count += 1

            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def _run_tool(self, tool: BaseTool, args: Dict[str, Any]) -> Any:
        """Wrapper para executar tool (sync ou async)"""
        if asyncio.iscoroutinefunction(tool.func):
            return await tool.func(**args)
        else:
            # Executa sync tool em thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool.run(args))

    # ===========================================
    # Utility Methods
    # ===========================================

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extrai tool calls do response (compatível com múltiplos formatos)"""

        # LangChain AIMessage com tool_calls
        if hasattr(response, "tool_calls"):
            return [
                {
                    "id": tc.get("id") or tc.get("tool_call_id") or tc.get("name"),
                    "name": tc.get("name") or tc.function.name,
                    "args": tc.get("args") or tc.function.arguments
                }
                for tc in response.tool_calls
            ] if response.tool_calls else []

        # OpenAI-style response
        if hasattr(response, "message") and hasattr(response.message, "tool_calls"):
            return [
                {
                    "id": tc.id or tc.tool_call_id or tc.function.name,
                    "name": tc.function.name,
                    "args": tc.function.arguments
                }
                for tc in response.message.tool_calls
            ] if response.message.tool_calls else []

        # Anthropic-style
        if hasattr(response, "content") and isinstance(response.content, list):
            tool_calls = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id or block.name,
                        "name": block.name,
                        "args": block.input
                    })
            return tool_calls

        return []



    async def _invoke_llm_with_messages(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Invoca LLM com lista de messages"""
        if hasattr(self.llm, "ainvoke"):
            return await self.llm.ainvoke(messages, **kwargs)
        elif hasattr(self.llm, "invoke"):
            return self.llm.invoke(messages, **kwargs)
        elif callable(self.llm):
            try:
                # Try as async call
                return await self.llm(messages, **kwargs)
            except TypeError:
                # Fallback to sync call
                return self.llm(messages, **kwargs)
        else:
            raise TypeError(f"LLM object {type(self.llm)} não suporta invocation com messages")

    async def _invoke_llm_with_messages_bound(self, messages: List[BaseMessage], bound_llm: Any, **kwargs) -> Any:
        """Invoca LLM bound com lista de messages"""
        if hasattr(bound_llm, "ainvoke"):
            return await bound_llm.ainvoke(messages, **kwargs)
        elif hasattr(bound_llm, "invoke"):
            return bound_llm.invoke(messages, **kwargs)
        elif callable(bound_llm):
            try:
                # Try as async call
                return await bound_llm(messages, **kwargs)
            except TypeError:
                # Fallback to sync call
                return bound_llm(messages, **kwargs)
        else:
            raise TypeError(f"Bound LLM object {type(bound_llm)} não suporta invocation com messages")

    async def _invoke_llm(self, llm_obj: Any, prompt: str, **kwargs) -> Any:
        """Invoca LLM (com compatibilidade universal)"""
        if hasattr(llm_obj, "ainvoke"):
            return await llm_obj.ainvoke(prompt, **kwargs)
        elif hasattr(llm_obj, "invoke"):
            # Fallback sync
            return llm_obj.invoke(prompt, **kwargs)
        elif callable(llm_obj):
            return await llm_obj(prompt, **kwargs)
        else:
            raise TypeError(f"LLM object {type(llm_obj)} não é invocável")

    async def _safe_call_hook(self, hook: Callable, *args, **kwargs):
        """Executa hook com error handling"""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args, **kwargs)
            else:
                hook(*args, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Hook execution failed: {e}")

    # ===========================================
    # Observability
    # ===========================================

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso"""
        return {
            "total_tool_calls": self.tool_calls_count,
            "total_errors": self.tool_errors_count,
            "success_rate": (
                (self.tool_calls_count - self.tool_errors_count) / self.tool_calls_count
                if self.tool_calls_count > 0
                else 0.0
            ),
            "registered_tools": list(self._tool_registry.keys()),
            "execution_mode": self.execution_mode.value,
        }

    def reset_stats(self):
        """Reseta contadores"""
        self.tool_calls_count = 0
        self.tool_errors_count = 0

    # ===========================================
    # Magic Methods
    # ===========================================

    def __repr__(self) -> str:
        return (
            f"ToolAwareLLM(tools={len(self.tools)}, "
            f"mode={self.execution_mode.value}, "
            f"calls={self.tool_calls_count})"
        )


# ===========================================
# Example Usage
# ===========================================

if __name__ == "__main__":
    print("🔧 LLMFactory - Standalone LLM Creation")
    print("=" * 50)

    # Example tools
    from langchain_core.tools import Tool

    def calculator(expression: str) -> str:
        """Avalia expressão matemática"""
        try:
            result = eval(expression)
            return f"Resultado: {result}"
        except Exception as e:
            return f"Erro: {e}"

    def search_web(query: str) -> str:
        """Busca na web (mock)"""
        return f"Resultados para '{query}': [mock data]"

    tools = [
        Tool(name="calculator", func=calculator, description="Calcula expressões"),
        Tool(name="search", func=search_web, description="Busca na web")
    ]

    print("\n✅ LLMFactory pronto para uso!")
    print("\nExemplos:")
    print("  • LLMFactory.create_llm('google:gemini-2.5-flash')  # Usa GOOGLE_API_KEY do .env")
    print("  • LLMFactory.create_llm('openai:gpt-4', config={'temperature': 0.1})  # Usa OPENAI_API_KEY do .env")
    print("  • LLMFactory.create_llm('anthropic:claude-3-haiku', tools=tools)  # Usa ANTHROPIC_API_KEY do .env")
    print("  • LLMFactory.create_llm('openrouter:meta-llama/llama-3.1-8b-instruct')  # Usa OPENROUTER_API_KEY do .env")
    print("  • Ou passe api_key explicitamente: config={'api_key': 'your_key_here'}")
    print("\n🎯 Framework só precisa se preocupar com coordenação, não com APIs!")
