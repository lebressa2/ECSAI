# ===========================================
# Utility Libraries
# ===========================================

"""
Basic utility libraries for the framework
"""

import logging
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type

# ===========================================
# Logger Utility
# ===========================================

class Logger:
    """Simple logging utility"""

    @staticmethod
    def get_logger(name: str):
        """Get a configured logger"""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# ===========================================
# LLM Factory - Zero Pain, Automatic, Multimodal
# ===========================================

class LLMFactory:
    """
    LLM Factory that completely eliminates pain:
    - Automatic API key loading from .env
    - Zero configuration needed
    - Multimodal capabilities
    - Automatic provider detection
    - Simple interface: create_llm() + run()
    """

    @staticmethod
    def _get_api_keys() -> dict:
        """Carrega chaves de API das variáveis de ambiente."""
        # Detecta automaticamente se usar OpenRouter
        openai_key = os.getenv('OPENAI_API_KEY')
        openrouter_key = os.getenv('OPENROUTER_API_KEY')

        # Se tem OPENROUTER_API_KEY específico, prioriza
        if openrouter_key:
            openai_key = openrouter_key

        return {
            'openai': openai_key,
            'openrouter': openrouter_key,  # Campo separado para OpenRouter
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY'),
            'cohere': os.getenv('COHERE_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_TOKEN'),
        }

    @staticmethod
    def _validate_config(provider: str, api_keys: dict) -> None:
        """Valida se as configurações necessárias estão presentes."""
        if provider not in ['huggingface', 'mock'] and not api_keys.get(provider):
            raise ValueError(f"API key não encontrada para provider: {provider}")

    @staticmethod
    def create_llm(
        config: Optional[dict] = None,
        llm_id: str = "google:gemini-2.5-flash",
        tools: Optional[list] = None,
        **kwargs
    ):
        """
        Interface simples e sem dor de cabeça para criar LLMs.

        Args:
            config: Configuração adicional (opcional)
            llm_id: ID do modelo no formato "provider:model"
            tools: Tools para o LLM (opcional)

        Returns:
            LLM configurado e pronto para uso
        """
        from langchain_core.language_models import BaseLanguageModel
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_cohere import ChatCohere
        from langchain_community.llms import HuggingFaceHub

        if not llm_id or ':' not in llm_id:
            raise ValueError("LLM ID deve estar no formato 'provider:model' (ex: 'openai:gpt-4')")

        provider, model = llm_id.split(':', 1)
        config = config or {}
        api_keys = LLMFactory._get_api_keys()

        LLMFactory._validate_config(provider, api_keys)

        # Configurações comuns
        common_config = {
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens'),
            'timeout': config.get('timeout', 60),
            'model_kwargs': config.get('model_kwargs', {}),
        }

        # Adiciona tools se fornecidos
        if tools:
            common_config['tools'] = tools

        # Merge com kwargs customizados
        common_config.update(kwargs)

        # Cria instância baseada no provider
        if provider == 'openai':
            # Detecta automaticamente se usar OpenRouter
            openrouter_key = api_keys.get('openrouter')
            openai_config = {
                'model': model,
                'api_key': api_keys['openai'],
                **common_config
            }

            # Se está usando OPENROUTER_API_KEY, configura URL custom
            if openrouter_key:
                openai_config['base_url'] = "https://openrouter.ai/api/v1"

            return ChatOpenAI(**openai_config)

        elif provider == 'anthropic':
            return ChatAnthropic(
                model=model,
                api_key=api_keys['anthropic'],
                **common_config
            )

        elif provider == 'google':
            return ChatGoogleGenerativeAI(
                model=model,
                api_key=api_keys['google'],
                **common_config
            )

        elif provider == 'cohere':
            return ChatCohere(
                model=model,
                api_key=api_keys['cohere'],
                **common_config
            )

        elif provider == 'huggingface':
            return HuggingFaceHub(
                repo_id=model,
                huggingfacehub_api_token=api_keys['huggingface'],
                **common_config
            )

        else:
            raise ValueError(f"Provider não suportado: {provider}")

    @staticmethod
    async def run(
        llm,
        input_data,
        config: Optional[dict] = None,
        **kwargs
    ):
        """
        Interface global run que aceita string ou lista de BaseMessage.
        Modos automáticos ativados!

        Args:
            llm: Instância do LLM criada pelo create_llm
            input_data: String ou lista de BaseMessage
            config: Configuração adicional (opcional)

        Returns:
            Resposta do LLM
        """
        from langchain_core.messages import BaseMessage, HumanMessage

        config = config or {}

        # Modo automático: detecta tipo de input
        if isinstance(input_data, str):
            # Modo texto simples - cria HumanMessage automaticamente
            messages = [HumanMessage(content=input_data)]
        elif isinstance(input_data, list):
            # Modo avançado - lista de messages já preparada
            messages = input_data
            # Validação básica
            if not all(isinstance(msg, BaseMessage) for msg in messages):
                raise ValueError("Todos os itens da lista devem ser BaseMessage")
        else:
            raise ValueError("Input deve ser string ou lista de BaseMessage")

        # Configurações multimodais automáticas
        invoke_config = {
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 1000),
        }
        invoke_config.update(kwargs)

        # Executa com tratamento multimodal
        try:
            response = await llm.ainvoke(messages, config=invoke_config)
            return response
        except Exception as e:
            # Fallback para modo compatibilidade se multimodal falhar
            if "multimodal" in str(e).lower():
                logger.warning(f"Multimodal falhou, tentando modo texto: {e}")
                # Converte para texto simples
                text_content = "\n".join([
                    msg.content if isinstance(msg.content, str)
                    else str(msg.content) for msg in messages
                ])
                response = await llm.ainvoke(text_content, config=invoke_config)
                return response
            else:
                raise e

    @staticmethod
    def list_available_models() -> list:
        """
        Retorna lista de modelos disponíveis para uso imediato.
        Inclui modelos do OpenRouter quando OPENROUTER_API_KEY está presente.
        """
        base_models = [
            # Google (mais estáveis e gratuitos inicialmente)
            'google:gemini-2.5-flash',
            'google:gemini-pro',
            'google:gemini-pro-vision',

            # OpenAI (mais caros mas poderosos)
            'openai:gpt-4',
            'openai:gpt-4-turbo',
            'openai:gpt-3.5-turbo',

            # Anthropic (mais analíticos)
            'anthropic:claude-3-opus',
            'anthropic:claude-3-sonnet',
            'anthropic:claude-3-haiku',

            # Cohere (bons para embeddings)
            'cohere:command',
            'cohere:command-light',

            # HuggingFace (open source)
            'huggingface:microsoft/DialoGPT-medium',
            'huggingface:google/flan-t5-base',
        ]

        # Adiciona modelos do OpenRouter se API key estiver presente
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            openrouter_models = [
                # Modelos via OpenRouter (mais acessíveis)
                'openai:gpt-4o',  # Nova versão via OpenRouter
                'openai:gpt-4o-mini',  # Mais barato via OpenRouter
                'anthropic:claude-3.5-sonnet',  # Via OpenRouter
                'anthropic:claude-3-haiku',  # Via OpenRouter
                'google:gemini-pro-1.5',  # Via OpenRouter
                'meta-llama:llama-3.1-405b-instruct',  # Llama via OpenRouter
                'meta-llama:llama-3.1-70b-instruct',  # Llama menor via OpenRouter
                'mistralai:mistral-7b-instruct',  # Mistral via OpenRouter
            ]
            base_models.extend(openrouter_models)

        return base_models

    @staticmethod
    def get_model_info(llm_id: str) -> dict:
        """
        Retorna informações sobre um modelo específico.
        Útil para auto-configuração.
        """
        model_db = {
            'google:gemini-2.5-flash': {
                'provider': 'google',
                'multimodal': True,
                'max_tokens': 8192,
                'recommended_temp': 0.7,
                'strengths': ['multimodal', 'fast', 'reliable']
            },
            'openai:gpt-4': {
                'provider': 'openai',
                'multimodal': True,
                'max_tokens': 8192,
                'recommended_temp': 0.7,
                'strengths': ['reasoning', 'coding', 'creative']
            },
            'anthropic:claude-3-opus': {
                'provider': 'anthropic',
                'multimodal': True,
                'max_tokens': 4096,
                'recommended_temp': 0.7,
                'strengths': ['analysis', 'long_context', 'safety']
            }
        }

        return model_db.get(llm_id, {
            'provider': llm_id.split(':')[0] if ':' in llm_id else 'unknown',
            'multimodal': False,
            'max_tokens': 4096,
            'recommended_temp': 0.7,
            'strengths': ['general']
        })

class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, model: str = "default", temperature: float = 0.7,
                 max_tokens: int = 1000, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response"""
        pass

    @abstractmethod
    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate with conversation context"""
        pass

    async def health_check(self) -> bool:
        """Basic health check"""
        try:
            test_response = await self.generate("Say 'OK' if you can read this.")
            return "OK" in test_response.upper()
        except:
            return False

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing and demonstrations"""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response with some intelligence"""
        await asyncio.sleep(0.1)  # Simulate API call delay

        prompt_lower = prompt.lower()

        # Simple pattern matching for demo responses
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! I'm a mock AI assistant. How can I help you today?"

        elif "analyze" in prompt_lower and "sentiment" in prompt_lower:
            return "Based on the text analysis, the sentiment appears to be neutral-positive with a confidence score of 0.75."

        elif "explain" in prompt_lower and "framework" in prompt_lower:
            return "The ECSAI framework is a modular, event-driven architecture for building AI agents with components that communicate via events."

        elif "benefits" in prompt_lower:
            return "Key benefits include: modularity, event-driven communication, middleware support, observability, and scalability."

        else:
            # Fallback response
            return f"I understand you're asking about: '{prompt[:50]}...'. This is a mock response from the {self.model} model. In a real implementation, this would be generated by an actual LLM API."

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate with conversation context"""
        await asyncio.sleep(0.1)

        # Simple context-aware responses
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if "remember" in last_user_msg.lower():
            return "Yes, I can maintain conversation context and remember previous messages in the conversation."

        return await self.generate(last_user_msg)

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider (placeholder - would integrate with openai library)"""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        # Placeholder - would integrate with actual OpenAI client
        # import openai
        # client = openai.AsyncClient(api_key=self.api_key)
        # response = await client.chat.completions.create(...)

        raise NotImplementedError("OpenAI integration requires 'openai' library")

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate with conversation context"""
        raise NotImplementedError("OpenAI integration requires 'openai' library")

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider (placeholder - would integrate with anthropic library)"""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API"""
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        # Placeholder - would integrate with actual Anthropic client
        # import anthropic
        # client = anthropic.AsyncAnthropic(api_key=self.api_key)
        # response = await client.messages.create(...)

        raise NotImplementedError("Anthropic integration requires 'anthropic' library")

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate with conversation context"""
        raise NotImplementedError("Anthropic integration requires 'anthropic' library")

# ===========================================
# Utils
# ===========================================

class Utils:
    """General utility functions"""

    @staticmethod
    def get_env_var(key: str, default: Optional[str] = None) -> str:
        """Get environment variable with fallback"""
        return os.getenv(key, default or "")

    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple dictionaries"""
        result = {}
        for d in dicts:
            result.update(d)
        return result

# Export utilities
logger = Logger.get_logger("ecsaai")
factory = LLMFactory()

__all__ = [
    'Logger', 'LLMFactory', 'BaseLLMProvider', 'MockLLMProvider',
    'OpenAIProvider', 'AnthropicProvider', 'Utils', 'logger', 'factory'
]
