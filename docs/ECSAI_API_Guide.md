# 🚀 ECSAI Framework - Guia Completo da API

**Framework ECSA (Event-Driven Component System Architecture) com LLMFactory sem dor de cabeça!**

---

## 📖 Índice

- [🚀 Introdução](#-introdução)
- [⚡ Instalação](#-instalação)
- [🔧 Configuração](#-configuração)
- [🧠 LLMFactory - Zero Dor de Cabeça](#-llmfactory---zero-dor-de-cabeça)
- [🏗️ Arquitetura Event-Driven](#️-arquitetura-event-driven)
- [🛠️ Sistema de Componentes](#️-sistema-de-componentes)
- [🎯 Middleware System](#-middleware-system)
- [🎭 Exemplos Práticos](#-exemplos-práticos)
- [🔧 Receitas Avançadas](#-receitas-avançadas)
- [🐛 Troubleshooting](#-troubleshooting)

---

## 🚀 Introdução

O **ECSAI Framework** é um framework moderno e poderoso para construção de aplicações de IA usando arquitetura **event-driven** com **componentes modulares**. Ele elimina completamente a **dor de cabeça** de trabalhar com LLMs, oferecendo:

### ✨ **Por que ECSAI?**

- **🎯 Zero Configuração**: API keys automaticamente do `.env`
- **🏗️ Arquitetura Moderna**: Event-driven com componentes desacoplados
- **🧠 LLMFactory Poderoso**: Suporte a 20+ modelos (Google, OpenAI, Anthropic, Llama, Mistral via OpenRouter)
- **⚡ Performance**: Middlewares para caching, rate limiting, circuit breaker
- **🔧 Extensível**: Fácil criar componentes customizados
- **🛡️ Robusto**: Tratamento automático de erros e fallbacks

### 🏗️ **Arquitetura Core**

```python
# 1. Componentes comunicam via Events (não diretamente)
ComponentA → Event → ComponentB → Event → ComponentC

# 2. LLMFactory cuida dos LLMs automaticamente
llm = LLMFactory.create_llm(llm_id="google:gemini-2.5-flash")
response = await LLMFactory.run(llm, "Sua pergunta")

# 3. Middlewares interceptam tudo automaticamente
LoggingMiddleware → RateLimitMiddleware → CircuitBreakerMiddleware
```

---

## ⚡ Instalação

### 📦 **Via pip (Recomendado)**

```bash
pip install ecsaai
```

### 🛠️ **Instalação Manual**

```bash
git clone https://github.com/your-repo/ecsaai.git
cd ecsaai
pip install -e .
```

### 🔍 **Verificar Instalação**

```python
import ecsaai
print(ecsaai.__version__)  # Deve mostrar versão atual
```

---

## 🔧 Configuração

### 📄 **Arquivo .env**

Crie um arquivo `.env` na raiz do seu projeto:

```env
# API Keys (automaticamente detectadas)
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=sk-or-v1-...your_openrouter_key
OPENAI_API_KEY=sk-...your_openai_key  # Opcional, cai para OpenRouter se ausente
ANTHROPIC_API_KEY=sk-ant-...your_anthropic_key
COHERE_API_KEY=your_cohere_key
HUGGINGFACE_API_TOKEN=hf_...your_huggingface_token

# Configurações opcionais
ECS_LOG_LEVEL=INFO
ECS_MAX_RETRIES=3
```

### 🎯 **Detecção Automática de Provedores**

O framework **prioriza provedores automaticamente**:

1. **OpenRouter** (se `OPENROUTER_API_KEY` presente) - Mais econômico
2. **Google Gemini** (sempre disponível) - Gratuito até certo limite
3. **OpenAI direto** (se `OPENAI_API_KEY` presente)
4. **Anthropic** (se `ANTHROPIC_API_KEY` presente)
5. **Outros** (Cohere, HuggingFace)

---

## 🧠 LLMFactory - Zero Dor de Cabeça

### 🏃 **Interface Simples**

```python
from ecsaai import LLMFactory

# Exemplo mais simples possível
llm = LLMFactory.create_llm(llm_id="google:gemini-2.5-flash")
resposta = await LLMFactory.run(llm, "Olá, como você está?")
print(resposta.content)
```

### 🔧 **Com Configuração**

```python
# Com configurações customizadas
config = {
    'temperature': 0.1,     # Mais determinístico (0.0 = sempre igual, 1.0 = criativo)
    'max_tokens': 1000,     # Máximo de tokens na resposta
    'timeout': 30,          # Timeout em segundos
}

llm = LLMFactory.create_llm(
    config=config,
    llm_id="google:gemini-2.5-flash",
    tools=tools_opcionais
)
```

### 📋 **Modelos Disponíveis**

```python
# Ver todos os modelos disponíveis
modelos = LLMFactory.list_available_models()
print(f"📊 {len(modelos)} modelos disponíveis!")

# Modelos principais
llm = LLMFactory.create_llm(llm_id="google:gemini-2.5-flash")      # Mais rápido
llm = LLMFactory.create_llm(llm_id="anthropic:claude-3-haiku")      # Mais inteligente
llm = LLMFactory.create_llm(llm_id="openai:gpt-4o-mini")            # Via OpenRouter (barato)
llm = LLMFactory.create_llm(llm_id="meta-llama:llama-3.1-70b-instruct")  # Open source
```

### 🏃 **Modos Automáticos da Interface run()**

```python
from ecsaai import LLMFactory

llm = LLMFactory.create_llm(llm_id="google:gemini-2.5-flash")

# MODO 1: String simples (automático)
resposta = await LLMFactory.run(llm, "Explique APIs REST")

# MODO 2: Lista de BaseMessage (avançado)
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Você é um professor paciente"),
    HumanMessage(content="Me explique sobre microserviços")
]

resposta = await LLMFactory.run(llm, messages)
```

### 🛠️ **Com Tools (Agentes)**

```python
from ecsaai import LLMFactory

# Definir tools
tools = [
    {
        "name": "get_weather",
        "description": "Obtém informações do clima",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
]

llm = LLMFactory.create_llm(
    llm_id="google:gemini-2.5-flash",
    tools=tools
)

resposta = await LLMFactory.run(llm, "Qual o clima em São Paulo?")
# LLM automaticamente chama tools quando necessário!
```

### 🌐 **Capacidades Multimodais**

```python
from ecsaai import LLMFactory
from langchain_core.messages import HumanMessage

# Para imagens
llm = LLMFactory.create_llm(llm_id="google:gemini-pro-vision")

messages = [
    HumanMessage(content=[
        {"type": "text", "text": "O que você vê nesta imagem?"},
        {"type": "image_url", "image_url": "https://exemplo.com/imagem.jpg"}
    ])
]

resposta = await LLMFactory.run(llm, messages)
```

---

## 🏗️ Arquitetura Event-Driven

### 🎯 **O que é Event-Driven?**

Imagine componentes como **caixas postais**: eles mandam mensagens (events) para outros componentes lerem, sem conhecer uns aos outros diretamente.

```
Componente A → Event "dados_processados" → Componente B → Event "resposta_pronta" → Output
```

### 🏗️ **Estrutura Básica**

```python
from ecsaai import Agent, Component, EventBus, BaseEvent

# 1. Criar Agent (contêiner de componentes)
agent = Agent("meu_agent")

# 2. Adicionar componentes
await agent.add_component(MeuComponent, config=minha_config)

# 3. Inicializar tudo
await agent.init_all()

# 4. Mandar primeiro evento
evento = InputEvent(content="Olá mundo", session_id="sessao_1")
resultados = await agent.send_event(evento)
```

### 📬 **Sistema de Eventos Built-in**

```python
from ecsaai.Events import (
    InputEvent,     # Entrada do usuário
    ContextEvent,   # Contexto formatado
    LLMResponseEvent, # Resposta do LLM
    OutputEvent,    # Saída final
    ErrorEvent      # Erro ocorrido
)

# Criar evento customizado
class MeuEvento(BaseEvent):
    type: str = "meu_evento"
    dados_importantes: str
```

---

## 🛠️ Sistema de Componentes

### 🎯 **Componente Básico**

```python
from ecsaai import Component, BaseEvent
from typing import List

class MeuComponent(Component):
    name: str = "MeuComponent"

    # Quais eventos recebe e emite (type-safe!)
    receives = [InputEvent]      # Recebe InputEvent
    emits = [OutputEvent]       # Emite OutputEvent

    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        # Lógica do seu componente
        if isinstance(event, InputEvent):
            processado = event.content.upper()

            return [OutputEvent(
                sender=self.name,
                target=f"{self.agent_id}:OutputComponent",
                content=processado,
                session_id=event.session_id
            )]

        return []
```

### 🧩 **Componentes Disponíveis**

#### **LLMComponent** - Para uso direto de LLMs

```python
from ecsaai.Components.LLMComponent import LLMComponent, create_fast_llm_config

# Adicionar ao agent
await agent.add_component(
    LLMComponent,
    config=create_fast_llm_config()  # Ou configurações customizadas
)
```

#### **ContextComponent** - Gerenciamento de contexto conversacional

```python
from ecsaai.Components.ContextComponent import ContextComponent, create_chatbot_config

# Com persistência SQLite
await agent.add_component(
    ContextComponent,
    config=create_chatbot_config()
)

# Ou banco customizado
config = ContextConfig(connection_string="postgresql://...")
```

#### **OutputComponent** - Tratamento de saídas

```python
from ecsaai.Components.OutputComponent import OutputComponent

await agent.add_component(
    OutputComponent,
    config=OutputConfig(output_format="json")
)
```

### 🔗 **Comunicação Inter-Componente**

```python
class ComponentComunica(Component):
    async def handle_event(self, event):
        # Chamar outro componente diretamente
        context_response = await self.call_component(
            "ContextComponent",
            GetContextRequest(session_id=event.session_id)
        )

        # Processar resposta...
        return [LLMResponseEvent(content=processada)]
```

---

## 🎯 Middleware System

### 🛡️ **O que são Middlewares?**

Middlewares são **interceptadores** que processam eventos automaticamente antes/depois dos componentes:

```
Evento Entrada → Middleware 1 → Middleware 2 → Component → Middleware 2 → Middleware 1 → Evento Saída
```

### 🔧 **Middlewares Built-in**

#### **LoggingMiddleware** - Log automático

```python
from ecsaai.middleware import LoggingMiddleware

component.middlewares = [LoggingMiddleware()]
# Loga todos os eventos automaticamente
```

#### **RateLimitMiddleware** - Controle de taxa

```python
from ecsaai.middleware import RateLimitMiddleware

component.middlewares = [
    RateLimitMiddleware(requests_per_minute=60)
]
```

#### **MetricsMiddleware** - Métricas e observabilidade

```python
from ecsaai.middleware import MetricsMiddleware

component.middlewares = [MetricsMiddleware()]
# Métricas disponíveis em component.metrics
```

#### **CircuitBreakerMiddleware** - Tolerância a falhas

```python
from ecsaai.middleware import CircuitBreakerMiddleware

component.middlewares = [
    CircuitBreakerMiddleware(
        failure_threshold=5,     # Após 5 falhas, abre o circuito
        recovery_timeout=60      # Reabre após 60 segundos
    )
]
```

#### **ValidationMiddleware** - Validação automática

```python
from ecsaai.middleware import ValidationMiddleware

component.middlewares = [ValidationMiddleware()]
# Valida automaticamente contratos de entrada/saída
```

### 🛠️ **Criando Middleware Customizado**

```python
from ecsaai import ComponentMiddleware

class MeuMiddleware(ComponentMiddleware):
    async def before_handle(self, event: BaseEvent) -> Optional[BaseEvent]:
        # Antes do componente processar
        print(f"📥 Processando evento: {event.type}")

        # Modificar evento se necessário
        if hasattr(event, 'content'):
            event.content = event.content.strip()

        return event  # Retornar None para CANCELAR o evento

    async def after_handle(self, event: BaseEvent, result: List[BaseEvent]) -> List[BaseEvent]:
        # Depois do componente processar
        print(f"📤 Gerados {len(result)} eventos")

        # Modificar resultados se necessário
        for r in result:
            r.meta['processed_by'] = 'MeuMiddleware'

        return result
```

---

## 🎭 Exemplos Práticos

### 💬 **Chatbot Simples**

```python
import asyncio
from ecsaai import Agent
from ecsaai.Components.ContextComponent import ContextComponent, create_chatbot_config
from ecsaai.Components.LLMComponent import LLMComponent, create_fast_llm_config
from ecsaai.Components.OutputComponent import OutputComponent
from ecsaai.Events import InputEvent

async def criar_chatbot():
    # 1. Configurar agent
    agent = Agent("chatbot_simples")

    # 2. Adicionar componentes
    await agent.add_component(ContextComponent, config=create_chatbot_config())
    await agent.add_component(LLMComponent, config=create_fast_llm_config())
    await agent.add_component(OutputComponent)

    # 3. Inicializar
    await agent.init_all()

    return agent

async def conversar_com_chatbot():
    agent = await criar_chatbot()

    # Conversa
    mensagens = [
        "Olá, como você está?",
        "Me conte uma curiosidade sobre IA",
        "Obrigado pela conversa!"
    ]

    for msg in mensagens:
        print(f"👤 Você: {msg}")

        # Envia input
        evento = InputEvent(
            sender="user",
            target="chatbot_simples:ContextComponent",
            content=msg,
            session_id="conversa_1"
        )

        # Processa pipeline completa
        resultados = await agent.send_event(evento)

        print(f"🤖 Bot: {len(resultados)} eventos processados")

    await agent.shutdown_all()

# Executar
asyncio.run(conversar_com_chatbot())
```

### 🤖 **Agent com Ferramentas**

```python
from ecsaai import Component
from ecsaai.Events import InputEvent, LLMResponseEvent
from typing import List

class AgentComFerramentas(Component):
    name = "AgentFerramentas"

    receives = [InputEvent]
    emits = [LLMResponseEvent]

    def __init__(self, config=None):
        super().__init__(config)
        # Ferramentas disponíveis
        self.tools = {
            'calcular': self._tool_calcular,
            'busca_web': self._tool_busca_web
        }

    async def handle_event(self, event: InputEvent) -> List[LLMResponseEvent]:
        # Análise básica da intenção
        prompt = f"""
        Analise esta mensagem do usuário e determine se precisa de uma ferramenta:

        Mensagem: {event.content}

        Ferramentas disponíveis:
        - calcular: para operações matemáticas
        - busca_web: para procurar informações na web

        Responda apenas com o nome da ferramenta ou "nenhuma".
        """

        # Usa LLMFactory para análise
        llm = await self._get_llm()
        analise = await LLMFactory.run(llm, prompt)

        tool_name = analise.content.strip().lower()

        if tool_name in self.tools:
            # Executar ferramenta
            resultado = await self.tools[tool_name](event.content)
            resposta = f"Resultado da ferramenta '{tool_name}': {resultado}"
        else:
            resposta = "Posso ajudar com cálculos ou busca na web. O que você precisa?"

        return [LLMResponseEvent(
            sender=self.name,
            target=f"{self.agent_id}:OutputComponent",
            response=resposta,
            session_id=event.session_id
        )]

    async def _get_llm(self):
        from ecsaai import LLMFactory
        return LLMFactory.create_llm(llm_id="openai:gpt-4o-mini")

    async def _tool_calcular(self, query: str) -> str:
        # Implementação simples
        try:
            # Extrair expressão matemática
            import re
            match = re.search(r'(\d[^\w]*\d)', query)
            if match:
                # Usar eval seguro (apenas para demo!)
                result = eval(match.group(1))
                return str(result)
        except:
            return "Não consegui calcular isso."
        return "Não encontrei nada para calcular."

    async def _tool_busca_web(self, query: str) -> str:
        # Implementação mockada
        return f"Resultados mockados para busca: '{query}'"
```

### 🗄️ **Sistema com Persistência Avançada**

```python
from ecsaai.Components.ContextComponent import ContextComponent, ContextConfig

# Configuração avançada
config = ContextConfig(
    connection_string="postgresql://user:pass@localhost:5432/chatdb",
    initial_system_prompt="""
    Você é um assistente especializado em desenvolvimento de software.
    Sempre forneça código bem documentado e explique os conceitos.
    """,
    max_history_messages=100,      # Lembra mais mensagens
    auto_cleanup_days=90          # Limpa mensagens antigas
)

component = ContextComponent(config=config)
```

---

## 🔧 Receitas Avançadas

### 🔄 **Fallback Automático Entre Provedores**

```python
class FallbackLLMComponent(Component):
    """Componente que testa múltiplos provedores automaticamente"""

    async def get_working_llm(self):
        from ecsaai import LLMFactory

        # Ordem de tentativa
        providers_to_try = [
            "openai:gpt-4o-mini",      # Barato via OpenRouter
            "google:gemini-2.5-flash", # Gratuito até limite
            "anthropic:claude-3-haiku", # Mais inteligente
            "openai:gpt-3.5-turbo",    # Sempre funciona
        ]

        for provider in providers_to_try:
            try:
                llm = LLMFactory.create_llm(llm_id=provider)
                # Testa conexão
                test = await LLMFactory.run(llm, "OK", config={'max_tokens': 10})
                if test.content.strip():
                    return llm
            except Exception as e:
                print(f"❌ {provider} falhou: {str(e)[:50]}")
                continue

        raise RuntimeError("Nenhum provedor LLM funcionando!")
```

### 📊 **Dashboard de Métricas**

```python
async def mostrar_metricas_agente(agent):
    print("📊 MÉTRICAS DO AGENTE")
    print("=" * 40)

    for name, component in agent.components.items():
        if hasattr(component, 'metrics'):
            m = component.metrics
            print(f"\n{name}:")
            print(f"  📥 Eventos recebidos: {m.events_received}")
            print(f"  📤 Eventos emitidos: {m.events_emitted}")
            print(f"  ❌ Erros: {m.errors}")
            print(f"  ⏱️ Latência média: {m.avg_latency_ms:.2f}ms")
            if m.last_error:
                print(f"  🚨 Último erro: {m.last_error[:50]}...")
```

### 🎛️ **Configuração por Ambiente**

```python
import os
from ecsaai.Components.LLMComponent import LLMConfig

def get_llm_config_por_ambiente():
    env = os.getenv('ENVIRONMENT', 'development')

    configs = {
        'development': LLMConfig(  # Mais barato para dev
            llm_id="openai:gpt-4o-mini",
            temperature=0.7,
            max_tokens=500
        ),
        'production': LLMConfig(   # Melhor qualidade para prod
            llm_id="openai:gpt-4",
            temperature=0.1,
            max_tokens=2000
        ),
        'testing': LLMConfig(      # Mock para testes
            llm_id="google:gemini-2.5-flash",
            temperature=0.0      # Determinístico
        )
    }

    return configs.get(env, configs['development'])
```

### 🔄 **Hot Reload de Configuração**

```python
from ecsaai.middleware import ComponentMiddleware

class HotReloadMiddleware(ComponentMiddleware):
    """Middleware que recarrega configuração automaticamente"""

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.last_modified = 0

    async def before_handle(self, event):
        # Verifica se arquivo mudou
        import os
        current_mtime = os.path.getmtime(self.config_file)

        if current_mtime > self.last_modified:
            print("🔄 Configuração mudada, recarregando...")

            # Recarregar config (component.config.reload_from_file() etc)
            self.last_modified = current_mtime

        return event
```

---

## 🐛 Troubleshooting

### ❌ **Erro: API key não encontrada**

```python
# Verificar variáveis de ambiente
import os
print("GOOGLE_API_KEY:", bool(os.getenv('GOOGLE_API_KEY')))
print("OPENROUTER_API_KEY:", bool(os.getenv('OPENROUTER_API_KEY')))

# Verificar arquivo .env existe
import os.path
print(".env existe:", os.path.exists('.env'))
```

### ❌ **Erro: Provider não suportado**

```python
# Ver provedores disponíveis
from ecsaai import LLMFactory
print("Disponíveis:", LLMFactory.list_available_models())
```

### ❌ **Erro: Timeout**

```python
# Aumentar timeout
llm = LLMFactory.create_llm(
    llm_id="google:gemini-2.5-flash",
    config={'timeout': 120}  # 2 minutos
)
```

### ❌ **Erro: Rate limit**

```python
# Adicionar middleware de rate limit
from ecsaai.middleware import RateLimitMiddleware

component.middlewares = [
    RateLimitMiddleware(requests_per_minute=30)  # 30 por minuto
]
```

### ❌ **Erro: Component já existe**

```python
# Verificar nomes únicos
async def adicionar_com_nome_unico(agent, component_class, config):
    nome_unico = f"{component_class.__name__}_{id(config)}"
    component_class.name = nome_unico
    await agent.add_component(component_class, config)
```

---

## 🎯 Próximos Passos

1. **📚 Leia a documentação completa** em [ecsaai-docs.com](https://ecsaai-docs.com)
2. **🎮 Experimente os tutoriais** em `/tutorial_comprehensive.md`
3. **🛠️ Crie componentes customizados** seguindo o padrão
4. **📊 Monitore métricas** com middlewares
5. **🚀 Faça deploy** usando os presets de produção

---

**🎉 Parabéns!** Você agora domina o ECSAI Framework. Com ele, você pode construir aplicações de IA robustas, escaláveis e fáceis de manter. Boa sorte com seus projetos! 🚀
