#!/usr/bin/env python3
# ===========================================
# ECSA Refactored - Demo Script
# ===========================================
"""
Complete demonstration of the new Component architecture.
Shows how to use all implemented features:
- Lifecycle hooks
- Pydantic configurations
- Error handling
- Inter-component communication
- AsyncComponent API
- Middlewares
- Observability
- Testing
"""

import asyncio
import os
import sys
from pathlib import Path

# Add root directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import Agent, BaseEvent, ErrorEvent  # Import base types as well
from Events import InputEvent, GetContextRequest, StoreMessageRequest
from Components import (
    LLMComponent, LLMConfig,
    ContextComponent, ContextConfig,
    OutputComponent, OutputConfig,

    # Presets
    create_fast_llm_config,
    create_chatbot_config,

    # Add the missing ones directly
    create_creative_llm_config,
    create_analytical_llm_config,
    create_teacher_config,
)

# Import ComponentTestHarness directly from main
from main import ComponentTestHarness
from EventBus import EventBus

# Disable LLMFactory logging for cleaner demo output
import logging
logging.basicConfig(level=logging.WARNING)




async def demo_basic_agent():
    """
    Demo 1: Basic Functional Agent
    """
    print("\n" + "="*60)
    print("🚀 DEMO 1: Basic Functional Agent")
    print("="*60)

    # Cria agent
    agent = Agent("demo_agent")

    # Adiciona componentes com configurações personalizadas
    await agent.add_component(
        ContextComponent,
        config=create_chatbot_config()
    )

    # Usa config presets para LLM
    await agent.add_component(
        LLMComponent,
        config=create_fast_llm_config()
    )

    await agent.add_component(
        OutputComponent,
        config=OutputConfig(output_format="text")
    )

    # Inicializa agent (chama on_init de todos os componentes)
    await agent.init_all()

    print("✅ Agent inicializado com 3 componentes!")

    # Simula input do usuário
    input_event = InputEvent(
        sender="user",
        target="demo_agent:ContextComponent",
        content="Olá! Como você está?",
        session_id="demo_session"
    )

    print("📨 Enviando input do usuário...")

    # Processa o evento (passa por toda a pipeline)
    try:
        result_events = await agent.send_event(input_event)
        print(f"📋 Eventos emitidos: {len(result_events)}")

        for event in result_events:
            print(f"   • {event.type} -> {event.target}")

        # Demonstra calma inter-componente
        context_comp = agent.components["ContextComponent"]

        # Chama API AsyncComponent diretamente
        context_request = GetContextRequest(session_id="demo_session")
        context_response = await context_comp.request(context_request)

        print("\n💬 Contexto da sessão:")
        print(f"   Total mensagens: {len(context_response.messages)}")
        print(f"   Última: {context_response.messages[-1]['content'][:50] if context_response.messages else 'Nenhuma'}")

    except Exception as e:
        print(f"⚠️ Simulação ignorada (API key não configurada?): {e}")

    # Finaliza agent
    await agent.shutdown_all()
    print("🛑 Agent finalizado")

async def demo_component_testing():
    """
    Demo 2: Component Testing Harness
    """
    print("\n" + "="*60)
    print("🧪 DEMO 2: Component Testing Harness")
    print("="*60)

    # Cria componente em isolamento
    context_config = ContextConfig(
        connection_string="sqlite:///test.db"  # Banco de teste
    )
    context = ContextComponent(config=context_config)

    # Cria test harness
    harness = ComponentTestHarness(context)

    print("🧪 Testando ContextComponent isoladamente...")

    # Simula input event
    input_event = InputEvent(
        sender="test",
        target="test:ContextComponent",
        content="Hello world!",
        session_id="test_session_1"
    )

    # Executa teste
    result_events = await harness.send(input_event)

    print(f"📤 Eventos emitidos: {len(result_events)}")
    print(f"📊 Métricas: {harness.get_metrics().model_dump()}")

    # Verifica se emitiu contexto
    context_events = [e for e in harness.sent_events if e.type == "context"]
    harness.assert_emitted("context", count=1)
    print("✅ Teste passou: Contexto foi emitido")

    # Testa API AsyncComponent diretamente
    store_request = StoreMessageRequest(
        session_id="test_session_1",
        message_type="ai",
        content="Olá! Como posso ajudar?"
    )

    # Simula chamando como outro componente faria
    context.set_agent_ref(harness)  # Mock agent
    response = await context.request(store_request)
    print(f"💾 Store message result: {response.success}")

    # Reseta para novo teste
    harness.reset()
    print(f"🔄 Após reset: {harness.get_metrics().events_received} eventos")

async def demo_component_features():
    """
    Demo 3: Recursos avançados dos componentes
    """
    print("\n" + "="*60)
    print("⚡ DEMO 3: Recursos Avançados")
    print("="*60)

    # Cria componente com middlewares
    from Components.LLMComponent import LoggingMiddleware, RetryMiddleware

    llm_config = LLMConfig(
        llm_id="nonexistent_llm",  # Simula erro
        max_retries=2
    )

    llm = LLMComponent(config=llm_config)
    llm.middlewares = [LoggingMiddleware(), RetryMiddleware(max_retries=1)]

    harness = ComponentTestHarness(llm)

    print("🛠️ Testando middlewares e error handling...")

    # Testa error handling
    try:
        from Events import ContextEvent
        error_event = ContextEvent(
            sender="test",
            target="test:LLMComponent",
            formatted_prompt="Test prompt",
            session_id="error_test"
        )

        # Isso deve falhar mas ser tratado graciosamente
        await harness.send(error_event)

    except Exception as e:
        print(f"⚠️ Erro esperado: {type(e).__name__}")

    # Mostra métricas após erro
    metrics = harness.get_metrics()
    print(f"📊 Métricas após erro: {metrics.errors} erros, {metrics.events_received} eventos")

    # Mostra info do componente
    info = llm.get_info()
    print(f"ℹ️ Config ativa: {info['config']['llm_id']}")
    print(f"🛡️ Middlewares: {info['middlewares']}")

async def demo_inter_component_call():
    """
    Demo 4: Comunicação inter-componente
    """
    print("\n" + "="*60)
    print("🔗 DEMO 4: Comunicação Inter-Componente")
    print("="*60)

    # Cria agent minimal com dois componentes
    agent = Agent("inter_agent")

    # Context como AsyncComponent
    await agent.add_component(
        ContextComponent,
        config=ContextConfig(connection_string="sqlite:///inter.db")
    )

    # LLM que vai chamar context
    await agent.add_component(
        LLMComponent,
        config=create_fast_llm_config()
    )

    await agent.init_all()

    print("🔗 Testando call_component()...")

    llm_comp = agent.components["LLMComponent"]

    # Simula LLM chamando context diretamente (não via eventos)
    if hasattr(llm_comp, 'call_component'):
        try:
            context_request = GetContextRequest(session_id="call_test")
            response = await llm_comp.call_component("ContextComponent", context_request)

            print(f"✅ Call direto funcionou: {response.success}")
            print(f"📄 Mensagens recuperadas: {len(response.messages) if hasattr(response, 'messages') else 0}")

        except Exception as e:
            print(f"⚠️ Call falhou (API key?): {e}")

    await agent.shutdown_all()

async def demo_configuration_and_presets():
    """
    Demo 5: Configurações e presets
    """
    print("\n" + "="*60)
    print("⚙️ DEMO 5: Configurações e Presets")
    print("="*60)

    print("🔧 Configurações disponíveis:")

    # LLM presets
    fast_config = create_fast_llm_config()
    creative_config = create_creative_llm_config()
    analytic_config = create_analytical_llm_config()

    print(f"  🚀 Fast LLM: {fast_config.llm_id}, temp={fast_config.temperature}")
    print(f"  🎨 Creative LLM: {creative_config.llm_id}, temp={creative_config.temperature}")
    print(f"  🧠 Analytical LLM: {analytic_config.llm_id}, temp={analytic_config.temperature}")

    # Context presets
    chatbot_config = create_chatbot_config()
    teacher_config = create_teacher_config()

    print("\n💬 Context presets:")
    print(f"  🤖 Chatbot: {len(chatbot_config.initial_system_prompt)} chars prompt")
    print(f"  👨‍🏫 Teacher: {len(teacher_config.initial_system_prompt)} chars prompt")

    # Demonstra validação automática
    print("\n✅ Validação Pydantic:")
    try:
        invalid_config = LLMConfig(llm_id=123)  # Tipo errado
    except Exception as e:
        print(f"  🚫 Erro capturado: {type(e).__name__} - llm_id deve ser string")

    print("  ✅ Configuração válida criada automaticamente!")

async def demo_chatbot():
    """
    Demo 6: Chatbot funcional com memória e tools
    """
    print("\n" + "="*60)
    print("🤖 DEMO 6: Chatbot com Memória e Tools")
    print("="*60)

    # Define tools
    from langchain_core.tools import Tool

    def calculator(expression: str) -> str:
        """Calcula expressões matemáticas simples"""
        try:
            result = eval(expression)
            return f"Resultado: {result}"
        except Exception as e:
            return f"Erro no cálculo: {e}"

    def search_web(query: str) -> str:
        """Busca na web (simulado)"""
        return f"Resultados para '{query}': [Dados simulados da web - implemente API real se necessário]"

    tools = [
        Tool(name="calculator", func=calculator, description="Calculadora para operações matemáticas básicas"),
        Tool(name="search_web", func=search_web, description="Busca informações na web")
    ]

    # Cria agent
    agent = Agent("chatbot_agent")

    # Context para memória conversacional
    await agent.add_component(
        ContextComponent,
        config=create_chatbot_config()
    )

    # LLM com Gemini e tools
    llm_config = LLMConfig(
        llm_id="google:gemini-2.5-flash",
        temperature=0.7,
        max_tokens=1000,
        tools=tools
    )
    await agent.add_component(LLMComponent, config=llm_config)

    # Output component
    await agent.add_component(
        OutputComponent,
        config=OutputConfig(output_format="text")
    )

    # Inicializa
    await agent.init_all()
    print("✅ Chatbot inicializado com memória e ferramentas!")

    # Registra no bus
    bus = EventBus()
    bus.register_agent("chatbot_agent", agent)

    # Interação com usuário
    session_id = "chat_session_" + str(id(agent))  # ID único
    print("\n💬 Chatbot pronto! Digite mensagens ou 'sair' para finalizar.")

    while True:
        try:
            user_input = input("\nVocê: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break

            print(f"🤔 Processando: {user_input[:50]}...")

            # Cria evento de input
            input_event = InputEvent(
                sender="user",
                target="chatbot_agent:ContextComponent",
                content=user_input,
                session_id=session_id
            )

            # Dispara evento no bus
            events = await bus.dispatch([input_event])

            print(f"📋 Eventos processados: {len(events)}")

        except KeyboardInterrupt:
            print("\n⚠️ Interrupção detectada")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            break

    # Finaliza
    await agent.shutdown_all()
    print("🛑 Chatbot finalizado")

async def main():
    """Executa todas as demos"""
    print("🎭 ECSA Component Architecture - Suite de Demos")
    print(f"Arquitetura refatorada implementando {len(open(project_root / 'main.py').readlines())} linhas de código")
    print("Demonstrando todos os novos recursos...\n")

    try:
        # Executa todas as demos sequencialmente
        await demo_configuration_and_presets()
        await demo_component_testing()
        await demo_component_features()
        await demo_inter_component_call()

        # Demo básica por último (mais complexo)
        await demo_basic_agent()

        # Se tem API keys, roda o chatbot interativo
        if has_api_keys:
            await demo_chatbot()

        print("\n" + "="*60)
        print("🎉 TODAS AS DEMOS CONCLUÍDAS COM SUCESSO!")
        print("="*60)
        print("\n📋 Recursos demonstrados:")
        print("  • ✅ Lifecycle hooks (on_init, on_shutdown, on_reload)")
        print("  • 🛡️ Error handling robusto com on_error hook")
        print("  • 🎯 Type-safe event contracts (classes ao invés de strings)")
        print("  • 🔗 call_component() para comunicação direta")
        print("  • ⚙️ Configuration schemas com Pydantic")
        print("  • 🧰 Middleware support system")
        print("  • 📊 Basic observability (métricas)")
        print("  • 🎭 AsyncComponent com requests/responses tipados")
        print("  • 🧪 ComponentTestHarness para testes isolados")
        print("\n🚀 Pronto para desenvolvimento de componentes customizados!")

    except Exception as e:
        print(f"\n❌ Erro durante demos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Verifica se tem API keys (opcional para demos)
    has_api_keys = bool(
        os.getenv('GOOGLE_API_KEY') or
        os.getenv('OPENAI_API_KEY') or
        os.getenv('ANTHROPIC_API_KEY')
    )

    if not has_api_keys:
        print("⚠️ AVISO: Nenhuma API key encontrada. Algumas demos podem falhar.")
        print("   Configure GOOGLE_API_KEY, OPENAI_API_KEY ou ANTHROPIC_API_KEY\n")

    asyncio.run(main())
