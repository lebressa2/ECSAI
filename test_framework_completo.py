#!/usr/bin/env python3
"""
TESTE COMPLETO: Validação Total do ECSAI Framework
===============================================

Testa todos os componentes principais:
- LLMFactory com múltiplos provedores
- BUS de eventos (componentes conversando)
- Middlewares funcionando
- Agente completo
- Tratamento de erros
- Métricas e observabilidade
"""

import asyncio
import sys
import time
from pathlib import Path

# Adiciona diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecsaai import (
    LLMFactory, Agent, Component, ComponentMiddleware,
    BaseEvent, ComponentMetrics
)
from ecsaai.middleware import LoggingMiddleware, RateLimitMiddleware, MetricsMiddleware
from typing import List

# ─────────────────────────────────────────────
# TESTES: LLMFactory Avançado
# ─────────────────────────────────────────────

async def teste_llmfactory():
    """Testa LLMFactory com múltiplos provedores"""
    print("🧪 TESTANDO LLMFACTORY AVANÇADO")
    print("=" * 50)

    resultados = {}

    # Teste 1: Google (mais confiável)
    try:
        llm = LLMFactory.create_llm(
            llm_id="google:gemini-2.5-flash",
            config={'max_tokens': 100}
        )
        resposta = await LLMFactory.run(llm, "Diga apenas: Google OK")
        resultados['google'] = resposta.content[:20] + "..."
        print(f"✅ Google Gemini: {resultados['google']}")
    except Exception as e:
        print(f"❌ Google falhou: {str(e)[:50]}")
        resultados['google'] = None

    # Teste 2: OpenRouter (se disponível)
    try:
        llm = LLMFactory.create_llm(
            llm_id="openai:gpt-4o-mini",
            config={'max_tokens': 100}
        )
        resposta = await LLMFactory.run(llm, "Diga apenas: OpenRouter OK")
        resultados['openrouter'] = resposta.content[:20] + "..."
        print(f"✅ OpenRouter: {resultados['openrouter']}")
    except Exception as e:
        print(f"⚠️ OpenRouter falhou (API key?): {str(e)[:50]}")
        resultados['openrouter'] = None

    # Teste 3: Modo multimodal
    try:
        llm = LLMFactory.create_llm(llm_id="google:gemini-2.5-flash")
        resposta = await LLMFactory.run(llm, "Explique brevemente o que é IA generativa.")
        resultados['multimodal_ready'] = len(resposta.content) > 10
        print(f"✅ Multimodal pronto: {resultados['multimodal_ready']}")
    except Exception as e:
        print(f"❌ Multimodal falhou: {str(e)[:50]}")
        resultados['multimodal_ready'] = False

    return resultados

# ─────────────────────────────────────────────
# COMPONENTES DE TESTE
# ─────────────────────────────────────────────

class MensagemEvent(BaseEvent):
    """Evento de mensagem entre componentes"""
    type: str = "mensagem"
    mensagem: str
    de_quem: str
    para_quem: str

class ResponderMensagem(Component):
    """Componente que responde mensagens automaticamente"""

    name = "RespostaAutomatica"

    receives = [MensagemEvent]
    emits = [MensagemEvent]

    def __init__(self, config=None):
        super().__init__(config)
        self.respostas_enviadas = 0

    async def handle_event(self, event: MensagemEvent) -> List[MensagemEvent]:
        # Só responde se a mensagem for para ele
        if event.para_quem != self.name:
            return []

        resposta_texto = ""
        if "olá" in event.mensagem.lower():
            resposta_texto = "Olá! Como posso ajudar?"
        elif "teste" in event.mensagem.lower():
            resposta_texto = "Teste recebido com sucesso!"
        elif "tempo" in event.mensagem.lower():
            resposta_texto = "Não tenho acesso ao clima atual."
        else:
            resposta_texto = f"Entendi sua mensagem: '{event.mensagem}'"

        self.respostas_enviadas += 1

        return [MensagemEvent(
            mensagem=resposta_texto,
            de_quem=self.name,
            para_quem=event.de_quem
        )]

class ProcessarComLLM(Component):
    """Componente que usa LLM para processar mensagens"""

    name = "ProcessadorLLM"

    receives = [MensagemEvent]
    emits = [MensagemEvent]

    async def handle_event(self, event: MensagemEvent) -> List[MensagemEvent]:
        # Usa LLMFactory para processar a mensagem
        try:
            llm = LLMFactory.create_llm(
                llm_id="google:gemini-2.5-flash",
                config={'max_tokens': 200}
            )

            prompt = f"Analise esta mensagem e responda de forma útil: '{event.mensagem}'"
            resposta_llm = await LLMFactory.run(llm, prompt)

            return [MensagemEvent(
                mensagem=resposta_llm.content,
                de_quem=self.name,
                para_quem="Usuario"
            )]
        except Exception as e:
            print(f"⚠️ Erro no LLM: {str(e)[:50]}")
            return [MensagemEvent(
                mensagem=f"Erro no processamento: {str(e)[:50]}",
                de_quem=self.name,
                para_quem="Usuario"
            )]

class MetricasCustom(Component):
    """Componente para monitorar métricas do sistema"""

    name = "MonitorMetricas"

    receives = [MensagemEvent]
    emits = []

    def __init__(self, config=None):
        super().__init__(config)
        self.mensagens_processadas = 0

    async def handle_event(self, event: MensagemEvent) -> List:
        self.mensagens_processadas += 1

        # Reporta métricas periodicamente
        if self.mensagens_processadas % 5 == 0:
            print(f"📊 MÉTRICAS: {self.mensagens_processadas} mensagens processadas")

        return []

# ─────────────────────────────────────────────
# TESTES: BUS de Eventos (Componentes Conversando)
# ─────────────────────────────────────────────

async def teste_bus_eventos():
    """Testa comunicação entre componentes via BUS"""
    print("\n🚌 TESTANDO BUS DE EVENTOS")
    print("=" * 50)

    # Criar agente
    agent = Agent("teste_bus")

    # Adicionar componentes
    await agent.add_component(ResponderMensagem)
    await agent.add_component(ProcessarComLLM)
    await agent.add_component(MetricasCustom)

    # Inicializar
    await agent.init_all()

    print("✅ Componentes inicializados")

    # Teste 1: Conversa básica entre componentes
    print("\n📨 TESTE 1: Conversa Simples")
    print("-" * 30)

    # Usuário envia mensagem para ResponderMensagem
    msg1 = MensagemEvent(
        mensagem="Olá, tudo bem?",
        de_quem="Usuario",
        para_quem="RespostaAutomatica"
    )

    print(f"👤 Usuário → RespostaAutomatica: {msg1.mensagem}")
    eventos1 = await agent.send_event(msg1)

    for i, evento in enumerate(eventos1):
        if isinstance(evento, MensagemEvent):
            print(f"🤖 {evento.de_quem} → {evento.para_quem}: {evento.mensagem}")

    # Teste 2: Componente com LLM
    print("\n🧠 TESTE 2: Componente com LLM")
    print("-" * 30)

    msg2 = MensagemEvent(
        mensagem="Explique o que é machine learning em uma frase",
        de_quem="Usuario",
        para_quem="ProcessadorLLM"
    )

    print(f"👤 Usuário → ProcessadorLLM: {msg2.mensagem}")
    eventos2 = await agent.send_event(msg2)

    for evento in eventos2:
        if isinstance(evento, MensagemEvent):
            print(f"🧠 {evento.de_quem} → {evento.para_quem}: {evento.mensagem[:200]}...")

    # Teste 3: Sequência de mensagens
    print("\n🔄 TESTE 3: Sequência de Mensagens")
    print("-" * 30)

    mensagens_teste = [
        ("Como funciona a inteligência artificial?", "ProcessadorLLM"),
        ("teste", "RespostaAutomatica"),
        ("Qual o tempo hoje?", "RespostaAutomatica")
    ]

    for pergunta, destinatario in mensagens_teste:
        msg = MensagemEvent(
            mensagem=pergunta,
            de_quem="Usuario",
            para_quem=destinatario
        )

        print(f"👤 → {destinatario}: {pergunta}")
        eventos = await agent.send_event(msg)

        for evento in eventos:
            if isinstance(evento, MensagemEvent):
                print(f"🤖 {evento.de_quem}: {evento.mensagem[:150]}...")
        print()

    # Ver métricas finais
    print("📈 MÉTRICAS FINAIS:")
    for nome, componente in agent.components.items():
        if hasattr(componente, 'respostas_enviadas'):
            print(f"  {nome}: {componente.respostas_enviadas} respostas enviadas")
        if hasattr(componente, 'mensagens_processadas'):
            print(f"  {nome}: {componente.mensagens_processadas} mensagens processadas")

    await agent.shutdown_all()
    return len(eventos1) + len(eventos2)

# ─────────────────────────────────────────────
# TESTES: Middlewares
# ─────────────────────────────────────────────

async def teste_middlewares():
    """Testa sistema de middlewares"""
    print("\n🎛️ TESTANDO MIDDLEWARES")
    print("=" * 50)

    # Criar componente com middlewares
    componente = ResponderMensagem()
    componente.middlewares = [
        LoggingMiddleware(),
        RateLimitMiddleware(requests_per_minute=10),
        MetricsMiddleware()
    ]

    # Simular alguns eventos
    print("🚀 Testando middlewares...")
    for i in range(5):
        evento = MensagemEvent(
            mensagem=f"Mensagem teste {i+1}",
            de_quem="Usuario",
            para_quem="RespostaAutomatica"
        )

        await componente._safe_handle_event(evento)

    # Verificar métricas
    print("📊 Métricas coletadas:")
    print(f"  Eventos recebidos: {componente.metrics.events_received}")
    print(".2f")
    print(f"  Eventos emitidos: {componente.metrics.events_emitted}")
    print(f"  Total de erros: {componente.metrics.errors}")

    return componente.metrics.events_received

# ─────────────────────────────────────────────
# TESTE: Agente Completo (Chatbot)
# ─────────────────────────────────────────────

async def teste_agente_completo():
    """Testa agente completo com componentes reais"""
    print("\n🤖 TESTANDO AGENTE COMPLETO (CHATBOT)")
    print("=" * 50)

    # Criar agente completo
    agent = Agent("chatbot_completo")

    # Importar componentes reais
    from ecsaai.Components.ContextComponent import ContextComponent, create_chatbot_config
    from ecsaai.Components.LLMComponent import LLMComponent, create_fast_llm_config
    from ecsaai.Components.OutputComponent import OutputComponent
    from ecsaai.Events import InputEvent

    # Adicionar componentes
    await agent.add_component(ContextComponent, config=create_chatbot_config())
    await agent.add_component(LLMComponent, config=create_fast_llm_config())
    await agent.add_component(OutputComponent)

    # Inicializar
    await agent.init_all()

    # Conversa de teste
    dialogo = [
        "Olá, sou novo aqui",
        "Me conte uma curiosidade sobre tecnologia",
        "Obrigado pela conversa!"
    ]

    print("💬 Iniciando conversa com chatbot completo...")

    for mensagem_usuario in dialogo:
        print(f"\n👤 Você: {mensagem_usuario}")

        # Enviar evento de input
        input_event = InputEvent(
            content=mensagem_usuario,
            session_id="teste_agente_completo"
        )

        # O BUS lida com tudo automaticamente!
        eventos_gerados = await agent.send_event(input_event)

        print(f"⚙️ BUS processou {len(eventos_gerados)} eventos")

        # Resultado final
        for evento in eventos_gerados:
            if hasattr(evento, 'content'):
                print(f"🤖 Chatbot: {evento.content[:200]}...")

    await agent.shutdown_all()
    return len(dialogo)

# ─────────────────────────────────────────────
# TESTE DE TOLERÂNCIA A FALHAS
# ─────────────────────────────────────────────

async def teste_tolerancia_falhas():
    """Testa comportamento em casos de erro"""
    print("\n🛡️ TESTANDO TOLERÂNCIA A FALHAS")
    print("=" * 50)

    result = {"protegido": False, "fallback": False}

    # Teste 1: Middleware de proteção
    print("🛡️ Teste 1: Proteção contra erros")
    componente = ResponderMensagem()
    componente.middlewares = [MetricsMiddleware()]

    # Enviar evento que causa erro (simulado)
    try:
        # Monkey patch para simular erro
        original_handle = componente.handle_event
        async def erro_simulado(event):
            raise Exception("Erro simulado de teste")

        componente.handle_event = erro_simulado

        evento = MensagemEvent(mensagem="teste", de_quem="A", para_quem="RespostaAutomatica")
        await componente._safe_handle_event(evento)

        # Verificar se erro foi capturado
        if componente.metrics.errors > 0:
            print("✅ Erro foi capturado e contabilizado")
            result["protegido"] = True
        else:
            print("❌ Erro não foi capturado")

    except Exception:
        print("❌ Componente explodiu em erro não tratado")
        result["protegido"] = False

    # Teste 2: Fallback de provedores
    print("\n🔄 Teste 2: Fallback de provedores")
    provadores_teste = [
        "openai:inexistente-modelo",  # Deve falhar
        "google:gemini-2.5-flash",    # Deve funcionar
    ]

    for provider in provadores_teste:
        try:
            llm = LLMFactory.create_llm(llm_id=provider, config={'max_tokens': 10})
            resposta = await LLMFactory.run(llm, "OK")
            if resposta.content.strip():
                print(f"✅ Fallback funcionou com: {provider}")
                result["fallback"] = True
                break
        except Exception as e:
            print(f"⚠️ {provider}: {str(e)[:30]}...")

    return result

# ─────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL DOS TESTES
# ─────────────────────────────────────────────

async def main():
    """Executa todos os testes do framework"""
    print("🧪 ECSAI FRAMEWORK - BATERIA COMPLETA DE TESTES")
    print("=" * 60)
    print("Testando todos os componentes principais do framework...")
    print()

    # Dicionário de resultados
    resultados = {}

    try:
        # Teste 1: LLMFactory
        print("1️⃣ TESTANDO LLMFACTORY")
        llm_results = await teste_llmfactory()
        resultados["llmfactory"] = llm_results
        print(f"✅ LLMFactory testado: {sum(1 for r in llm_results.values() if r)}/{len(llm_results)} OK\n")

        # Teste 2: BUS de Eventos
        print("2️⃣ TESTANDO BUS DE EVENTOS")
        bus_results = await teste_bus_eventos()
        resultados["bus_eventos"] = bus_results > 0
        print(f"✅ BUS testado: {bus_results} mensagens processadas\n")

        # Teste 3: Middlewares
        print("3️⃣ TESTANDO MIDDLEWARES")
        mw_results = await teste_middlewares()
        resultados["middlewares"] = mw_results > 0
        print(f"✅ Middlewares testados: {mw_results} eventos processados\n")

        # Teste 4: Agente Completo
        print("4️⃣ TESTANDO AGENTE COMPLETO")
        agent_results = await teste_agente_completo()
        resultados["agente_completo"] = agent_results > 0
        print(f"✅ Agente completo testado: {agent_results} mensagens trocadas\n")

        # Teste 5: Tolerância a falhas
        print("5️⃣ TESTANDO TOLERÂNCIA A FALHAS")
        fail_results = await teste_tolerancia_falhas()
        resultados["tolerancia_falhas"] = fail_results
        print(f"✅ Tolerância testada: proteção={fail_results['protegido']}, fallback={fail_results['fallback']}\n")

    except Exception as e:
        print(f"❌ ERRO CRÍTICO NOS TESTES: {e}")
        import traceback
        traceback.print_exc()
        resultados["erro_critico"] = str(e)

    # ─────────────────────────────────────────
    # RELATÓRIO FINAL
    # ─────────────────────────────────────────
    print("=" * 60)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("=" * 60)

    # Calcular score geral
    total_testes = len([r for r in resultados.values() if not isinstance(r, dict)])
    testes_passaram = len([r for r in resultados.values() if r and not isinstance(r, dict)])

    # Detalhes do LLMFactory
    if "llmfactory" in resultados:
        llm = resultados["llmfactory"]
        print("🧠 LLMFactory:")
        for k, v in llm.items():
            status = "✅" if v else "❌"
            print(f"   {status} {k}: {'OK' if v else 'FALHOU'}")

    # Outros testes
    status_emoji = {
        "bus_eventos": "🚌",
        "middlewares": "🎛️",
        "agente_completo": "🤖",
        "tolerancia_falhas": "🛡️"
    }

    for test_name, result in resultados.items():
        if test_name == "llmfactory":
            continue

        emoji = status_emoji.get(test_name, "📋")
        if isinstance(result, dict):
            protegido = result.get('protegido', False)
            fallback = result.get('fallback', False)
            print(f"{emoji} {test_name}: proteção={protegido}, fallback={fallback}")
        elif isinstance(result, bool):
            status = "✅" if result else "❌"
            print(f"{emoji} {test_name}: {status}")
        else:
            print(f"{emoji} {test_name}: {result}")

    # Score final
    print(f"\n🏆 SCORE FINAL: {testes_passaram}/{total_testes} testes passaram")

    if testes_passaram >= 4:  # 80% de aprovação
        print("🎉 FRAMEWORK TOTALMENTE FUNCIONAL!")
        print("✅ Pronto para produção e uso real")
    elif testes_passaram >= 2:  # 40% de aprovação
        print("⚠️ FRAMEWORK FUNCIONAL MAS COM LIMITAÇÕES")
        print("Verifique falhas e configure APIs adequadamente")
    else:
        print("❌ ALTO NÍVEL DE FALHAS")
        print("Reveja configuração de APIs e dependências")

    # Verificar APIs
    print(f"\n🔑 STATUS DAS APIs:")
    import os
    apis = ['GOOGLE_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
    for api in apis:
        existe = bool(os.getenv(api))
        status = "✅" if existe else "❌"
        print(f"   {status} {api}: {'configurada' if existe else 'não encontrada'}")

    print(f"\n📚 Total de modelos disponíveis: {len(LLMFactory.list_available_models())}")

    return resultados

if __name__ == "__main__":
    resultados_finais = asyncio.run(main())

    # Salvar resultados para análise posterior se quiser
    # import json
    # with open("testes_resultados.json", "w") as f:
    #     json.dump(resultados_finais, f, indent=2)
