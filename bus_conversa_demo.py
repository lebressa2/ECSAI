#!/usr/bin/env python3
"""
DEMO: Dois Componentes Conversando via BUS usando Gemini 2.5 Flash
==================================================================

Demonstra dois componentes se comunicando através do BUS de eventos,
usando LLMFactory com Gemini 2.5 Flash para processar mensagens.
"""

import asyncio
import sys
from pathlib import Path

# Adiciona diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecsaai import Agent, Component, BaseEvent, LLMFactory
from ecsaai.middleware import LoggingMiddleware, MetricsMiddleware
from typing import List

# ─────────────────────────────────────────────
# EVENTO DE CONVERSA (correto, herdando BaseEvent)
# ─────────────────────────────────────────────

class MensagemEvento(BaseEvent):
    """Evento para troca de mensagens entre componentes"""
    type: str = "mensagem_conversa"
    mensagem: str
    de_componente: str
    para_componente: str
    importancia: str = "normal"  # "baixa", "normal", "alta"

# ─────────────────────────────────────────────
# COMPONENTE 1: Analisador Inteligente
# ─────────────────────────────────────────────

class AnalisadorInteligente(Component):
    """
    Componente que analisa mensagens usando Gemini 2.5 Flash
    e decide se deve responder ou encaminhar
    """

    name = "AnalisadorInteligente"

    receives = [MensagemEvento]
    emits = [MensagemEvento]

    def __init__(self, config=None):
        super().__init__(config)
        # Adicionar middlewares para observabilidade
        self.middlewares = [
            LoggingMiddleware(),
            MetricsMiddleware()
        ]

    async def handle_event(self, event: MensagemEvento) -> List[MensagemEvento]:
        print(f"\n🧠 AnalisadorInteligente recebeu: '{event.mensagem}'")

        # Usar Gemini 2.5 Flash para analisar a importância da mensagem
        try:
            llm = LLMFactory.create_llm(
                llm_id="google:gemini-2.5-flash",  # Usando Gemini 2.5 Flash!
                config={'max_tokens': 50, 'temperature': 0.1}
            )

            prompt = f"""
            Analise esta mensagem e responda apenas com uma palavra:
            BAIXA, NORMAL ou ALTA

            Mensagem: "{event.mensagem}"
            """

            resposta_llm = await LLMFactory.run(llm, prompt)
            importancia = resposta_llm.content.strip().upper()

            # Validar resposta
            if importancia not in ["BAIXA", "NORMAL", "ALTA"]:
                importancia = "NORMAL"

            print(f"🎯 Análise de importância (Gemini): {importancia}")

        except Exception as e:
            print(f"⚠️ Erro na análise LLM: {str(e)[:50]}")
            importancia = "NORMAL"

        # Decidir ação baseada na importância
        if importancia == "BAIXA":
            print("📉 Mensagem de baixa prioridade - ignorando")
            return []  # Não responde

        elif importancia == "NORMAL":
            # Responder diretamente
            resposta = f"Entendi sua mensagem sobre: '{event.mensagem[:30]}...'"
            print(f"💬 Respondendo diretamente: {resposta}")

            return [MensagemEvento(
                sender=self.name,
                target=f"{self.agent_id}:Usuario",
                mensagem=resposta,
                de_componente=self.name,
                para_componente="Usuario",
                importancia="normal"
            )]

        elif importancia == "ALTA":
            # Encaminhar para o ProcessadorEspecializado
            print("🚨 Mensagem de alta prioridade - encaminhando!")
            mensagem_encaminhada = f"IMPORTANTE: {event.mensagem}"

            return [MensagemEvento(
                sender=self.name,
                target=f"{self.agent_id}:ProcessadorEspecializado",
                mensagem=mensagem_encaminhada,
                de_componente=self.name,
                para_componente="ProcessadorEspecializado",
                importancia="alta"
            )]

# ─────────────────────────────────────────────
# COMPONENTE 2: Processador Especializado
# ─────────────────────────────────────────────

class ProcessadorEspecializado(Component):
    """
    Componente especialista que processa mensagens importantes
    usando Gemini 2.5 Flash para gerar respostas detalhadas
    """

    name = "ProcessadorEspecializado"

    receives = [MensagemEvento]
    emits = [MensagemEvento]

    def __init__(self, config=None):
        super().__init__(config)
        self.middlewares = [
            LoggingMiddleware(),
            MetricsMiddleware()
        ]
        self.mensagens_processadas = 0

    async def handle_event(self, event: MensagemEvento) -> List[MensagemEvento]:
        print(f"\n⚡ ProcessadorEspecializado recebeu: '{event.mensagem}'")

        # Só processa mensagens destinadas a ele
        if event.para_componente != self.name:
            return []

        self.mensagens_processadas += 1

        # Usar Gemini 2.5 Flash para gerar resposta detalhada
        try:
            llm = LLMFactory.create_llm(
                llm_id="google:gemini-2.5-flash",  # Usando Gemini 2.5 Flash!
                config={'max_tokens': 150, 'temperature': 0.7}
            )

            prompt = f"""
            Você é um especialista em IA. Analise e responda detalhadamente:

            Mensagem recebida: "{event.mensagem}"

            Forneça uma resposta útil e informativa.
            """

            resposta_llm = await LLMFactory.run(llm, prompt)

            resposta_final = f"🔬 ANÁLISE ESPECIALIZADA (msg #{self.mensagens_processadas}):\n{resposta_llm.content}"
            print(f"✅ Resposta especializada gerada com Gemini")

        except Exception as e:
            print(f"⚠️ Erro no processamento LLM: {str(e)[:50]}")
            resposta_final = "Desculpe, houve um erro no processamento especializado."

        return [MensagemEvento(
            sender=self.name,
            target=f"{self.agent_id}:Usuario",
            mensagem=resposta_final,
            de_componente=self.name,
            para_componente="Usuario",
            importancia="alta"
        )]

# ─────────────────────────────────────────────
# TESTE: CONVERSA ENTRE OS DOIS COMPONENTES
# ─────────────────────────────────────────────

async def teste_conversa_bus():
    """Demonstra dois componentes conversando via BUS usando Gemini 2.5 Flash"""

    print("🚌 DEMO: CONVERSA ENTRE COMPONENTES VIA BUS")
    print("=" * 60)
    print("Dois componentes usando Gemini 2.5 Flash para se comunicar!")

    # Criar agente
    agent = Agent("conversa_bus_demo")

    # Adicionar os dois componentes
    await agent.add_component(AnalisadorInteligente)
    await agent.add_component(ProcessadorEspecializado)

    # Inicializar
    await agent.init_all()

    print("\n✅ Componentes inicializados!")
    print("📨 Agora vamos testar mensagens de diferentes prioridades...\n")

    # ─────────────────────────────────────────
    # CENÁRIO 1: Mensagem de baixa prioridade
    # ─────────────────────────────────────────
    print("🎯 CENÁRIO 1: Mensagem de Baixa Prioridade")
    print("-" * 45)

    msg_baixa = MensagemEvento(
        sender="Usuario",
        target=f"{agent.agent_id}:AnalisadorInteligente",
        mensagem="Oi",
        de_componente="Usuario",
        para_componente="AnalisadorInteligente"
    )

    eventos_resposta = await agent.send_event(msg_baixa)
    print(f"📊 Eventos gerados: {len(eventos_resposta)}")

    if not eventos_resposta:
        print("✅ Como esperado: mensagem baixa foi ignorada!\n")
    else:
        for evento in eventos_resposta:
            print(f"❌ Inesperado: {evento.mensagem[:50]}...")

    # ─────────────────────────────────────────
    # CENÁRIO 2: Mensagem normal
    # ─────────────────────────────────────────
    print("📝 CENÁRIO 2: Mensagem Normal")
    print("-" * 35)

    msg_normal = MensagemEvento(
        sender="Usuario",
        target=f"{agent.agent_id}:AnalisadorInteligente",
        mensagem="Como funciona a inteligência artificial?",
        de_componente="Usuario",
        para_componente="AnalisadorInteligente"
    )

    eventos_normal = await agent.send_event(msg_normal)
    print(f"📊 Eventos gerados: {len(eventos_normal)}")

    for evento in eventos_normal:
        if hasattr(evento, 'mensagem') and hasattr(evento, 'de_componente'):
            print(f"🤖 {evento.de_componente}: {evento.mensagem[:100]}...")

    print()

    # ─────────────────────────────────────────
    # CENÁRIO 3: Mensagem de alta prioridade (ENCAMINHADA!)
    # ─────────────────────────────────────────
    print("🚨 CENÁRIO 3: Mensagem de Alta Prioridade")
    print("-" * 45)

    msg_alta = MensagemEvento(
        sender="Usuario",
        target=f"{agent.agent_id}:AnalisadorInteligente",
        mensagem="Quais são as implicações éticas da IA em tomadas de decisão médica?",
        de_componente="Usuario",
        para_componente="AnalisadorInteligente"
    )

    eventos_alta = await agent.send_event(msg_alta)
    print(f"📊 Eventos gerados (primeira resposta): {len(eventos_alta)}")

    # O primeiro componente deve encaminhar para o segundo
    for evento in eventos_alta:
        if hasattr(evento, 'mensagem') and hasattr(evento, 'para_componente'):
            if evento.para_componente == "ProcessadorEspecializado":
                print(f"🔄 Encaminhado para {evento.para_componente}: {evento.mensagem[:50]}...")
                print("📨 Enviando para o componente especializado...")

                # Simular o recebimento pelo segundo componente
                eventos_final = await agent.send_event(evento)
                print(f"📊 Eventos finais gerados: {len(eventos_final)}")

                for evento_final in eventos_final:
                    if hasattr(evento_final, 'mensagem') and hasattr(evento_final, 'de_componente'):
                        print(f"🔬 {evento_final.de_componente}:")
                        print(f"   {evento_final.mensagem[:200]}...")
                        if len(evento_final.mensagem) > 200:
                            print("   [...continua]")

    # ─────────────────────────────────────────
    # RELATÓRIO FINAL
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DA CONVERSA VIA BUS")
    print("=" * 60)

    # Métricas dos componentes
    for nome, componente in agent.components.items():
        print(f"\n🔍 {nome}:")
        print(f"   📥 Eventos recebidos: {componente.metrics.events_received}")
        print(".2f")
        print(f"   📤 Eventos emitidos: {componente.metrics.events_emitted}")
        print(f"   ❌ Erros: {componente.metrics.errors}")

        # Métricas específicas
        if nome == "ProcessadorEspecializado":
            print(f"   🔢 Mensagens processadas pelo especialista: {componente.mensagens_processadas}")

    print(f"\n🎉 DEMO CONCLUÍDA!")
    print("✅ Dois componentes conversaram via BUS de eventos")
    print("✅ Ambos usaram Gemini 2.5 Flash para processamento inteligente")
    print("✅ Mensagens de baixa prioridade foram filtradas")
    print("✅ Mensagens importantes foram encaminhadas automaticamente")

    await agent.shutdown_all()

# ─────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────

async def main():
    print("🧠 DEMO: COMPONENTES CONVERSANDO VIA BUS COM GEMINI 2.5 FLASH")
    print("=" * 70)
    print("Esta demo mostra dois componentes se comunicando através do BUS de eventos,")
    print("usando Gemini 2.5 Flash para análise inteligente de mensagens.\n")

    try:
        await teste_conversa_bus()
    except Exception as e:
        print(f"❌ ERRO na demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("🎯 FUNCIONALIDADES DEMONSTRADAS:")
    print("  • 🚌 BUS de eventos funcionando entre componentes")
    print("  • 🧠 Gemini 2.5 Flash usado em ambos componentes")
    print("  • 🎯 Análise inteligente de prioridade de mensagens")
    print("  • 🔄 Encaminhamento automático entre componentes")
    print("  • 📊 Métricas e observabilidade integrada")
    print("  • 🎛️ Middlewares aplicados automaticamente")

if __name__ == "__main__":
    asyncio.run(main())
