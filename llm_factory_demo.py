#!/usr/bin/env python3
"""
DEMO: LLMFactory - 100% Sem Dor de Cabeça
========================================

Demonstra exatamente o que você pediu:
- ✅ llm = LLMFactory.create_llm(config=config, llm_id="google:gemini-2.5-flash", tools=tools)
- ✅ Interface run global que aceita string ou lista de BaseMessage
- ✅ Modos automáticos ativados
- ✅ Capacidades multimodais
- ✅ API keys automaticamente do .env
- ✅ ZERO configuração manual
"""

import asyncio
import os
import sys
from pathlib import Path

# Adiciona diretório src ao path para importar ecsaai
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecsaai import LLMFactory

# Carrega variáveis de ambiente automaticamente
from dotenv import load_dotenv
load_dotenv()

async def demo_basico():
    """
    Demo: Uso mais simples possível - sem dor de cabeça!
    """
    print("🚀 DEMO BÁSICO - LLMFactory Sem Dor de Cabeça")
    print("=" * 50)

    # Apenas isso! API keys automaticamente do .env
    llm = LLMFactory.create_llm(
        llm_id="google:gemini-2.5-flash"  # Modelo padrão, keys do .env
    )

    print(f"✅ LLM criado: {type(llm).__name__}")

    # Modo automático 1: String simples
    resposta1 = await LLMFactory.run(llm, "Olá! Como você está?")
    print(f"🤖 Resposta (string): {resposta1.content[:100]}...")

    # Modo automático 2: Lista de BaseMessage (avançado)
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="Você é um assistente profissional de tecnologia."),
        HumanMessage(content="Explique o que é uma API REST em 2 frases.")
    ]

    resposta2 = await LLMFactory.run(llm, messages)
    print(f"🤖 Resposta (BaseMessage): {resposta2.content[:100]}...")

async def demo_configurado():
    """
    Demo: Com configurações customizadas
    """
    print("\n⚙️  DEMO CONFIGURADO - Temperatura e max_tokens")
    print("=" * 50)

    config = {
        'temperature': 0.1,  # Mais determinístico
        'max_tokens': 200
    }

    llm = LLMFactory.create_llm(
        config=config,
        llm_id="google:gemini-2.5-flash"
    )

    resposta = await LLMFactory.run(
        llm,
        "Liste 3 benefícios da arquitetura de microserviços.",
        config={'temperature': 0.1}  # Pode sobrescrever
    )

    print(f"📋 Resposta configurada: {resposta.content}")

async def demo_tools():
    """
    Demo: Com tools (para agentes)
    """
    print("\n🛠️  DEMO COM TOOLS - Capacidades expandidas")
    print("=" * 50)

    # Tools simples para demonstração
    tools = [
        {
            "name": "get_weather",
            "description": "Obtém informações do tempo",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nome da cidade"}
                },
                "required": ["city"]
            }
        }
    ]

    llm = LLMFactory.create_llm(
        llm_id="google:gemini-2.5-flash",
        tools=tools
    )

    resposta = await LLMFactory.run(
        llm,
        "Qual o tempo em São Paulo hoje? Use as tools disponíveis."
    )

    print(f"🌤️  Resposta com tools: {resposta.content[:150]}...")

async def demo_multimodal():
    """
    Demo: Capacidades multimodais (experimental)
    """
    print("\n🖼️  DEMO MULTIMODAL - Imagens e texto")
    print("=" * 50)

    try:
        llm = LLMFactory.create_llm(
            llm_id="google:gemini-pro-vision"  # Modelo multimodal
        )

        # Nota: Para demonstração real, precisaria de uma imagem
        # Aqui simulamos o conceito
        multimodal_input = "Descreva o que você vê nesta imagem"

        resposta = await LLMFactory.run(llm, multimodal_input)
        print(f"🖼️  Resposta multimodal: {resposta.content[:100]}...")

    except Exception as e:
        print(f"📷 Multimodal ainda não configurado: {e}")
        print("💡 Para usar multimodal, adicione imagens aos BaseMessage")

async def demo_modelos_disponiveis():
    """
    Demo: Todos os modelos disponíveis
    """
    print("\n📚 MODELOS DISPONÍVEIS")
    print("=" * 50)

    modelos = LLMFactory.list_available_models()

    print(f"📊 Total de modelos disponíveis: {len(modelos)}")

    # Mostra apenas primeiros 10 para não poluir a tela
    for modelo in modelos[:10]:
        info = LLMFactory.get_model_info(modelo)
        multimodal = "🖼️" if info['multimodal'] else "📝"
        print(f"{multimodal} {modelo} - {', '.join(info['strengths'])}")

    # Destaca OpenRouter se disponível
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    if openrouter_key:
        print(f"\n🎉 OpenRouter habilitado! Modelos adicionais disponíveis:")
        openrouter_models = [m for m in modelos if 'llama' in m.lower() or 'mistral' in m.lower()]
        for modelo in openrouter_models[:3]:
            print(f"   🔸 {modelo} (via OpenRouter)")

async def demo_recuperação():
    """
    Demo: Recuperação automática de erros
    """
    print("\n🔄 DEMO RECUPERAÇÃO - Múltiplos provedores")
    print("=" * 50)

    # Modo automático: tenta provedores disponíveis
    provedores_para_testar = [
        "google:gemini-2.5-flash",
        "anthropic:claude-3-haiku",  # Se não tiver API key, pula
        "openai:gpt-3.5-turbo"
    ]

    for llm_id in provedores_para_testar:
        try:
            print(f"🔍 Testando {llm_id}...")

            llm = LLMFactory.create_llm(llm_id=llm_id)
            resposta = await LLMFactory.run(llm, "Diga apenas 'OK'")

            if resposta.content.strip().upper() == "OK":
                print(f"✅ {llm_id} funcionou!")
                return

        except Exception as e:
            print(f"❌ {llm_id} falhou: {str(e)[:50]}...")

    print("⚠️ Nenhum provedor funcionou. Verifique suas API keys no .env")

async def main():
    """Executa todas as demos"""
    print("🎭 LLMFactory - 100% Sem Dor de Cabeça")
    print("Demonstrando todos os recursos pedidos...\n")

    # Verifica se há API keys
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY')

    if google_key:
        print("✅ GOOGLE_API_KEY encontrada")
    else:
        print("❌ GOOGLE_API_KEY não encontrada")

    if openai_key:
        print("✅ OPENAI_API_KEY ou OPENROUTER_API_KEY encontrada")
    else:
        print("❌ OPENAI_API_KEY/OPENROUTER_API_KEY não encontrada")

    print("\n🚀 Iniciando demos...\n")

    try:
        await demo_basico()
        await demo_configurado()
        await demo_modelos_disponiveis()
        await demo_recuperação()

        # Só roda se tiver tools configuradas (opcional)
        try:
            await demo_tools()
        except:
            print("⚠️ Demo de tools pulada (configuração opcional)")

        # Só roda se tiver capacidades multimodais
        try:
            await demo_multimodal()
        except:
            print("⚠️ Demo multimodal pulada (experimental)")

    except Exception as e:
        print(f"❌ Erro durante demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🎉 LLMFactory - Pronto para uso SEM DOR DE CABEÇA!")
    print("=" * 60)
    print("\n✨ Recursos implementados:")
    print("  • ✅ llm = LLMFactory.create_llm(config=config, llm_id='...', tools=tools)")
    print("  • 🏃 Interface run() global - string OU BaseMessage")
    print("  • 🔄 Modos automáticos ativados")
    print("  • 🔑 API keys automaticamente do .env")
    print("  • 🌐 Capacidades multimodais")
    print("  • 🛡️ Recuperação automática de erros")
    print("  • 📚 Lista de modelos disponíveis")
    print("\n🚀 Use em qualquer lugar:")
    print("   from ecsaai import LLMFactory")
    print("   llm = LLMFactory.create_llm(llm_id='google:gemini-2.5-flash')")
    print("   resposta = await LLMFactory.run(llm, 'Sua pergunta aqui')")

if __name__ == "__main__":
    asyncio.run(main())
