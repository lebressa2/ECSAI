# 🎯 ECSAI Framework - Tutorial Completo: Todos os Canais de Comunicação

Este tutorial abrangente ensina todos os canais de comunicação do framework ECSAI através de exemplos didáticos práticos.

## 📚 Visão Geral dos Canais

| Canal                    | Propósito                              | Quando Usar                            | Target Format       |
| ------------------------ | -------------------------------------- | -------------------------------------- | ------------------- |
| `agent.send_event()`     | Entrada externa para pipeline completa | Input de usuário/API externo           | `"agent:component"` |
| `agent.dispatch_intra()` | Comunicação intra-agent recursiva      | Workflow interno entre componentes     | `"component"`       |
| `bus.dispatch()`         | Coordenação multi-agent                | Comunicação entre agents independentes | `"agent:component"` |

## 🏗️ Setup - Componentes e Eventos Customizados

### 1. Eventos Customizados

```python
from ecsaai.main import BaseEvent
from ecsaai.src.Events import BaseModel, Field

# Evento para processamento de texto
class TextAnalysisEvent(BaseEvent):
    """Evento para análise de texto"""
    type: str = "text_analysis"
    text: str = Field(..., description="Texto para analisar")
    analysis_type: str = Field(default="sentiment", description="tipo de análise")

# Evento com resultados
class AnalysisResultEvent(BaseEvent):
    """Resultado de análise processada"""
    type: str = "analysis_result"
    original_text: str
    score: float = Field(default=0.0)
    sentiment: str = Field(default="neutral")
    keywords: list = Field(default_factory=list)

# Evento de alerta
class AlertEvent(BaseEvent):
    """Alerta gerado pelo sistema"""
    type: str = "system_alert"
    message: str
    severity: str = "info"
```

### 2. Componente Customizado - TextAnalyzer

```python
from ecsaai.main import Component
from typing import List

class TextAnalyzerComponent(Component):
    """Analisador de texto personalizado"""

    name = "TextAnalyzer"
    receives = [TextAnalysisEvent]  # Recebe análise de texto
    emits = [AnalysisResultEvent, AlertEvent]  # Emite resultados

    async def on_init(self):
        """Inicialização do componente"""
        await super().on_init()
        print("🧠 TextAnalyzer: Ready to analyze text!")
        self.processed_count = 0

    async def handle_event(self, event) -> List[BaseEvent]:
        """Processa evento de análise de texto"""

        if not isinstance(event, TextAnalysisEvent):
            return []

        print(f"📊 Analyzing text: '{event.text}'")

        # Lógica de análise didática
        text = event.text.lower()
        score = 0.5  # Score base

        # Análise simples de sentimento
        positive_words = ["good", "excellent", "amazing", "love", "great"]
        negative_words = ["bad", "terrible", "hate", "awful", "worst"]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            sentiment = "positive"
            score = min(1.0, score + 0.3)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(0.0, score - 0.3)
        else:
            sentiment = "neutral"

        # Extrai keywords simples
        keywords = []
        if "good" in text: keywords.append("good")
        if "bad" in text: keywords.append("bad")
        if "love" in text: keywords.append("love")

        self.processed_count += 1

        # Se é texto muito longo, gera alerta
        alerts = []
        if len(event.text) > 100:
            alerts.append(AlertEvent(
                sender=self.name,
                target=f"{self.agent_id}:AlertLogger",
                message=f"Long text detected ({len(event.text)} chars)",
                severity="warning"
            ))

        # Retorna resultado principal + alertas
        return [
            AnalysisResultEvent(
                sender=self.name,
                target=f"{self.agent_id}:ResultProcessor",
                original_text=event.text,
                score=score,
                sentiment=sentiment,
                keywords=keywords
            )
        ] + alerts

    async def on_shutdown(self):
        """Cleanup do componente"""
        await super().on_shutdown()
        print(f"🧠 TextAnalyzer: Processed {self.processed_count} texts")
```

### 3. Componente ResultProcessor (Para Demonstrar Comunicação)

```python
class ResultProcessorComponent(Component):
    """Processa resultados de análise"""

    name = "ResultProcessor"
    receives = [AnalysisResultEvent]
    emits = []

    async def handle_event(self, event) -> List[BaseEvent]:
        """Exibe resultado formatado"""

        if not isinstance(event, AnalysisResultEvent):
            return []

        print("🎯 ANALYSIS RESULT:")
        print(f"   Text: '{event.original_text}'")
        print(f"   Sentiment: {event.sentiment}")
        print(f".2f")
        print(f"   Keywords: {', '.join(event.keywords) if event.keywords else 'None'}")
        print("-" * 50)

        # Consulta ao contexto via call_component
        if hasattr(self, 'call_component'):
            try:
                from ecsaai.src.Events import GetContextRequest
                context_req = GetContextRequest(session_id="tutorial")
                context_resp = await self.call_component("ContextManager", context_req)

                if context_resp.success:
                    print("💬 Context History:"
                    for msg in context_resp.messages[-2:]:  # Últimas 2
                        print(f"   {msg.get('type', 'unknown')}: {msg.get('content', '')[:30]}...")
            except:
                pass

        return []
```

### 4. Componente AlertLogger (Para Demonstrar Bus)

```python
class AlertLoggerComponent(Component):
    """Logger de alertas - funciona cross-agent"""

    name = "AlertLogger"
    receives = [AlertEvent]
    emits = []

    async def handle_event(self, event) -> List[BaseEvent]:
        """Loga alertas do sistema"""

        if not isinstance(event, AlertEvent):
            return []

        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(event.severity, "ℹ️")
        print(" {emoji} ALERT: {event.message} (Severity: {event.severity})")

        return []
```

## 🎭 Demonstrações - Todos os Canais

### 🎯 Canal 1: `agent.send_event()` - Entrada Externa

```python
async def demo_send_event():
    """Demonstra send_event - entrada direta na pipeline"""

    print("🚀 DEMO 1: send_event() - Pipeline Completa Automática"    print("=" * 60)

    # Cria agent didático
    agent = Agent("analyst")

    # Adiciona componentes customizados
    await agent.add_component(TextAnalyzerComponent)
    await agent.add_component(ResultProcessorComponent)

    # Inicializa
    await agent.init_all()

    # 📨 Entrada via send_event - PROCESSAMENTO COMPLETO AUTOMÁTICO
    user_text = "I absolutely love this amazing product! It's the best thing ever."

    result_events = await agent.send_event(TextAnalysisEvent(
        sender="user",
        target="analyst:TextAnalyzer",  # Target: agent:component
        text=user_text,
        analysis_type="sentiment"
    ))

    print(f"📋 Eventos retornados: {len(result_events)}")
    for event in result_events:
        print(f"   • {event.type} → {event.target}")

    await agent.shutdown_all()
```

### 🔄 Canal 2: `agent.dispatch_intra()` - Comunicação Intra-Agent

```python
async def demo_dispatch_intra():
    """Demonstra dispatch_intra - comunicação recursiva interna"""

    print("\n🚀 DEMO 2: dispatch_intra() - Workflow Interno Recursivo"    print("=" * 60)

    agent = Agent("workflow")

    # Mesmo setup
    await agent.add_component(TextAnalyzerComponent)
    await agent.add_component(ResultProcessorComponent)
    await agent.init_all()

    # 🔗 Chain de eventos intra-agent
    workflow_events = [
        TextAnalysisEvent(
            sender="system",
            target="TextAnalyzer",  # Target: apenas component (sem agent)
            text="This is terrible and awful. I hate it!",
            analysis_type="sentiment"
        ),
        TextAnalysisEvent(
            sender="system",
            target="TextAnalyzer",  # Outro evento automaticamente processado
            text="But the customer support is excellent!",
            analysis_type="sentiment"
        )
    ]

    # Processamento RECURSIVO automático
    chain_results = await agent.dispatch_intra(workflow_events)

    print(f"🔗 Chain processou {len(chain_results)} eventos totais")

    await agent.shutdown_all()
```

### 🚇 Canal 3: `bus.dispatch()` - Coordenação Multi-Agent

```python
async def demo_bus_multi_agent():
    """Demonstra Bus - comunicação cross-agent"""

    print("\n🚀 DEMO 3: bus.dispatch() - Coordenação Multi-Agent"    print("=" * 60)

    # Cria múltiplos agents independentes
    analyzer_agent = Agent("text_analyzer_agent")
    await analyzer_agent.add_component(TextAnalyzerComponent)
    await analyzer_agent.add_component(ResultProcessorComponent)

    alert_agent = Agent("alert_monitor_agent")
    await alert_agent.add_component(AlertLoggerComponent)

    # Inicializa agents
    await analyzer_agent.init_all()
    await alert_agent.init_all()

    # Registra no bus
    bus = EventBus()
    bus.register_agent("analyzer", analyzer_agent)
    bus.register_agent("alert_monitor", alert_agent)

    # 📨 Primeiro envia análise para analyzer_agent
    analysis_events = await bus.dispatch([
        TextAnalysisEvent(
            sender="api_user",
            target="analyzer:TextAnalyzer",  # Agent específico
            text="This product is terrible and I hate it completely!",
            analysis_type="sentiment"
        )
    ])

    # 🚨 Os alertas gerados vão automaticamente para o alert_monitor
    # (através do bus, pois target aponta para alert_monitor:AlertLogger)
    cross_agent_results = await bus.dispatch(analysis_events)

    print("🚇 Bus coordination: Messages processed across different agents")
    print(f"   Total events processed: {len(cross_agent_results)}")

    await analyzer_agent.shutdown_all()
    await alert_agent.shutdown_all()
```

## 🎪 Demo Interativo - Todos os Canais Juntos

```python
async def demo_complete_workflow():
    """Demonstra TODOS os canais funcionando juntos"""

    print("🎪 DEMO COMPLETA: ECOSistema Full com Todos os Canais"    print("=" * 60)

    # Setup multi-agent com bus
    bus = EventBus()

    # Agent Analisador
    analyzer_agent = Agent("analyzer")
    await analyzer_agent.add_component(TextAnalyzerComponent)
    await analyzer_agent.add_component(ResultProcessorComponent)
    bus.register_agent("analyzer", analyzer_agent)

    # Agent Monitor de Alertas (separado)
    alert_agent = Agent("monitor")
    await alert_agent.add_component(AlertLoggerComponent)
    bus.register_agent("monitor", alert_agent)

    await analyzer_agent.init_all()
    await alert_agent.init_all()

    # 📨 Entrada externa - usando send_event no analyzer
    print("📨 Entrada externa via send_event():")
    external_input = await analyzer_agent.send_event(
        TextAnalysisEvent(
            sender="web_app",
            target="analyzer:TextAnalyzer",
            text="This is amazing! I love it so much! But it has some tiny issues.",
            analysis_type="sentiment"
        )
    )

    # 🔗 Processamento interno adicional
    print("\n🔗 Workflow interno via dispatch_intra():")
    internal_workflow = [
        TextAnalysisEvent(
            sender="analyzer_internal",
            target="TextAnalyzer",  # Mesmo agent - sem prefixo
            text="Overall excellent product with great support!",
            analysis_type="sentiment"
        )
    ]

    internal_results = await analyzer_agent.dispatch_intra(internal_workflow)

    # 🚇 Coordenação cross-agent
    print("\n🚇 Coordenação cross-agent via bus.dispatch():")
    all_events_together = external_input + internal_results

    # Os alertas gerados nos events vão para monitor_agent automaticamente
    cross_agent_final = await bus.dispatch(all_events_together)

    print("
🎯 WORKFLOW COMPLETO:"    print(f" - 📨 Entrada externa: {len(external_input)} eventos"    print(f" - 🔗 Workflow interno: {len(internal_results)} eventos"    print(f" - 🚇 Cross-agent: {len(cross_agent_final)} eventos finais"    print("✅ Todos os canais coordenados automaticamente!")

    await analyzer_agent.shutdown_all()
    await alert_agent.shutdown_all()
```

## 🏁 Execução Completa

```python
async def main():
    """Executa todas as demonstrações de canais"""

    print("🎭 ECSAI Framework - Tutorial de Todos os Canais")
    print("=" * 60)

    try:
        await demo_send_event()
        await demo_dispatch_intra()
        await demo_bus_multi_agent()
        await demo_complete_workflow()

        print("\n🎉 TUTORIAL CONCLUÍDO - Todos os canais dominados!")
        print("📚 Framework ECSAI é poderoso e flexível!")

    except Exception as e:
        print(f"❌ Erro no tutorial: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Lições Aprendidas

### `send_event()`:

- ✅ **Interface principal** do agent com mundo externo
- ✅ **Target com nome do agent**: `"agent:component"`
- ✅ **Pipeline completa automática**

### `dispatch_intra()`:

- ✅ **Comunicação interna** entre componentes do mesmo agent
- ✅ **Target apenas component**: `"component"`
- ✅ **Processamento recursivo** de chains de eventos

### `bus.dispatch()`:

- ✅ **Coordenação entre múltiplos agents independentes**
- ✅ **Targets flexíveis**: `"agent:component"` ou apenas `"component"`
- ✅ **Comunicação cross-agent** seamless

---

**🎊 Framework ECSAI masterizado! Parabéns pela compreensão completa dos canais de comunicação!**
