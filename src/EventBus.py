from typing import Dict, List, Optional
from main import BaseEvent, Agent
import logging
import asyncio

logger = logging.getLogger(__name__)


class DumpComponent:
    """Component simple para armazenar mensagens sem target"""

    def __init__(self):
        self.stored_messages: List[BaseEvent] = []

    def store_message(self, event: BaseEvent):
        self.stored_messages.append(event)
        logger.info(f"Mensagem sem target armazenada: {event.type} de {event.sender}")

    def get_messages(self) -> List[BaseEvent]:
        return self.stored_messages


class EventBus:
    """
    Bus principal de eventos com registro de agentes.
    Permite comunicação inter-agentes via 'agent_id:component' targets,
    e intra-agente através do bus interno do agente.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.dump_component = DumpComponent()

    def register_agent(self, agent: Agent):
        """
        Registra um agente no bus principal para comunicação inter-agente.
        """
        agent_id = agent.agent_id

        if agent_id in self.agents:
            raise ValueError(f"Agente '{agent_id}' já registrado")

        self.agents[agent_id] = agent
        # Pass a reference to the external bus if needed
        agent.set_external_bus(self)
        logger.info(f"Agente '{agent_id}' registrado no EventBus")

    def unregister_agent(self, agent_id: str):
        """Remove um agente do bus"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agente '{agent_id}' removido do EventBus")

    async def _dispatch_single_event(self, event: BaseEvent) -> List[BaseEvent]:
        """Dispara um evento único e retorna novos eventos gerados"""
        new_events: List[BaseEvent] = []
        if event.target:
            if ":" in event.target:
                # Target inter-agente: agent_id:component ou agent_id:other_agent:component
                parts = event.target.split(":", 2)
                if len(parts) == 2:
                    agent_id, component_name = parts
                    agent = self.agents.get(agent_id)
                    if agent and component_name in agent.components:
                        # Dispara para o componente específico
                        result_events = await agent.components[component_name]._safe_handle_event(event)
                        new_events.extend(result_events)
                    else:
                        # Target inválido ou agente não registrado, vai para dump
                        self.dump_component.store_message(event)
                else:
                    # Possivelmente forwarding, tratar como inválido por agora
                    self.dump_component.store_message(event)
            else:
                # Target agente - dispara para o agente diretamente
                agent = self.agents.get(event.target)
                if agent:
                    result_events = await agent.send_event(event)
                    new_events.extend(result_events)
                else:
                    # Agente não encontrado, vai para dump
                    self.dump_component.store_message(event)
        else:
            # Sem target, vai direto para dump
            self.dump_component.store_message(event)

        return new_events

    async def dispatch(self, events: List[BaseEvent]):
        """
        Dispara uma lista de eventos para os agentes registrados.
        Processa evento por evento e coleta novos eventos gerados pela handlers.
        Continua processamento até a fila de eventos estar vazia.

        Args:
            events: Lista de BaseEvent para despachar
        """
        pending_events = events.copy()

        while pending_events:
            current_event = pending_events.pop(0)
            new_events = await self._dispatch_single_event(current_event)
            pending_events.extend(new_events)

    def get_dumped_messages(self) -> List[BaseEvent]:
        """Retorna mensagens armazenadas no dump"""
        return self.dump_component.get_messages()

    def get_registered_agents(self) -> List[str]:
        """Retorna lista de IDs de agentes registrados"""
        return list(self.agents.keys())


# Demo simples do EventBus
async def demo():
    """Demonstração simples do EventBus"""
    print("🚀 Demo do EventBus Simples")
    print("=" * 30)

    # Import necessário
    try:
        from main import BaseEvent
    except ImportError:
        print("❌ Não foi possível importar BaseEvent (relativo)")
        return

    # Cria event bus
    bus = EventBus()

    # Simula agentes (sem componentes reais)
    class MockAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def set_external_bus(self, bus):
            pass  # Mock

        async def send_event(self, event):
            print(f"✅ Agente recebeu evento: {event.type} de {event.sender}")
            return []

    agent1 = MockAgent("agent1")
    bus.register_agent(agent1)

    # Cria eventos de exemplo
    events = [
        BaseEvent(sender="test", target="agent1", type="test_com_target"),
        BaseEvent(sender="test", target="agent2", type="test_target_inexistente"),  # Inexistente
        BaseEvent(sender="test", target="", type="test_sem_target"),  # Sem target
    ]

    print("📨 Disparando eventos...")
    await bus.dispatch(events)

    dumped = bus.get_dumped_messages()
    print(f"📦 Mensagens no dump: {len(dumped)}")
    for msg in dumped:
        print(f"  • {msg.type} -> {msg.target}")

    print("✅ Demo concluída!")


if __name__ == "__main__":
    asyncio.run(demo())
